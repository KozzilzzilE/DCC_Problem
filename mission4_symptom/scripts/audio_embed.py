"""발화 단위 음성 임베딩을 만든다 (frozen 음성 모델, 미세조정 없음).

    python scripts/audio_embed.py --ids runs/E020_audio/sample_ids.txt \
        --audio-dir ../../../data/train/audio --out runs/E020_audio/emb.npz

    # 점검: 20통화, 템플릿 파형과 바이트 단위 비교 + 묶음/한 개씩 코사인
    python scripts/audio_embed.py --ids runs/E020_audio/sample_ids.txt --limit 20 \
        --audio-dir ../../../data/train/audio --out runs/E020_audio/check20/emb.npz \
        --check-template --check-utts 50

파형은 추론 템플릿(inference.py)과 같은 방식으로 만든다.
  * load_wav       = inference.load_waveform 의 soundfile 경로
                     (soundfile always_2d=True → 채널 평균 → scipy resample_poly 로 16kHz float32)
  * slice_template = inference.Sample._slice
                     (반올림 round(ms / 1000 * 16000), 끝은 파형 길이로 자름, b <= a 면 빈 배열)
  즉 Predictor 가 실제로 받는 utt.audio 와 같은 배열에서 출발한다. --check-template 이면 이 둘이
  템플릿 함수 출력과 완전히 같은지 처리한 통화 전부에서 확인한다.

발화 선택 규칙(초안과 같음): 화자(기본 신고자 1)만, 템플릿 조각 길이가 --min-sec 미만이면 버림,
--max-sec 보다 길면 조각의 가운데 --max-sec 만 씀, 통화당 앞에서부터 --max-utts 개까지(버린 발화는 세지 않음).

패딩 처리: WavLM-base-plus 는 feat_extract_norm="group" 이라 첫 합성곱 층의 GroupNorm 이 시간축 전체
(0 패딩 포함)로 평균·분산을 낸다. attention_mask 로도 이 영향은 막지 못한다.
  --batching single (기본) : 발화를 한 개씩 넣는다. 패딩이 전혀 없고, 스트리밍 추론에서 Predictor 가
                             발화를 한 개씩 받아 넣는 것과도 같다.
  --batching sorted        : 통화 묶음(--chunk-calls)마다 길이순 정렬 → 패딩 포함 길이 합 <= --batch-sec 로 묶음.
2026-11-13 20통화 점검(앞 50발화)에서 sorted 의 한 개씩 대비 코사인이 최소 0.776 / 평균 0.987 로
기준 0.999 에 못 미쳐 기본값을 single 로 바꿨다 (runs/E020_audio/check20/emb.npz.meta.json).
--check-utts N 이면 앞 N개 발화로 "sorted 방식" / "N개 한 묶음" 결과를 한 개씩 넣은 결과와 코사인으로
비교해 메타에 남긴다.

출력: 발화마다 --layers 의 은닉층(hidden_states 튜플 인덱스. 0=임베딩 출력, k=k번째 트랜스포머 층 출력)을
패딩 프레임을 뺀 프레임 평균으로 줄여 (층 수, 768) float16 으로 저장한다. 2단계(누적 예측)에서 그대로
쓸 수 있도록 발화 단위로 남긴다.
  emb      (N, L, 768) float16
  call     (N,) int32   call_ids 의 위치
  utt      (N,) int32   통화 안 발화 번호 (템플릿 Utterance.index 와 같음)
  n_samples(N,) int32   템플릿 조각 길이(자르기 전, 16kHz 표본 수)
  used     (N,) int32   실제로 넣은 길이 (가운데 자르기 후)
  call_ids (C,) str     --ids 를 걸러 캐시에 있는 통화 (캐시 파일 순서)
  layers   (L,) int32
행은 (call, utt) 순으로 정렬해 저장한다. 메타는 <out>.meta.json.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from m4.data import Call, load_cache, read_ids  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    if hasattr(_s, "reconfigure"):
        _s.reconfigure(encoding="utf-8", errors="replace")

TARGET_SR = 16000


# ----------------------------------------------------------------------------- 파형 (템플릿과 같게)
def load_wav(path: Path) -> np.ndarray:
    """inference.load_waveform 의 soundfile 경로를 그대로 옮긴 것 (수정 금지 구역의 동작과 맞춤)."""
    import soundfile as sf
    wav, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        from math import gcd
        from scipy.signal import resample_poly
        g = gcd(int(sr), int(TARGET_SR))
        wav = resample_poly(wav, TARGET_SR // g, sr // g).astype(np.float32)
    return np.ascontiguousarray(wav, dtype=np.float32)


def slice_template(wav: np.ndarray, start_ms: int, end_ms: int) -> np.ndarray:
    """inference.Sample._slice 와 같은 구간 (반올림). 초안의 int() 버림은 1표본씩 어긋날 수 있었다."""
    a = max(0, int(round(start_ms / 1000.0 * TARGET_SR)))
    b = min(len(wav), int(round(end_ms / 1000.0 * TARGET_SR)))
    if b <= a:
        return np.zeros(0, dtype=np.float32)
    return wav[a:b]


def center_crop(x: np.ndarray, cap: int) -> np.ndarray:
    """cap 표본보다 길면 가운데 cap 표본만. Predictor 가 utt.audio 를 받아 같은 일을 할 수 있다."""
    if len(x) <= cap:
        return x
    a = len(x) // 2 - cap // 2
    return x[a:a + cap]


def extract_call(call: Call, ci: int, audio_dir: Path, want: set, min_len: int, cap: int,
                 max_utts: int) -> Tuple[int, List[tuple], bool]:
    """한 통화의 wav 를 읽어 쓸 발화 조각만 돌려준다: (통화 위치, [(발화 번호, 원래 길이, 조각)], wav 있음)."""
    path = audio_dir / f"{call.call_id}.wav"      # 템플릿 _resolve_audio 의 stem 재탐색과 같은 파일
    if not path.exists():
        return ci, [], False
    wav = load_wav(path)
    out = []
    for ui, (s, e, sp) in enumerate(zip(call.starts, call.ends, call.speakers)):
        if sp not in want or len(out) >= max_utts:
            continue
        piece = slice_template(wav, s, e)
        if len(piece) < min_len:
            continue
        out.append((ui, len(piece), np.array(center_crop(piece, cap), dtype=np.float32, copy=True)))
    return ci, out, True


# ----------------------------------------------------------------------------- 음성 모델
class Embedder:
    def __init__(self, model_name: str, device: str, layers: List[int]):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.device = torch.device(device)
        # fp32 그대로 쓴다 (autocast 를 쓰면 GroupNorm 등이 fp16 이 되어 비교 기준이 흔들린다).
        # cuDNN 합성곱의 TF32 기본값도 꺼서 묶음/한 개씩 비교에 정밀도 잡음이 섞이지 않게 한다.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = AutoModel.from_pretrained(model_name).to(self.device).eval()
        self.layers = list(layers)
        self.dim = int(self.model.config.hidden_size)
        n_hidden = int(self.model.config.num_hidden_layers) + 1
        assert all(0 <= l < n_hidden for l in self.layers), f"층 번호는 0..{n_hidden - 1}"

    def run(self, pieces: List[np.ndarray]) -> np.ndarray:
        """조각 목록을 한 묶음으로 넣는다 (오른쪽 0 패딩 + attention_mask). -> (B, L, D) float32."""
        torch = self.torch
        n = max(len(x) for x in pieces)
        batch = np.zeros((len(pieces), n), dtype=np.float32)
        mask = np.zeros((len(pieces), n), dtype=np.int64)
        for i, x in enumerate(pieces):
            batch[i, :len(x)] = x
            mask[i, :len(x)] = 1
        with torch.inference_mode():
            x = torch.from_numpy(batch).to(self.device)
            m = torch.from_numpy(mask).to(self.device)
            out = self.model(input_values=x, attention_mask=m, output_hidden_states=True)
            hs = out.hidden_states
            # 패딩이 아닌 프레임 수: 모델이 내부 마스크를 만들 때 쓰는 것과 같은 함수
            lens = self.model._get_feat_extract_output_lengths(
                torch.tensor([len(y) for y in pieces], device=self.device)).long()
            T = hs[0].shape[1]
            fm = (torch.arange(T, device=self.device)[None, :] < lens[:, None]).float()   # (B, T)
            denom = fm.sum(1, keepdim=True).clamp(min=1)
            pooled = [(hs[l].float() * fm[:, :, None]).sum(1) / denom for l in self.layers]
            res = torch.stack(pooled, dim=1)                                               # (B, L, D)
        return res.cpu().numpy()

    def run_sorted(self, pieces: List[np.ndarray], budget: int, max_batch: int) -> np.ndarray:
        """파이프라인의 묶음 방식: 길이 내림차순 정렬 → 패딩 포함 길이 합 <= budget 로 묶음. 입력 순서로 돌려준다."""
        order = sorted(range(len(pieces)), key=lambda i: -len(pieces[i]))
        out = np.zeros((len(pieces), len(self.layers), self.dim), dtype=np.float32)
        i = 0
        while i < len(order):
            longest = len(pieces[order[i]])            # 정렬했으므로 묶음의 첫 조각이 가장 길다
            k = max(1, min(max_batch, budget // max(longest, 1)))
            idx = order[i:i + k]
            out[idx] = self.run([pieces[j] for j in idx])
            i += k
        return out

    def run_single(self, pieces: List[np.ndarray]) -> np.ndarray:
        """한 개씩 넣는다 (패딩 없음). 입력 순서로 (N, L, D) 를 돌려준다."""
        out = np.zeros((len(pieces), len(self.layers), self.dim), dtype=np.float32)
        for i, p in enumerate(pieces):
            out[i] = self.run([p])[0]
        return out


def cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, L, D) 두 개의 발화·층별 코사인 -> (N, L)."""
    num = (a * b).sum(-1)
    den = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return num / np.maximum(den, 1e-12)


def check_batching(emb: Embedder, pieces: List[np.ndarray], budget: int, max_batch: int) -> dict:
    """묶음 결과와 한 개씩 넣은 결과의 코사인. 층별 최소·평균을 돌려준다.

    묶음 안 순서는 결과에 영향이 없다(패딩 길이만 영향). 그래서 "N개 한 묶음"은 정렬 여부와 무관하다.
    """
    single = emb.run_single(pieces)
    variants = {
        "sorted_budget": emb.run_sorted(pieces, budget, max_batch),   # --batching sorted 방식
        "one_batch": emb.run(pieces),                                 # N개를 한 묶음에
    }
    lens = np.array([len(p) for p in pieces])
    res = {"n_utts": len(pieces), "len_sec_min": float(lens.min() / TARGET_SR),
           "len_sec_max": float(lens.max() / TARGET_SR), "layers": emb.layers}
    for name, v in variants.items():
        c = cosine_rows(v, single)
        res[name] = {"min": float(c.min()), "mean": float(c.mean()),
                     "min_by_layer": [float(x) for x in c.min(0)],
                     "mean_by_layer": [float(x) for x in c.mean(0)]}
    return res


def check_template(calls: List[Call], audio_dir: Path, want: set, min_len: int, cap: int,
                   max_utts: int) -> dict:
    """우리 파형·조각이 inference.py 의 load_waveform / Sample._slice 와 완전히 같은지 확인한다."""
    import inference as tpl
    n_calls = n_utts = bad_wav = bad_slice = 0
    for call in calls:
        path = audio_dir / f"{call.call_id}.wav"
        if not path.exists():
            continue
        ours = load_wav(path)
        theirs = tpl.load_waveform(str(path))
        n_calls += 1
        if ours.dtype != theirs.dtype or ours.shape != theirs.shape or not np.array_equal(ours, theirs):
            bad_wav += 1
        sample = tpl.Sample(label_file=f"{call.call_id}.json", audio_path=str(path))
        for s, e, sp in zip(call.starts, call.ends, call.speakers):
            n_utts += 1
            if not np.array_equal(slice_template(ours, s, e), sample._slice(s, e)):
                bad_slice += 1
        sample.release()
    # 초안(버림)과 템플릿(반올림)의 경계가 다른 발화 비율도 같이 적는다
    floor_diff = sum(1 for c in calls for s, e in zip(c.starts, c.ends)
                     if int(s * TARGET_SR / 1000) != int(round(s / 1000.0 * TARGET_SR))
                     or int(e * TARGET_SR / 1000) != int(round(e / 1000.0 * TARGET_SR)))
    return {"calls": n_calls, "utts": n_utts, "wav_mismatch": bad_wav, "slice_mismatch": bad_slice,
            "draft_floor_boundary_differs": floor_diff}


# ----------------------------------------------------------------------------- 메인
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--calls", default="runs/cache/train.jsonl")
    ap.add_argument("--ids", help="통화 목록 파일 (없으면 캐시 전체)")
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="microsoft/wavlm-base-plus",
                    help="frozen 음성 모델. 파라링귀스틱(말투·호흡·목소리 상태)에 강한 것을 기본으로 둔다")
    ap.add_argument("--layers", default="4,8,12", help="저장할 hidden_states 인덱스 (0=임베딩, 12=마지막 층)")
    ap.add_argument("--speakers", default="1", help="임베딩할 화자. '1'=신고자만, '01'=둘 다")
    ap.add_argument("--max-sec", type=float, default=6.0, help="발화당 최대 길이. 길면 가운데를 쓴다")
    ap.add_argument("--min-sec", type=float, default=0.3, help="이보다 짧은 발화는 건너뛴다")
    ap.add_argument("--max-utts", type=int, default=40, help="통화당 최대 발화 수 (앞에서부터)")
    ap.add_argument("--batching", choices=("single", "sorted"), default="single",
                    help="single=한 개씩(패딩 없음, 기본), sorted=길이순 묶음 (점검에서 기준 미달)")
    ap.add_argument("--batch-sec", type=float, default=60.0, help="sorted: 한 묶음의 패딩 포함 길이 합 (초)")
    ap.add_argument("--max-batch", type=int, default=256, help="한 묶음 최대 발화 수")
    ap.add_argument("--chunk-calls", type=int, default=100, help="이만큼 통화를 모아 길이순 정렬 후 묶는다")
    ap.add_argument("--workers", type=int, default=4, help="wav 읽기 스레드 수")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=0, help="점검용. 통화 수 제한")
    ap.add_argument("--check-utts", type=int, default=0, help="앞 N개 발화로 묶음/한 개씩 코사인 비교")
    ap.add_argument("--check-template", action="store_true", help="inference.py 파형·조각과 완전 일치 확인")
    args = ap.parse_args()

    want = {int(c) for c in args.speakers}
    layers = [int(x) for x in args.layers.split(",")]
    calls = load_cache(HERE / args.calls, read_ids(HERE / args.ids) if args.ids else None)
    if args.limit:
        calls = calls[:args.limit]
    audio_dir = Path(args.audio_dir)
    if not audio_dir.is_absolute():
        audio_dir = HERE / audio_dir
    min_len = int(round(args.min_sec * TARGET_SR))
    cap = int(round(args.max_sec * TARGET_SR))
    budget = int(args.batch_sec * TARGET_SR)

    t_start = time.time()
    emb = Embedder(args.model, args.device, layers)
    print(f"통화 {len(calls)}건, 모델 {args.model} ({emb.dim}차원), 층 {layers}, 화자 {sorted(want)}", flush=True)

    meta_check = {}
    if args.check_template:
        meta_check["template"] = check_template(calls, audio_dir, want, min_len, cap, args.max_utts)
        print("템플릿 비교:", json.dumps(meta_check["template"], ensure_ascii=False), flush=True)

    out_emb, out_call, out_utt, out_n, out_used = [], [], [], [], []
    missing: List[str] = []
    check_pool: List[np.ndarray] = []
    t0 = time.time()
    chunks = [list(range(i, min(i + args.chunk_calls, len(calls))))
              for i in range(0, len(calls), args.chunk_calls)]

    def submit(ex, chunk):
        return [ex.submit(extract_call, calls[ci], ci, audio_dir, want, min_len, cap, args.max_utts)
                for ci in chunk]

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = submit(ex, chunks[0]) if chunks else []
        for k in range(len(chunks)):
            results = [f.result() for f in futs]
            if k + 1 < len(chunks):                 # 다음 묶음 wav 를 GPU 계산과 겹쳐 읽는다
                futs = submit(ex, chunks[k + 1])
            pieces, keys = [], []
            for ci, items, ok in results:
                if not ok:
                    missing.append(calls[ci].call_id)
                for ui, n_raw, piece in items:
                    pieces.append(piece)
                    keys.append((ci, ui, n_raw))
            if args.check_utts and len(check_pool) < args.check_utts:
                check_pool.extend(pieces[:args.check_utts - len(check_pool)])
            if pieces:
                vec = (emb.run_single(pieces) if args.batching == "single"
                       else emb.run_sorted(pieces, budget, args.max_batch))
                out_emb.append(vec.astype(np.float16))
                out_call.extend(k_[0] for k_ in keys)
                out_utt.extend(k_[1] for k_ in keys)
                out_n.extend(k_[2] for k_ in keys)
                out_used.extend(len(p) for p in pieces)
            done += len(results)
            if done % 500 < args.chunk_calls or done == len(calls):
                el = time.time() - t0
                print(f"  {done}/{len(calls)} 통화, 발화 {len(out_call)}, {el:.0f}s", flush=True)
    embed_seconds = time.time() - t0

    if args.check_utts:
        meta_check["batching"] = check_batching(emb, check_pool, budget, args.max_batch)
        print("묶음/한 개씩 코사인:", json.dumps(meta_check["batching"], ensure_ascii=False), flush=True)

    L, D = len(layers), emb.dim
    E = np.concatenate(out_emb, axis=0) if out_emb else np.zeros((0, L, D), np.float16)
    call_arr = np.asarray(out_call, np.int32)
    utt_arr = np.asarray(out_utt, np.int32)
    order = np.lexsort((utt_arr, call_arr))           # (call, utt) 순으로 정렬해 저장
    n_per_call = np.bincount(call_arr, minlength=len(calls))
    out = Path(args.out)
    if not out.is_absolute():
        out = HERE / out
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, emb=E[order], call=call_arr[order], utt=utt_arr[order],
                        n_samples=np.asarray(out_n, np.int32)[order],
                        used=np.asarray(out_used, np.int32)[order],
                        call_ids=np.asarray([c.call_id for c in calls]),
                        layers=np.asarray(layers, np.int32))
    meta = {"model": args.model, "dim": D, "layers": layers,
            "layers_note": "hidden_states 튜플 인덱스 (0=임베딩 출력, k=k번째 트랜스포머 층 출력)",
            "pooling": "패딩 제외 프레임 평균", "precision": "fp32 계산, float16 저장",
            "speakers": sorted(want), "max_sec": args.max_sec, "min_sec": args.min_sec,
            "max_utts": args.max_utts, "batching": args.batching,
            "batch_sec": args.batch_sec, "max_batch": args.max_batch,
            "chunk_calls": args.chunk_calls, "slice": "inference.Sample._slice 와 같은 반올림",
            "n_calls": len(calls), "n_utts": int(len(E)), "missing_wav_calls": len(missing),
            "calls_with_zero_utts": int((n_per_call == 0).sum()),
            "utts_per_call_mean": float(n_per_call.mean()) if len(calls) else 0.0,
            "audio_hours_fed": float(np.sum(out_used) / TARGET_SR / 3600) if out_used else 0.0,
            "embed_seconds": round(embed_seconds, 1), "total_seconds": round(time.time() - t_start, 1),
            "check": meta_check}
    Path(str(out) + ".meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "check"}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
