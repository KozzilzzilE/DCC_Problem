"""제출용 model_train.ipynb 생성기.

노트북의 모든 셀은 실제로 실행된다. 학습 셀은 체크포인트가 이미 있으면
디스크에 기록된 학습 이력을 읽어 출력하고, 없으면 그 자리에서 학습한다.
따라서 노트북에 남는 출력은 전부 실제 실행 결과다.

    python mission1_gender/build_notebook.py            # 노트북 생성
    python mission1_gender/build_notebook.py --execute  # 생성 후 실행까지
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import nbformat as nbf

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = REPO_ROOT / "mission1_gender" / "model_train.ipynb"

CELLS: list[tuple[str, str]] = [
    ("md", """# Mission 1 — 신고자 성별 분류

담당: 김승윤

음성으로부터 신고자의 성별(남/여)을 분류한다. 같은 전처리 캐시 위에 ResNet50(2D CNN)과
Wav2Vec2(음성 특화 파인튜닝) 갈래를 올려 비교했고, **제출 모델은 dev(sliding) 기준으로 고른
`w2v2_full.pt`** 다 (dev 0.9901 / Validation 0.9843). 이 노트북은 그 모델의 학습 이력과
Validation 평가, 제출 규격 실행을 담는다.

**핵심 구조 — 2단 집계**

정답 `gender`는 통화 단위 라벨인데 통화 음성(평균 72.5초)에는 신고자와 119대원이
섞여 있다. 규칙상 추론 시에도 `startAt`/`endAt`/`speaker`를 쓸 수 있으므로:

1. `speaker == 1`(신고자) 발화 조각만 잘라 조각 단위로 이진 분류기를 학습
2. 한 통화의 조각별 확률을 평균(soft voting)해 통화의 남/여를 결정

통화당 신고자 조각이 평균 15.8개라, 조각 하나하나의 오류가 집계에서 상쇄된다."""),

    ("code", """import os, sys, json, time
from pathlib import Path

REPO = Path.cwd().parent if Path.cwd().name == "mission1_gender" else Path.cwd()
sys.path.insert(0, str(REPO / "mission1_gender"))
os.chdir(REPO)

import numpy as np
import torch

print("repo      :", REPO)
print("python    :", sys.version.split()[0])
print("torch     :", torch.__version__)
print("CUDA      :", torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
import torchvision; print("torchvision:", torchvision.__version__)"""),

    ("md", """## 1. 대회 규칙을 코드로 강제하기

Mission 1은 라벨링 데이터에서 `startAt`, `endAt`, `speaker`만 쓸 수 있다
(`gender`는 학습 타깃). `m1.labels`가 반환하는 자료구조에는 **그 외 필드가 아예
담기지 않는다.** `text`, `symptom`, `address` 등은 파싱 단계에서 버려지므로 하위
코드가 실수로라도 규칙을 위반할 수 없다."""),

    ("code", """from m1.labels import read_call, caller_utterances
import dataclasses, glob

sample_json = sorted(glob.glob("data/train/label/*.json"))[0]

raw = json.load(open(sample_json, encoding="utf-8"))
print("원본 JSON 의 최상위 키:")
print(" ", sorted(raw.keys()))

rec = read_call(sample_json)
print()
print("CallRecord 가 담는 필드:", [f.name for f in dataclasses.fields(rec)])
print("Utterance 가 담는 필드 :", [f.name for f in dataclasses.fields(rec.utterances[0])])
print()
print("전체 발화 %d개 중 신고자 발화 %d개" % (len(rec.utterances), len(caller_utterances(rec))))
print("첫 신고자 발화:", caller_utterances(rec)[0])
print()

# 금지 필드가 레코드 어디에도 남아 있지 않음을 확인
blob = repr(rec)
leaked = [k for k in ("text", "symptom", "address", "triage", "urgencyLevel") if k in blob]
print("레코드에 남은 금지 필드:", leaked or "없음")"""),

    ("md", """## 2. 데이터 실측

원천 오디오는 **8 kHz mono 16-bit** 다 (흔히 가정하는 16 kHz가 아니다). 16 kHz로
읽으면 없는 정보를 만들어내지 않으면서 계산량만 2배가 된다."""),

    ("code", """import soundfile as sf
import glob

wavs = sorted(glob.glob("data/train/audio/*.wav"))
info = sf.info(wavs[0])
print("wav 포맷: %d Hz, %d channel, %s" % (info.samplerate, info.channels, info.subtype))
print("통화 길이: %.1f 초" % info.duration)
print("Training 통화 수:", len(wavs))
print("Validation 통화 수:", len(glob.glob("data/val/audio/*.wav")))"""),

    ("md", """## 3. 조각 오디오 캐시

발화 조각을 꺼낼 때마다 통화 전체 wav를 다시 디코딩하면 46만 조각 × 약 1.1 MB의
읽기가 매 epoch 반복된다. 통화당 wav를 **한 번만** 읽어 신고자 조각을 미리 잘라
8 kHz int16으로 저장해 둔다.

원본 샘플레이트를 그대로 보관하므로 CNN 갈래(스펙트로그램)와 Wav2Vec2 갈래
(16 kHz 업샘플)가 같은 캐시를 공유한다 — 두 갈래의 속도 차이가 I/O가 아니라
모델에서 나오게 하려는 의도다."""),

    ("code", """from m1.cache import CacheIndex, build_cache, cache_size_bytes, default_workers

for split in ("train", "val"):
    if not (Path("cache") / split / "index.csv").exists():
        print("캐시 생성 중:", split)
        build_cache(f"data/{split}/label", f"data/{split}/audio", f"cache/{split}",
                    workers=default_workers(), progress=True)

train_index = CacheIndex.load("cache/train")
val_index = CacheIndex.load("cache/val")

for name, index in (("Training", train_index), ("Validation", val_index)):
    groups = index.by_call()
    lengths = np.array([r.length for r in index.rows]) / 8000.0
    n_per_call = np.array([len(v) for v in groups.values()])
    print("%s: %d 통화, %d 조각, %.2f GB" % (
        name, len(groups), len(index.rows), cache_size_bytes(index.cache_dir) / 1e9))
    print("   조각/통화 평균 %.1f | 조각 길이 평균 %.2fs p50 %.2fs p90 %.2fs | 총 %.1f 시간" % (
        n_per_call.mean(), lengths.mean(), np.percentile(lengths, 50),
        np.percentile(lengths, 90), lengths.sum() / 3600))"""),

    ("code", """from m1.aggregate import gender_to_target
import collections

groups = val_index.by_call()
dist = collections.Counter(rows[0].gender for rows in groups.values())
majority = max(dist.values()) / sum(dist.values())
print("Validation 통화 성별 분포:", dict(dist))
print("다수결 기준선 (통화 단위 Accuracy): %.4f" % majority)
print()
print("=> 모델이 이 값을 못 넘으면 아무것도 학습하지 못한 것이다.")"""),

    ("md", """## 4. 전처리 — log-Mel / MFCC 프런트엔드

설치된 `torch 2.13+cu130`에 맞는 torchaudio 빌드가 없어(cu130 채널 최대 2.11)
`torch.stft` + 자체 mel 필터뱅크로 직접 구현했다. 부수 효과로 피처 계산이 GPU에서
배치 단위로 돌아간다. 수치가 librosa와 일치함은 `tests/test_features.py`에서
대조 검증한다.

8 kHz 기준 `n_fft=1024`(128 ms)는 성별 판별의 주 단서인 F0(남 85–180 Hz /
여 165–255 Hz)를 7.8 Hz 해상도로 분해한다."""),

    ("code", """from m1.config import FeatureConfig
from m1.features import MelFrontend
from m1.datasets import to_waveform

cfg = FeatureConfig()
print("FeatureConfig:", cfg)
print("창 길이: %d samples = %.2f 초 -> %d 프레임" % (
    cfg.window_samples, cfg.window_samples / cfg.sample_rate, cfg.window_frames))

frontend = MelFrontend(cfg)

# 남성 통화와 여성 통화에서 각각 조각 하나씩 뽑아 스펙트로그램을 비교
picked = {}
for call_id, rows in train_index.by_call().items():
    g = rows[0].gender
    if g in ("M", "F") and g not in picked:
        longest = max(rows, key=lambda r: r.length)
        picked[g] = (call_id, longest)
    if len(picked) == 2:
        break

feats = {}
for g, (call_id, row) in picked.items():
    seg = train_index.load_segment(row)[: cfg.window_samples]
    wave = torch.from_numpy(to_waveform(seg, "resnet")).unsqueeze(0)
    feats[g] = frontend(wave)[0, 0].numpy()
    print("%s: call=%s  조각 %.2f초 -> 피처 %s" % (g, call_id, row.length / 8000, feats[g].shape))"""),

    ("code", """import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 3.6))
for ax, g in zip(axes, ("M", "F")):
    im = ax.imshow(feats[g], origin="lower", aspect="auto", cmap="magma")
    ax.set_title("caller gender = %s" % g)
    ax.set_xlabel("time frame (16 ms)")
    ax.set_ylabel("mel bin")
    fig.colorbar(im, ax=ax)
plt.suptitle("Normalised log-Mel spectrogram of one caller segment")
plt.tight_layout()
plt.show()"""),

    ("md", """## 5. 모델 — 제출 갈래 (Wav2Vec2-base 파인튜닝)

`facebook/wav2vec2-base` 백본에 mean-pooling + 선형 헤드. 입력은 8 kHz 원본을 16 kHz로
업샘플한 raw waveform 이다 (원본에 없던 4 kHz 이상 대역은 비어 있어 사전학습 도메인과
갭이 남지만, 실측으로는 스펙트로그램 CNN 보다 앞섰다). feature encoder 는 동결하고
트랜스포머와 헤드만 학습한다.

체크포인트에 `FeatureConfig`·갈래·보정 임계값·HF config 가 함께 저장돼, 아래처럼
`load_checkpoint` 하나로 학습과 동일한 전처리·결정 경계가 복원되고 허브 접속도 필요 없다.
비교용 ResNet50 갈래(`resnet_aug_m80.pt`, Validation 0.9819)는 폴백으로 둔다."""),

    ("code", """from m1.models import load_checkpoint, checkpoint_threshold

CKPT = Path("mission1_gender/ckpt/w2v2_full.pt")
model, branch, ckpt_cfg, payload = load_checkpoint(CKPT, device="cpu")
n_params = sum(p.numel() for p in model.parameters())
print("branch      :", branch, "|", payload["extra"].get("model_name"))
print("파라미터 수 :", "%.1fM" % (n_params / 1e6))
print("보정 임계값 :", checkpoint_threshold(payload), "(dev 에서 선택, 0.5 가 아님)")
print("HF config 동봉:", bool(payload["extra"].get("hf_config")), "-> 오프라인 로딩 가능")
print("head        :", model.head)

with torch.no_grad():
    dummy = torch.zeros(2, ckpt_cfg.window_samples * 2)   # 16 kHz
    print("\\n(B, samples@16k) %s -> logits %s" % (tuple(dummy.shape), tuple(model(dummy).shape)))"""),

    ("md", """## 6. 학습

**데이터 분할** — Training 폴더만 **통화 ID 기준** 90/10으로 나눠 dev를 만든다.
조각이 아니라 통화 단위로 나눠야 같은 화자가 train과 dev 양쪽에 들어가는 누수가
없다. Validation 폴더는 학습·모델선택에 일절 쓰지 않는다 (대회 규칙).

아래 셀은 체크포인트가 이미 있으면 기록된 학습 이력을 읽어 보여주고, 없으면 그
자리에서 학습한다. 표의 dev 는 학습 중 `center` 모드(조각당 창 1개) 값이고, 제출 모델
선정에 쓴 `sliding` dev 는 `m1.calibrate` 가 따로 잰다 (w2v2_full: 0.9901)."""),

    ("code", """HISTORY = CKPT.with_suffix(".history.json")

if not CKPT.exists():
    from m1.train import main as train_main
    train_main(["--branch", "w2v2", "--cache", "cache/train",
                "--out", str(CKPT), "--epochs", "3", "--lr", "3e-5",
                "--batch-size", "32", "--num-workers", "6"])

record = json.loads(HISTORY.read_text(encoding="utf-8"))

print("dev 다수결 기준선 : %.4f" % record["dev_majority_baseline"])
print("최고 dev 통화 Acc : %.4f" % record["best_dev_call_accuracy"])
print()
print("%-6s %-11s %-16s %-14s %-10s" % ("epoch", "train loss", "dev 통화 Acc", "dev 조각 Acc", "학습 시간"))
for row in record["history"]:
    print("%-6d %-11.4f %-16.4f %-14.4f %-10.0fs" % (
        row["epoch"], row["train_loss"], row["dev_call_accuracy"],
        row["dev_segment_accuracy"], row["train_seconds"]))"""),

    ("code", """hist = record["history"]
epochs = [r["epoch"] for r in hist]

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6))
axes[0].plot(epochs, [r["train_loss"] for r in hist], marker="o")
axes[0].set_title("training loss"); axes[0].set_xlabel("epoch"); axes[0].grid(alpha=.3)

axes[1].plot(epochs, [r["dev_call_accuracy"] for r in hist], marker="o", label="call-level")
axes[1].plot(epochs, [r["dev_segment_accuracy"] for r in hist], marker="s", label="segment-level")
axes[1].axhline(record["dev_majority_baseline"], ls="--", c="grey", label="majority baseline")
axes[1].set_title("dev accuracy"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].grid(alpha=.3)
plt.tight_layout(); plt.show()

gap = hist[-1]["dev_call_accuracy"] - hist[-1]["dev_segment_accuracy"]
print("조각 -> 통화 집계로 얻은 이득: +%.4f" % gap)"""),

    ("md", """## 7. Validation 평가

여기서 처음으로 Validation 폴더를 쓴다. 학습에도, 모델 선택에도 쓰지 않았다."""),

    ("code", """from m1.datasets import samples_from_rows
from m1.evaluate import majority_baseline, predict_segment_probs, score, suggested_workers, truth_from_samples
from m1.models import checkpoint_threshold, load_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
trained, branch, trained_cfg, payload = load_checkpoint(CKPT, device=device)
threshold = checkpoint_threshold(payload)

val_samples = samples_from_rows(val_index.rows)
val_truth = truth_from_samples(val_samples)

started = time.perf_counter()
probs = predict_segment_probs(trained, val_index, val_samples, trained_cfg, branch,
                              device, batch_size=128, mode="sliding",
                              num_workers=suggested_workers(branch))
elapsed = time.perf_counter() - started

metrics = score(val_samples, probs, val_truth, threshold)
print("결정 임계값 (dev 보정)  : %.3f" % threshold)
print("Validation 통화 Accuracy : %.4f" % metrics.call_accuracy)
print("Validation 조각 Accuracy : %.4f" % metrics.segment_accuracy)
print("다수결 기준선            : %.4f" % majority_baseline(val_truth))
print()
print("혼동행렬(통화 단위):", metrics.confusion)
print("성별별 정확도       :", {k: round(v, 4) for k, v in metrics.per_gender_accuracy.items()})
print()
print("%d 통화 / %d 조각 추론에 %.1f초 (%.1f ms/통화)" % (
    metrics.n_calls, metrics.n_segments, elapsed, elapsed * 1000 / metrics.n_calls))"""),

    ("code", """from m1.evaluate import call_probabilities

call_probs = call_probabilities(val_samples, probs)
values = np.array(list(call_probs.values()))
gold = np.array([gender_to_target(val_truth[c]) for c in call_probs])

plt.figure(figsize=(7, 3.6))
plt.hist(values[gold == 0], bins=40, alpha=.65, label="true: male")
plt.hist(values[gold == 1], bins=40, alpha=.65, label="true: female")
plt.axvline(threshold, ls="--", c="k", label="threshold %.3f" % threshold)
plt.xlabel("call-level mean P(female)"); plt.ylabel("calls")
plt.title("Call-level probability after soft voting")
plt.legend(); plt.tight_layout(); plt.show()

margin = np.abs(values - threshold)
print("결정 경계에서 0.1 이내인 애매한 통화: %d / %d (%.1f%%)" % (
    (margin < 0.1).sum(), len(values), 100 * (margin < 0.1).mean()))"""),

    ("md", """## 8. 갈래 비교 — CNN vs 음성 특화 파인튜닝

두 갈래가 같은 캐시·같은 조각 분할·같은 집계를 쓰므로 표에 남는 차이는 모델에서
나온 것이다. `m1.benchmark`가 생성한 결과를 읽어온다."""),

    ("code", """report = Path("mission1_gender/reports/comparison.json")
if report.exists():
    data = json.loads(report.read_text(encoding="utf-8"))
    print("Validation 다수결 기준선: %.4f\\n" % data["baseline"])
    for row in data["results"]:
        print("%-18s branch=%-7s params=%5.1fM  val_call_acc=%.4f  "
              "train=%.0fs/ep  infer=%.1fms/call  vram=%.0fMB" % (
            row["label"], row["branch"], row["n_params_m"], row["val_call_accuracy"],
            row["train_seconds_per_epoch"], row["inference_ms_per_call"], row["peak_vram_mb"]))
else:
    print("비교표가 아직 없습니다. 다음을 먼저 실행하세요:")
    print("  python -m m1.benchmark --ckpt mission1_gender/ckpt/w2v2_full.pt \\\\")
    print("                         --ckpt mission1_gender/ckpt/w2v2_full.pt")"""),

    ("md", """## 9. 제출 규격 확인

대회가 실제로 실행하는 명령을 그대로 한 번 돌려 CSV 형식을 확인한다.

```
python inference.py --audio_dir {wav} --label_dir {json} --ckpt_path {ckpt} --output ./outputs/mission1.csv
```

Mission 1 CSV: `[audio file name], [gender]`"""),

    ("code", """import subprocess, pandas as pd

cmd = [sys.executable, "inference.py",
       "--audio_dir", "./data/val/audio",
       "--label_dir", "./data/val/label",
       "--ckpt_path", str(CKPT),
       "--output", "./outputs/mission1.csv"]
print(" ".join(cmd))

result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
print("exit code:", result.returncode)
print(result.stdout[-800:])

df = pd.read_csv("outputs/mission1.csv")
print("\\n행 수:", len(df), "| 컬럼:", list(df.columns))
print(df.head())
print("\\ngender 분포:", df["gender"].value_counts().to_dict())"""),

    ("md", """## 10. 정리

**결과 요약**

- 조각 단위 정확도보다 **통화 단위 정확도가 뚜렷하게 높다.** 통화당 신고자 조각이
  평균 15.8개라, 개별 조각의 오류가 soft voting에서 상쇄된다. 조각 하나의 성능을
  올리는 것보다 집계 단위를 통화로 맞춘 설계가 더 크게 기여했다.
- 8 kHz 전화 음성을 16 kHz 로 올려 넣어도 Wav2Vec2 가 다수결 기준선(0.538)을 크게
  넘어선다. 스펙트로그램 CNN(ResNet50, 0.9819)보다 0.24%p 앞서고, dev 에서도 같은
  방향(+0.28%p)이라 노이즈로 보지 않았다. 세 갈래가 같은 통화(1.2%)에서 틀리는 것이
  이 데이터의 상한이다 — 자세한 오류 분석은 `mission1_gender/README.md`.

**사회안전 관점의 시사점**

119 신고접수에서 신고자 속성을 자동으로 파악하면, 접수 요원이 통화 초반 수 초
안에 상황을 분류하는 데 도움이 된다. 다만 성별 추정은 **보조 정보로만** 쓰여야
한다. 위 확률 분포에서 보듯 결정 경계 근처의 애매한 통화가 존재하고, 이런 사례에
자동 판정을 강하게 신뢰하면 오히려 대응이 지연될 수 있다. 확률값을 그대로 노출해
불확실성을 함께 전달하는 편이 안전하다.

**한계**

- `gender`는 통화 단위 단일 라벨이라 신고자가 도중에 바뀌는 통화는 다루지 못한다.
- 학습·평가 데이터가 전량 서울·Mobile이라 다른 지역이나 유선 통화로의 일반화는
  검증되지 않았다."""),
]


def build() -> Path:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(body) if kind == "md" else nbf.v4.new_code_cell(body)
        for kind, body in CELLS
    ]
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": sys.version.split()[0]},
    }
    NOTEBOOK.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NOTEBOOK)
    return NOTEBOOK


def execute(timeout: int) -> None:
    subprocess.run(
        [sys.executable, "-m", "nbconvert", "--to", "notebook", "--execute", "--inplace",
         f"--ExecutePreprocessor.timeout={timeout}", str(NOTEBOOK)],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--timeout", type=int, default=7200)
    args = parser.parse_args()

    path = build()
    print(f"wrote {path} ({len(CELLS)} cells)")
    if args.execute:
        execute(args.timeout)
        print(f"executed {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
