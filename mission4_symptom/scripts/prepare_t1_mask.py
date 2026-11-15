"""동결된 Fit 후보 검토 기록과 T1 마스크 계획/예상 노출 통계만 만든다. 모델 실행 없음."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from m4.data import load_cache, read_ids
from m4 import t1_loss_masking as t1
import prefix_experiment as runner


def prepare(out):
    for p, h in runner.INPUT_SHA256.items():
        if runner.digest(p) != h:
            raise ValueError("팀 캐시 또는 분할이 변경됨")
    for p, h in runner.BACKBONE_SHA256.items():
        if runner.digest(runner.BACKBONE / p) != h:
            raise ValueError("TAPT 초기 파일 변경")
    out = Path(out).resolve()
    if not out.is_relative_to((runner.OUTPUTS / "t1_loss_masking").resolve()):
        raise ValueError("T1 준비 산출물은 t1_loss_masking 아래에만 저장")
    calls = load_cache(runner.CALLS, read_ids(runner.FIT))
    records = t1.make_review(calls)
    manifest = {"version": t1.VERSION, "scope": "Fit only; no Dev/Validation performance used",
        "cache_sha256": t1.sha(runner.CALLS), "fit_ids_sha256": t1.sha(runner.FIT),
        "fit_ids": [c.call_id for c in calls], "review": records,
        "review_counts": {"candidate_related_utterance_screen": 255, "deeper_context_review": 30,
                          **dict(Counter(r["decision"] for r in records))},
        "rules": ["수동 검토된 Fit 허용 목록과 검토 근거 발화만 사용",
                  "오심 라벨0, 신고자 평서 진술, 현재 대상 환자 귀속 확인",
                  "단순 질문/부정/해소된 과거/다른 사람/구역질 단독/모호한 주체 제외",
                  "근거가 현재 Prefix에 있어야 하며 512토큰 잘림 뒤 근거는 제외",
                  "원래 B*9 분모 유지; 원본 라벨/입력/표본/Pos Weight 불변"],
        "regulations": "학습 손실 마스킹 명시 허용 회신 미확보. README 라벨별 전처리 금지와의 적용 범위 확인 필요"}
    tokenizer = AutoTokenizer.from_pretrained(runner.BACKBONE, local_files_only=True)
    tokenizer.truncation_side = "right"
    if not tokenizer.is_fast:
        raise ValueError("Fast Tokenizer가 필요합니다")
    plan = {r["call_id"]: t1.maskable_prefixes(next(c for c in calls if c.call_id == r["call_id"]), r, tokenizer)
            for r in records if r["decision"] == "accepted"}
    stats = t1.MaskStats()
    sampling_hashes = []
    lengths = [c.n_utt for c in calls]
    for epoch in (1, 2, 3):
        rng = np.random.default_rng([42, epoch])
        items = [(i, int(rng.integers(1, n + 1))) for i, n in enumerate(lengths) for _ in range(2)]
        order = rng.permutation(len(items))
        sampling_hashes.append(hashlib.sha256(np.array(items, dtype=np.int64).tobytes() + order.tobytes()).hexdigest())
        stats.start(epoch)
        for b in range(len(items) // 16):
            batch = [items[j] for j in order[b * 16:(b + 1) * 16]]
            labels = torch.tensor([calls[i].labels for i, _ in batch], dtype=torch.float32)
            ids, ns = [calls[i].call_id for i, _ in batch], [n for _, n in batch]
            stats.update(labels, ids, ns, t1.loss_mask(labels, ids, ns, plan))
        stats.finish()
    expected = stats.report()
    expected.update({"note": "기존 seed/샘플링을 CPU에서 재생한 예상치. 실제 학습 결과 아님",
        "sampling_sha256_by_epoch": sampling_hashes, "planned_optimizer_steps": 9855,
        "accepted_review_calls": len(plan), "maskable_calls_with_visible_evidence": sum(bool(v) for v in plan.values())})
    out = runner.reserve(out)
    runner.save(out / "manifest.json", manifest)
    runner.save(out / "expected_masking_stats.json", expected)
    print(json.dumps({"out": str(out), "review_counts": manifest["review_counts"], "expected": expected}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    prepare(ap.parse_args().out)
