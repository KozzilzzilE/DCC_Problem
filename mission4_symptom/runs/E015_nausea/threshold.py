"""E015 대책 후보 2: 4-2 중단 시점 예측에서 오심만 임계를 낮춘다.
임계는 학습용 통화(교차 학습 확률표 + 현재 학습형 정책의 중단 시점)에서 4-2 점수가 가장 큰 값으로 고르고,
dev 2,920통화(3개 평균, 현재 제출 후보)에는 그 값 하나만 적용해 판정한다."""
import dataclasses, json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
from m4 import SYMPTOM_CLASSES as C, halting as hl, simulate as sim
from m4.data import load_cache
from train_halt_policy import merge_tables, build
for _s in (sys.stdout, sys.stderr): _s.reconfigure(encoding="utf-8", errors="replace")
N = C.index("오심")
shift = np.log(json.loads((HERE / "runs/E002_prefix_uniform/run_config.json").read_text(encoding="utf-8"))["pos_weight"])
policy = hl.HaltPolicy.load(HERE / "runs/E012_halt_policy/ens3/policy.json")
fit_t = merge_tables([HERE / f"runs/E011_oof/fold{k}/oof_prefix.npz" for k in range(5)], True)
dev_t = merge_tables([HERE / p for p in ("runs/E001_m3_baseline/dev_prefix.npz", "runs/E002_prefix_uniform/dev_prefix.npz", "runs/E007_prefix_uniform_s43/dev_prefix.npz")], False)
calls = {c.call_id: c for c in load_cache(HERE / "runs/cache/train.jsonl", set(fit_t.call_ids) | set(dev_t.call_ids))}

def with_thr(data, table, th, cls=N):
    pred = data.pred.copy(); pred[:, cls] = table.probs[:, cls] >= th
    empty = ~pred.any(axis=1); pred[empty] = data.pred[empty]            # 비면 원래 예측(가장 높은 1개)
    return dataclasses.replace(data, pred=pred)

def cls_f1(data, halts, cls=N):
    p = data.pred[data.offsets[:-1] + halts - 1][:, cls]; y = data.labels[:, cls]
    return 2 * (p & y).sum() / max(p.sum() + y.sum(), 1)

fit, dev = build(fit_t, calls, shift), build(dev_t, calls, shift)
hf, hd = policy.halts(fit), policy.halts(dev)
ths = np.round(np.arange(0.20, 0.51, 0.02), 2)
print("== 학습용 통화에서 오심 임계별 4-2 (중단 시점은 현재 정책 그대로)")
res = {}
for th in ths:
    r = hl.score_halts(with_thr(fit, fit_t, th), hf); res[th] = r["score"]
    print(f"  {th:.2f}: {r['score']:.4f} (F1 {r['macro_f1']:.4f}, 오심 F1 {cls_f1(with_thr(fit, fit_t, th), hf):.3f})")
best = max(ths, key=lambda h: res[h])
# 통화 단위 5겹: 4겹에서 고른 임계를 남은 겹에 적용했을 때의 이득 (선택이 안정적인지)
fold = np.random.default_rng(3).permutation(len(fit)) % 5
parts = {th: hl._score_parts(with_thr(fit, fit_t, th), hf) for th in ths}
def sc(th, mask): return hl._weighted_scores(parts[th], mask.astype(float))[0]
gains = []
for k in range(5):
    pick = max(ths, key=lambda h: sc(h, fold != k)); gains.append(sc(pick, fold == k) - sc(0.5, fold == k))
print(f"  고른 임계 {best:.2f}, 학습용 이득 {res[best] - res[0.5]:+.4f}, 겹 밖 이득 평균 {np.mean(gains):+.4f} (겹별 {', '.join(f'{g:+.4f}' for g in gains)})")

print("== dev 판정 (3개 평균, 현재 제출 후보)")
base = hl.score_halts(dev, hd); new_d = with_thr(dev, dev_t, best); new = hl.score_halts(new_d, hd)
W = np.random.default_rng(2026).multinomial(len(dev), np.full(len(dev), 1 / len(dev)), size=2000).astype(float)
d = hl._weighted_scores(hl._score_parts(new_d, hd), W) - hl._weighted_scores(hl._score_parts(dev, hd), W)
print(f"  현재 {base['score']:.4f} (F1 {base['macro_f1']:.4f}, 오심 F1 {cls_f1(dev, hd):.3f}) → 오심 임계 {best:.2f}: {new['score']:.4f} (F1 {new['macro_f1']:.4f}, 오심 F1 {cls_f1(new_d, hd):.3f})")
print(f"  차이 {new['score'] - base['score']:+.4f}  95% [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  P(>0) {(d > 0).mean():.3f}")
json.dump({"fit_scores": {str(k): v for k, v in res.items()}, "picked": float(best), "fit_gain": res[best] - res[0.5], "cv_gains": gains,
           "dev_base": base, "dev_new": new, "dev_diff": new["score"] - base["score"], "dev_ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]},
          open(HERE / "runs/E015_nausea/threshold.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)
