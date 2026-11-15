"""E015 대책 후보: 오심 확률을 2단계 모델(9종 확률 + 표현 유무 + 길이)로 다시 내면 나아지는가.
학습용 통화의 교차 학습 확률표 안에서 통화 단위 5겹 교차검증으로만 본다 (dev 는 통과했을 때만 연다)."""
import re, sys
from pathlib import Path
import numpy as np, lightgbm as lgb
from sklearn.metrics import average_precision_score, roc_auc_score
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
from m4 import SYMPTOM_CLASSES as C
from m4.data import load_cache
from train_halt_policy import merge_tables
for _s in (sys.stdout, sys.stderr): _s.reconfigure(encoding="utf-8", errors="replace")
t = merge_tables([HERE / f"runs/E011_oof/fold{k}/oof_prefix.npz" for k in range(5)], True)
calls = {c.call_id: c for c in load_cache(HERE / "runs/cache/train.jsonl", set(t.call_ids))}
calls = [calls[i] for i in t.call_ids]
N = C.index("오심"); Y = t.labels.astype(bool)
PATS = [r"메스|메슥|매스꺼|매슥|미식|미슥", r"울렁", r"구역|구토감|토할 ?것|토할거|토할 거|토 ?나올|넘어올", r"속이? ?안 ?좋|속이? ?불편|속이? ?이상", r"오심",
        r"토했|토하|구토|토를|토도|토해|오바이트|게워", r"어지|빙빙|핑", r"머리|두통", r"배가|복통|배 ?아", r"술|음주|마셨", r"먹고|드시고|체한|체했|식사|음식", r"약을|약 ?먹|복용"]
def maxf1(p, y):
    o = np.argsort(-p); tp = np.cumsum(y[o]); return float((2 * tp / (np.arange(1, len(p) + 1) + y.sum())).max())
def f1(p, y, th=0.5):
    q = p >= th; return float(2 * (q & y).sum() / max(q.sum() + y.sum(), 1))
for name, steps in (("통화 전체 시점", None), ("7번째 발화 시점", 7)):
    idx = t.offsets[1:] - 1 if steps is None else np.minimum(t.offsets[:-1] + steps - 1, t.offsets[1:] - 1)
    P = t.probs[idx]; z = np.log(np.clip(P, 1e-6, 1 - 1e-6) / np.clip(1 - P, 1e-6, 1))
    texts = [" ".join(x for x in (c.texts if steps is None else c.texts[:steps]) if x) for c in calls]
    K = np.array([[len(re.findall(p, x)) for p in PATS] for x in texts], dtype=np.float32)
    L = np.array([[len(x), min(len(c.texts), steps or 10 ** 9)] for x, c in zip(texts, calls)], dtype=np.float32)
    y = Y[:, N]; fold = np.random.default_rng(7).permutation(len(y)) % 5
    print(f"== {name}: 오심 (원래 확률) AUC {roc_auc_score(y, P[:, N]):.4f}  AP {average_precision_score(y, P[:, N]):.4f}  F1@0.5 {f1(P[:, N], y):.4f}  최대 F1 {maxf1(P[:, N], y):.4f}")
    for tag, X in (("9종 확률만", z), ("+ 표현·길이", np.hstack([z, K, L]))):
        out = np.zeros(len(y))
        for k in range(5):
            tr, te = fold != k, fold == k
            m = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=50, subsample=0.8,
                                   subsample_freq=1, colsample_bytree=0.8, verbose=-1, n_jobs=4).fit(X[tr], y[tr])
            out[te] = m.predict_proba(X[te])[:, 1]
        ths = np.arange(0.05, 0.6, 0.01)          # 임계는 다른 겹에서 고른다
        f = np.mean([f1(out[fold == k], y[fold == k], max(ths, key=lambda h: f1(out[fold != k], y[fold != k], h))) for k in range(5)])
        f0 = np.mean([f1(P[fold == k, N], y[fold == k], max(ths, key=lambda h: f1(P[fold != k, N], y[fold != k], h))) for k in range(5)])
        print(f"  2단계 {tag:<10} AUC {roc_auc_score(y, out):.4f}  AP {average_precision_score(y, out):.4f}  최대 F1 {maxf1(out, y):.4f}  임계를 다른 겹에서 고른 F1 {f:.4f} (원래 확률로 같은 방식 {f0:.4f})")
