"""E015 진단: 오심이 왜 낮은지 학습용 통화의 교차 학습 확률표(E011, 26,280통화)에서 본다. dev 는 쓰지 않는다."""
import json, re, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
from m4 import SYMPTOM_CLASSES as C, simulate as sim
from m4.data import load_cache
from train_halt_policy import merge_tables
for _s in (sys.stdout, sys.stderr): _s.reconfigure(encoding="utf-8", errors="replace")

t = merge_tables([HERE / f"runs/E011_oof/fold{k}/oof_prefix.npz" for k in range(5)], True)
calls = {c.call_id: c for c in load_cache(HERE / "runs/cache/train.jsonl", set(t.call_ids))}
calls = [calls[i] for i in t.call_ids]
Y = t.labels.astype(bool); last = t.probs[t.offsets[1:] - 1]
N, V = C.index("오심"), C.index("구토")
print("== 통화 전체를 들은 시점, 증상별 (임계 0.5 F1 / 임계를 정답 보고 고른 최대 F1 / 양성률)")
for j, c in enumerate(C):
    def f1(th):
        p = last[:, j] >= th; tp = (p & Y[:, j]).sum(); return 2 * tp / max(p.sum() + Y[:, j].sum(), 1)
    best = max(np.arange(0.05, 0.96, 0.01), key=f1)
    print(f"  {c:<5} {f1(0.5):.3f} / {f1(best):.3f} (임계 {best:.2f}) / {Y[:, j].mean():.3f}")
print("== 오심과 구토")
print(f"  오심 양성 {Y[:, N].sum()}, 구토 양성 {Y[:, V].sum()}, 둘 다 {(Y[:, N] & Y[:, V]).sum()}")
print(f"  P(오심|구토) {Y[Y[:, V], N].mean():.3f}  P(구토|오심) {Y[Y[:, N], V].mean():.3f}")
print(f"  오심 양성 중 증상이 오심 하나뿐 {(Y[Y[:, N]].sum(1) == 1).mean():.3f}")
mx = np.maximum.reduceat(t.probs[:, N], t.offsets[:-1])
print(f"  오심 양성 중 끝까지 0.5 를 못 넘는 통화 {(mx[Y[:, N]] < 0.5).mean():.3f}")
print(f"  오심 확률 평균: 오심만 {last[Y[:, N] & ~Y[:, V], N].mean():.3f} / 오심+구토 {last[Y[:, N] & Y[:, V], N].mean():.3f} / 구토만 {last[~Y[:, N] & Y[:, V], N].mean():.3f} / 둘 다 아님 {last[~Y[:, N] & ~Y[:, V], N].mean():.3f}")
print(f"  구토 확률 평균: 오심만 {last[Y[:, N] & ~Y[:, V], V].mean():.3f} / 오심+구토 {last[Y[:, N] & Y[:, V], V].mean():.3f} / 구토만 {last[~Y[:, N] & Y[:, V], V].mean():.3f}")
print("== 표현별 (통화 전체 전사에 있으면): 통화 비율 / 그중 오심 / 그중 구토 / 오심 양성 중 이 표현이 있는 비율")
full = [" ".join(x for x in c.texts if x) for c in calls]
pats = {"메스껍·메슥·매스": r"메스|메슥|매스꺼|매슥|미식|미슥", "울렁": r"울렁", "구역·구토감·토할 것": r"구역|구토감|토할 ?것|토할거|토할 거|토 ?나올|넘어올", "속이 안 좋": r"속이? ?안 ?좋|속이? ?불편|속이? ?이상", "오심(단어)": r"오심",
        "토했·구토·토를": r"토했|토하|구토|토를|토도|토해|오바이트|게워"}
hit = {}
for name, pat in pats.items():
    h = np.array([bool(re.search(pat, x)) for x in full]); hit[name] = h
    print(f"  {name:<14} {h.mean():.3f} / {Y[h, N].mean():.3f} / {Y[h, V].mean():.3f} / {h[Y[:, N]].mean():.3f}")
naus = hit["메스껍·메슥·매스"] | hit["울렁"] | hit["구역·구토감·토할 것"] | hit["오심(단어)"]
vom = hit["토했·구토·토를"]
print("== 메스꺼움 계열 표현(앞 넷 중 속 제외) x 구토 표현: 통화 수 / 오심 양성률 / 구토 양성률 / 오심 확률 평균")
for a in (1, 0):
    for b in (1, 0):
        s = (naus == a) & (vom == b)
        print(f"  메스꺼움 {a} 구토 {b}: {s.sum():>6} / {Y[s, N].mean():.3f} / {Y[s, V].mean():.3f} / {last[s, N].mean():.3f}")
s = Y[:, N] & ~naus
print(f"== 오심 양성인데 메스꺼움 계열 표현이 없는 통화 {s.sum()} ({s.sum() / Y[:, N].sum():.3f}). 그중 구토 표현 있음 {vom[s].mean():.3f}, 구토 라벨 {Y[s, V].mean():.3f}")
