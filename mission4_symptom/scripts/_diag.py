import sys, itertools; sys.path.insert(0,'.')
import numpy as np
from m4 import simulate as sim
sys.stdout.reconfigure(encoding='utf-8')
T={k:sim.PrefixTable.load(f'runs/{d}/dev_prefix.npz') for k,d in (('E001','E001_m3_baseline'),('E002','E002_prefix_uniform'))}
t=T['E001']; n=t.n_utt; step=np.concatenate([np.arange(1,k+1) for k in n]); gold=np.repeat(t.labels,n,axis=0)
rule=lambda p: sim.decide(p,force_top1=True)
# 사전확률만 쓰는 예측(항상 가장 흔한 증상 1개)
prior=np.zeros_like(gold); prior[:,t.labels.mean(0).argmax()]=1
print('사전확률 1개만:', round(float((np.add.reduceat(sim.moment_scores(prior,gold),t.offsets[:-1])/n).mean()),4))
ms={k:sim.moment_scores(rule(v.probs),gold) for k,v in T.items()}
print('step  n통화  E001   E002   사전')
pm=sim.moment_scores(prior,gold)
for a,b in ((1,1),(2,2),(3,3),(4,4),(5,5),(6,7),(8,10),(11,15),(16,20),(21,30),(31,200)):
    m=(step>=a)&(step<=b); print(f'{a:>2}-{b:<3} {m.sum():>6} {ms["E001"][m].mean():.3f}  {ms["E002"][m].mean():.3f}  {pm[m].mean():.3f}')
# 기대 Jaccard 최대화 규칙: 9종 독립 가정, 정답 집합 512가지를 모두 따져 확률 상위 k개 중 기대 점수가 가장 큰 k 선택
sets=np.array(list(itertools.product([0,1],repeat=9)),dtype=np.float32)[1:]   # 빈 집합 제외
def expj(p):
    out=np.zeros(p.shape,dtype=bool)
    for s in range(0,len(p),20000):
        q=np.clip(p[s:s+20000].astype(np.float64),1e-4,1-1e-4)
        logw=np.log(q)@sets.T+np.log(1-q)@(1-sets).T; w=np.exp(logw); w/=w.sum(1,keepdims=True)
        order=np.argsort(-q,axis=1); best=np.full(len(q),-1.0); bestk=np.ones(len(q),int)
        for k in range(1,5):
            pred=np.zeros(q.shape); np.put_along_axis(pred,order[:,:k],1,axis=1)
            inter=pred@sets.T; union=k+sets.sum(1)[None,:]-inter
            e=(w*inter/union).sum(1); up=e>best; best[up]=e[up]; bestk[up]=k
        rank=order.argsort(1); out[s:s+20000]=rank<bestk[:,None]
    return out
for k,v in T.items():
    r=sim.score_4_1(v,expj); print(k,'기대점수 최대화 규칙 4-1:',round(r['score'],4),'평균 개수',round(r['mean_labels'],2),r['by_third'])
print('--- 추가')
w=np.array([2.15,2.35,3.02,1.83,1.90,2.74,2.77,2.11,2.53])   # 학습 pos_weight
def unshift(p):
    z=np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6)))-np.log(w); return 1/(1+np.exp(-z))
for k,v in T.items():
    r=sim.score_4_1(v,lambda p: expj(unshift(p))); print(k,'pos_weight 보정 뒤 기대점수 최대화:',round(r['score'],4),'평균 개수',round(r['mean_labels'],2))
avg=sim.PrefixTable(t.call_ids,t.offsets,(T['E001'].probs+T['E002'].probs)/2,t.labels)
print('E001+E002 평균 4-1:',round(sim.score_4_1(avg,rule)['score'],4), sim.score_4_1(avg,rule)['by_third'])
for tau in (0.7,0.8,0.9):
    r=sim.score_4_2(avg,sim.halt_confidence(tau),rule); print(f'E001+E002 평균 4-2 τ={tau}: {r["score"]:.4f} = {r["macro_f1"]:.4f} x {r["mean_earliness"]:.4f}')
# 끝까지 들은 예측을 처음부터 알았다면(모델 천장), 정답 개수를 알았다면
last=np.repeat(rule(np.stack([v[-1] for v in (T['E001'].call(i) for i in range(len(t)))])),n,axis=0)
print('마지막 순간 예측을 모든 순간에 썼다면:',round(float((np.add.reduceat(sim.moment_scores(last,gold),t.offsets[:-1])/n).mean()),4))
