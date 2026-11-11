import sys, json; sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from m4 import simulate as sim, decision as dec
from halt_study import first_stop, score
sys.stdout.reconfigure(encoding='utf-8')
t1=sim.PrefixTable.load('runs/E001_m3_baseline/dev_prefix.npz'); t2=sim.PrefixTable.load('runs/E002_prefix_uniform/dev_prefix.npz')
avg=sim.PrefixTable(t2.call_ids,t2.offsets,(t1.probs+t2.probs)/2,t2.labels)
shift=np.log(json.load(open('runs/E002_prefix_uniform/run_config.json',encoding='utf-8'))['pos_weight'])
for name,t in (('E002',t2),('E001+E002 평균',avg)):
    n=t.n_utt; off=t.offsets; step=np.concatenate([np.arange(1,k+1) for k in n]); gold=np.repeat(t.labels,n,axis=0)
    value=dec.expected_rule(dec.calibrate(t.probs,shift))[1]
    print('==',name)
    for thr in (0.3,0.4,0.5,0.6):
        pred=dec.threshold_rule(t.probs.copy(),thr)
        best=max(((score(first_stop(value>=tau,off),pred,t),tau) for tau in np.arange(0.40,0.80,0.02)),key=lambda x:x[0]['score'])
        r=best[0]; print(f'  중단 시점 임계 {thr}: τ={best[1]:.2f} {r["score"]:.4f} = F1 {r["macro_f1"]:.4f} x {r["earliness"]:.4f}')
    pred=dec.threshold_rule(t.probs.copy())
    # 사후 최적: 통화마다 (순간 점수 x 조기판단력)이 가장 큰 시점에 끊었다면
    rew=sim.moment_scores(pred,gold)*sim.earliness(step,np.repeat(n,n))
    h=np.array([int(np.argmax(np.where(np.arange(1,k+1)>=min(5,k),rew[off[i]:off[i+1]],-1)))+1 for i,k in enumerate(n)])
    r=score(h,pred,t); print(f'  사후 최적 중단(천장): {r["score"]:.4f} = F1 {r["macro_f1"]:.4f} x {r["earliness"]:.4f} 중단 중앙 {r["halt_median"]:.0f}')
    # 중단 시점의 클래스별 F1 과 끝까지 들었을 때의 F1
    h=first_stop(value>=0.52,off); p=pred[off[:-1]+h-1]; full=pred[off[1:]-1]
    a=sim.macro_f1(p,t.labels)['per_class']; b=sim.macro_f1(full,t.labels)['per_class']
    print('  클래스별 F1 (중단 시점 / 끝까지): '+', '.join(f'{c} {a[c]:.2f}/{b[c]:.2f}' for c in a))
    print('  중단 시점 예측 개수 평균', round(float(p.sum(1).mean()),2), '정답 개수 평균', round(float(t.labels.sum(1).mean()),2))
