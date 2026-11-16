"""계산된 T8 사후 분석 JSON을 한국어 표·보고서·그림으로 정리한다. 모델 추론 없음."""
import argparse
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_t8 import RUNS, ROOT, BASE, T8, HERE, CLASSES, read, digest, sim

OUT = RUNS / "t8_analysis"


def f(x, digits=6):
    return f"{x:.{digits}f}"


def table(headers, rows):
    return "\n| " + " | ".join(headers) + " |\n|" + "|".join(["---"] * len(headers)) + "|\n" + "\n".join("| " + " | ".join(str(c).replace("|", "/").replace("\n", " ") for c in row) + " |" for row in rows) + "\n\n"


def comparison(v0, v1, keys):
    return table(["항목", "Baseline", "T8", "변화량(T8−Baseline)"], [[name, f(v0[key]), f(v1[key]), f(v1[key]-v0[key])] for name, key in keys])


def perclass(v0, v1, title):
    text = "\n### " + title + "\n"
    text += table(["증상", "Precision B→T8", "Recall B→T8", "F1 B→T8", "ΔF1"],
                  [[c, *[f(v0[c][k], 4) + " → " + f(v1[c][k], 4) for k in ["precision", "recall", "f1"]],
                    f(v1[c]["f1"]-v0[c]["f1"], 4)] for c in CLASSES])
    text += table(["증상", "TP B→T8 (Δ)", "FP B→T8 (Δ)", "FN B→T8 (Δ)", "TN B→T8 (Δ)"],
                  [[c, *[f"{v0[c][k]:,} → {v1[c][k]:,} ({v1[c][k]-v0[c][k]:+,})" for k in ["tp", "fp", "fn", "tn"]]] for c in CLASSES])
    return text


def figures(r, s):
    plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10})
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
    names = ["초반 Score", "중반 Score", "후반 Score", "4-1 Score"]
    d = np.array([r["bootstrap"][name]["delta"] for name in names])
    low = np.array([r["bootstrap"][name]["lo"] for name in names])
    high = np.array([r["bootstrap"][name]["hi"] for name in names])
    ax[0].errorbar(d, np.arange(4), xerr=[d-low, high-d], fmt="o", capsize=4)
    ax[0].set_yticks(np.arange(4), names); ax[0].axvline(0, color="gray", lw=1)
    ax[0].set_title("통화 단위 Bootstrap 95% 구간"); ax[0].set_xlabel("T8 - Baseline")
    x = np.arange(4)
    for j, (name, label) in enumerate([("macro_f1", "Macro-F1"), ("earliness", "Earliness"), ("score", "4-2 Score")]):
        ax[1].bar(x+(j-1)*.24, [r["counterfactual"][c][name] for c in "ABCD"], .24, label=label)
    ax[1].set_xticks(x, list("ABCD")); ax[1].set_ylim(0, 1); ax[1].legend(); ax[1].set_title("저장 배열의 반사실적 조합")
    fig.savefig(OUT / "성능과_반사실적_비교.png", dpi=180); plt.close(fig)
    arrays = np.load(OUT / "diagnostic_arrays.npz")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    ax[0].hist(arrays["delay"], bins=np.arange(-2, 82, 2), color="#506ea0")
    ax[0].set_xlabel("중단 발화 변화량(T8-Baseline)"); ax[0].set_ylabel("통화 수")
    ax[0].set_title("2,480통화에서 중단 지연")
    density = ax[1].hexbin(arrays["baseline_halts"], arrays["t8_halts"], gridsize=35, mincnt=1, bins="log", cmap="Blues")
    fig.colorbar(density, ax=ax[1], label="통화 수")
    ax[1].plot([0, 100], [0, 100], "--", color="gray"); ax[1].set_xlabel("Baseline 중단 발화"); ax[1].set_ylabel("T8 중단 발화")
    fig.savefig(OUT / "중단_위치_변화.png", dpi=180); plt.close(fig)
    tabs = [sim.PrefixTable.load(d / "dev_prefix.npz") for d in (BASE, T8)]
    yy = np.repeat(tabs[0].labels.astype(bool), tabs[0].n_utt, axis=0)
    fixed = [sim.PrefixTable.load(HERE / "runs" / d / "dev_prefix.npz").probs for d in ["E001_m3_baseline", "E007_prefix_uniform_s43"]]
    ens = [np.mean([fixed[0], t.probs, fixed[1]], axis=0) for t in tabs]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), layout="constrained")
    for row, (model, pp) in enumerate([("E002 단독", [t.probs for t in tabs]), ("앙상블", ens)]):
        for col, (label, mask) in enumerate([("실제 양성", yy), ("실제 음성", ~yy)]):
            for name, p in zip(["Baseline", "T8"], pp):
                axes[row, col].hist(p[mask], bins=np.linspace(0, 1, 51), density=True, histtype="step", lw=1.5, label=name)
            axes[row, col].axvline(.5, color="gray", ls="--")
            axes[row, col].set_title(model+" / "+label); axes[row, col].set_xlabel("보정 전 확률"); axes[row, col].set_ylabel("밀도"); axes[row, col].legend()
    fig.savefig(OUT / "예측_확률_분포.png", dpi=180); plt.close(fig)


def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=OUT)
    OUT = ap.parse_args().out.resolve()
    if OUT.parent != RUNS.resolve() or OUT in [BASE.resolve(), T8.resolve(), (RUNS / "localsequential").resolve()]:
        raise ValueError("분석 결과 폴더만 지정하십시오")
    r, s = read(OUT / "analysis.json"), read(OUT / "supplement.json")
    manifest = read(OUT / "input_manifest.json")
    assert all(digest(ROOT / p) == expected for p, expected in manifest["before"].items())
    b, t = r["batch"]
    cf = r["counterfactual"]
    text = """# T8 Asymmetric Loss 실험 심층 분석

분석 대상은 Local Baseline `localcontrol`과 `t8_asymmetric_loss`의 동일 Dev 2,920통화·87,814발화다.
새 학습·Fine-tuning·분류 모델 추론·E026 정책 추론·Threshold/Calibration 최적화를 실행하지 않았다.
원본 결과 파일의 분석 전후 SHA256이 동일함을 확인했다. 원문 사례는 분석 출력 폴더에만 저장하고 사례 ID를 해시로 표시했다.

**판단:** 초반 Score 개선은 이 Dev에서 확인되지만, 중심 변화는 확률 및 양성 예측 확대다.
동일 중단 문맥의 Macro-F1과 마지막 문맥의 Macro-F1은 낮아진다. 관측된 4-2 Macro-F1 상승은 중단 지연에 따른 추가 문맥 영향으로 설명된다.
현재 고정 E026 조합의 최종 개선 후보로는 부적합하며, 확률·순위 정보를 활용하는 별도 연구 후보로 보존할 수 있다.

## 1. 실험 적용 및 통제 변수

실제 로그·`run_config.json`은 ASL `gamma_pos=0`, `gamma_neg=4`, `clip=0.05`, `eps=1e-8`,
Focusing Weight detach, FP32 확률/로그, Batch×9 mean, Loss Pos Weight 미적용을 기록한다.
현재 Loss·학습기·평가기 소스 해시가 T8 시작 시 저장한 해시와 일치한다. 실제 개별 학습 Gradient는 저장되지 않았다.
Pos Weight 값이 config에 남아 있다는 것은 ASL에 사용했다는 뜻이 아니다. factory는 ASL에 그 값을 전달하지 않는다.

실제 수식은 `p=sigmoid(z)`, `q=min(1,1-p+0.05)`에 대해
`mean[-y*log(max(p,eps)) - (1-y)*detach((1-q)^4)*log(max(q,eps))]`다.
양성의 focusing 지수는 0이며 9개 증상 전체를 평균한다. Loss 내부 autocast를 끄고 FP32로 계산한다.

| 검증 항목 | 근거 및 판단 수준 |
|---|---|
| 캐시·Fit/Dev ID·E026 정책 | 현재 파일 SHA256과 두 실행 영수증 직접 대조, 모두 일치 |
| TAPT·Tokenizer | 현재 초기 파일 해시와 두 초기 영수증 일치. 실제 초기화된 분류 헤드 Tensor는 저장되지 않아 미확인 |
| 모델 구조 | 학습 완료 config 전체 직접 대조 일치. 초기화 조건은 seed42와 동일 생성 코드·설정에 근거 |
| 하이퍼파라미터 | 3 epoch, Batch16, Accumulation1, lr5e-5, LLRD0.8, AdamW/Decay0.01, Linear/Warmup0.1, max512/right, CUDA FP16, Gradient Checkpointing 비활성 동일 |
| Sampling | T8 실제 Sampling 해시가 Baseline 코드 CPU 재생 해시와 3 epoch 모두 일치 |
| 샘플·라벨·토큰·Padding | T8 학습 중 실제 기록과 CPU 재생 통계 일치. Baseline 과거 학습의 해당 영수증은 없어 Baseline 코드 재생 근거 |
| Truncation | CPU 재생에서 원래 길이 >512인 입력 수가 동일. T8 학습 중 독립 Truncation 계수는 없음 |
| E001/E007 | 현재 모델·확률표 해시와 Batch/Sequential 기록 직접 대조 일치 |
| 앙상블·Threshold·Calibration | 저장 확률을 동일 비중 3개 멤버 평균·0.5 판정으로 재계산하여 Batch 지표 정확히 재현. 동일 policy/logit_shift 해시 확인 |
| E026 | beta128/tau0.5/min_step5와 정책 전체 해시 동일. 정책 재추론은 하지 않음 |
| Batch/Sequential 교체 모델 | 각 실행의 현재 checkpoint 해시가 Batch 및 Sequential 영수증과 일치 |

이 실험은 **Weighted BCE를 ASL로 교체하는 복합 Loss 전략 비교**다. Pos Weight 제거와 ASL 각 요소의 효과는 분리할 수 없다.
형식적 통제 변수는 유지되었지만 실제 업데이트 수가 완전히 같다고 단정할 수는 없다.
T8은 예정 9,855회 중 **실제 9,854회, AMP Skip 1회(epoch3)**다. Baseline은 `total_steps=9855`만 있고 실제 Skip 별도 기록이 없다.
AMP Skip 발생 step과 원인은 저장되지 않았다. Scheduler는 기존과 같이 Skip에도 진행하므로 T8에는 갱신 없이 진행한 Scheduler step이 1회 있다.

### 학습량 및 시간
"""
    epochs = r["controls"]["cpu_epochs"]
    text += table(["Epoch", "샘플", "양성 라벨 노출 합", "Padding 제외 토큰", "Padding 토큰", "Truncation", "T8 업데이트/Skip"],
                  [[i+1, e["t8"]["samples"], sum(e["t8"]["label_positive_exposures"]), e["t8"]["nonpadding_tokens"],
                    e["t8"]["padding_tokens"], e["t8"]["truncated_samples"], f"{r['controls']['history'][1][i]['optimizer_updates']} / {r['controls']['history'][1][i]['amp_skips']}"] for i,e in enumerate(epochs)])
    times = r["controls"]["training_seconds"]
    text += f"총 샘플 **157,680개**. 학습 시간은 **{times[0]:,.1f} → {times[1]:,.1f}초**, +{times[1]-times[0]:.1f}초(+{100*(times[1]/times[0]-1):.2f}%)다. 환경 변동과 새 기록 코드의 오버헤드를 분리할 수 없으므로 증가분을 ASL 계산 비용만으로 해석하지 않는다.\n\n"
    text += "양성 라벨 노출은 각 epoch 증상 순서대로 `[9372,8078,5208,12124,11434,6162,6054,9606,7084]`다. 음성 노출은 52,560에서 각 값을 뺀 값이다. Weighted BCE와 ASL의 학습 Loss 척도가 다르므로 그 수치로 성능을 비교하지 않는다.\n\n"
    text += "## 2. 전체 결과와 발화 구간\n\n4-1 공식 Score는 발화별 Jaccard를 통화 안에서 평균한 뒤 통화를 동일 가중한다. 구간 Score와 4-1 혼동행렬은 기존 보고 형식대로 해당 구간의 발화 시점을 모은다. 4-2는 통화별 중단 시점 한 번을 사용한다.\n"
    text += table(["경로", "지표", "Baseline", "T8", "변화량"], [[path, label, f(r[path][0][task][key]), f(r[path][1][task][key]), f(r[path][1][task][key]-r[path][0][task][key])]
                 for path in ["batch", "sequential"] for label, task, key in [("4-1 Score","4-1","score"),("4-2 Score","4-2","score"),("Macro-F1(실제 중단)","4-2","macro_f1"),("Earliness","4-2","earliness")]])
    text += "### 상대·절대 구간 및 마지막 발화\n"
    text += table(["구간", "발화 수", "통화 수", "Baseline", "T8", "ΔScore"], [[name,v["moments"],v["calls"], f(v["baseline"]["score"]),f(v["t8"]["score"]),f(v["t8"]["score"]-v["baseline"]["score"])] for name,v in r["segments"].items()])
    text += "구간별 분모는 해당 발화 위치까지 존재하는 통화에 한정된다. 상대 구간은 전체 길이를 이용한 **사후 채점 구분**이며 추론 입력으로 사용하지 않았다.\n\n통화를 동일 가중한 상대 구간 평균도 같은 방향이다.\n"
    text += table(["구간", "통화 동일 가중 Baseline", "통화 동일 가중 T8", "변화량"], [[name,*[f(x) for x in s["stages"][name]["call_balanced_score"]], f(np.diff(s["stages"][name]["call_balanced_score"])[0])] for name in ["초반","중반","후반"]])
    text += table(["전체 길이", "통화 수", "4-1 Baseline", "4-1 T8", "변화량", "평균 중단 지연"], [[name,v["calls"],f(v["baseline"]),f(v["t8"]),f(v["t8"]-v["baseline"]),f(v["halt_delta_mean"],2)] for name,v in r["length_groups"].items()])
    text += "1~10발화 집단은 11통화뿐이다. 소수 통화의 개선을 전체 길이 집단으로 일반화하지 않는다. 전체 4-1의 작은 순증가와 후반 하락을 함께 봐야 한다.\n\n"
    text += "### T5와 달리 Sampling 변경 없이 초반 Score가 오른 이유\n\n"
    if "T5" in r["prior"]:
        text += table(["실험", "초반", "중반", "후반", "4-1 Score"], [[name,*[f(v["4-1"]["by_third"][k]) for k in ["앞","가운데","뒤"]],f(v["4-1"]["score"])] for name,v in r["prior"].items()])
    text += "T5의 저장 결과는 초반 개선을 보여주지 않는다. 그러나 이 사실만으로 Sampling 변경이 항상 무효라는 결론은 낼 수 없다. T8의 초반 증가는 다음과 같이 예측 범위 확대로 분해된다.\n"
    text += table(["초반 시점 집단", "시점 수", "Baseline Score", "T8 Score", "초반 전체 ΔScore 기여"], [[name,v["moments"],"—" if v["baseline_score"] is None else f(v["baseline_score"]),"—" if v["t8_score"] is None else f(v["t8_score"]),f(v["contribution_to_segment_delta"])] for name,v in s["stages"]["초반"]["score_change_groups"].items()])
    text += "**관찰된 집계 분해:** 빈 예측에서 벗어난 3,859시점이 +0.029397을 만들고, 기존에 증상을 예측하던 시점에서는 −0.009483이 발생해 순증가 +0.019914가 된다. 따라서 초반 상승의 집계상 이득은 빈 예측 감소에 집중된다. 이는 빈 예측 감소의 독립적인 인과 효과를 추정한 실험은 아니다.\n\n"
    text += table(["초반 확률 순위 지표", "Baseline", "T8", "변화량"], [[model+" "+key,*[f(v[key]) for v in vals],f(vals[1][key]-vals[0][key])] for model,vals in s["stages"]["초반"]["ranking"].items() for key in ["auroc_macro","auprc_macro","top1_in_gold"]])
    text += "앙상블 초반 AUROC와 Top1 정답 포함률은 개선되지 않고 AUPRC는 소폭만 변한다. 초반 Score 이득은 유효하지만, 그 크기에 상응하는 증상 판별·순위 능력 개선의 근거는 부족하다. 일부 명시적 증상을 더 잡는 사례도 있으므로 전부 무작위 추측이라고 단정하지 않는다.\n\n"
    text += "## 3. 증상별 분류 및 오류 변화\n"
    for task in ["4-1","4-2"]:
        text += perclass(b[task]["per_class"],t[task]["per_class"],task+" 실제 평가 시점")
        text += table(["전체 계수", "Baseline", "T8", "변화량"], [[k,b[task]["totals"][k],t[task]["totals"][k],t[task]["totals"][k]-b[task]["totals"][k]] for k in ["tp","fp","fn","tn"]])
    text += "4-2의 표는 서로 다른 중단 시점의 비교다. 9개 증상 모두 F1이 올라도 동일 문맥의 분류 개선을 뜻하지 않는다.\n\n구토는 4-1 Recall 상승과 함께 Precision이 크게 떨어져 F1이 하락했다. 오심은 Recall 확대가 F1을 높였지만 낮은 Precision과 큰 FP 증가가 남았다. 두통의 4-1 F1 증가는 작다. 이 세 증상의 실제 중단 F1 상승에는 추가 문맥 영향이 섞여 있다. 동일 Baseline 중단 위치의 증상별 결과는 `세부_표.md`에 별도로 제공한다.\n\n"
    text += table(["상대 구간", "ΔTP", "ΔFP", "ΔFN", "ΔTN", "새 양성 중 정답 비율", "평균 증상 B→T8", "빈 예측 B→T8"], [[name,*[v["t8"]["totals"][k]-v["baseline"]["totals"][k] for k in ["tp","fp","fn","tn"]],f(v["transitions"]["new_positive_true"]/v["transitions"]["new_positive"],4),f(v["baseline"]["mean_labels"],4)+" → "+f(v["t8"]["mean_labels"],4),f(v["baseline"]["empty_rate"],4)+" → "+f(v["t8"]["empty_rate"],4)] for name,v in r["segments"].items() if name in ["초반","중반","후반"]])
    text += "초반 새 양성의 약 25.6%만 정답이다. 후반에는 이미 빈 예측이 적어 얻을 수 있는 이득이 작고, TP 증가보다 FP 증가가 훨씬 커 Jaccard의 합집합 분모가 늘어난다. **초반 개선이 후반 하락을 일으켰다는 인과 주장이 아니라**, 같은 예측 확대가 구간별로 다른 손익을 보인다는 해석이다.\n\n"
    text += table(["경로", "평균 예측 수 B→T8", "빈 예측 비율 B→T8"], [[task,f(b[task]["mean_labels"],4)+" → "+f(t[task]["mean_labels"],4),f(b[task]["empty_rate"],4)+" → "+f(t[task]["empty_rate"],4)] for task in ["4-1","4-2"]])
    text += table(["예측 개수", "4-1 Baseline", "4-1 T8", "4-2 Baseline", "4-2 T8"], [[k,*[v[task]["label_count_distribution"][str(k)] for task in ["4-1","4-2"] for v in [b,t]]] for k in range(10)])
    text += "## 4. E002 단독과 앙상블의 출력 확률\n\n확률 통계·Brier·ECE·AUROC/AUPRC는 모든 발화×증상 위치를 사용한다. 긴 통화가 더 큰 비중을 갖는 진단 지표이며 공식 Score와 집계 단위가 다르다. ECE는 사전에 고정한 동일 너비 10구간으로 계산했으며 보정값을 학습하거나 조정하지 않았다.\n"
    for name,p in r["probabilities"].items():
        text += "### "+name+"\n"
        text += table(["지표", "Baseline", "T8", "변화량"], [[label,f(p["classification"][0][key]),f(p["classification"][1][key]),f(p["classification"][1][key]-p["classification"][0][key])] for label,key in [("4-1 Score","score"),("발화 단위 Macro-F1","macro_f1"),("평균 예측 수","mean_labels"),("빈 예측 비율","empty_rate")]])
        text += table(["확률 조건", "평균 B→T8", "중앙값 B→T8", "5% B→T8", "95% B→T8"], [[label,*[f(p["raw"][0][field][key] if key=="mean" else p["raw"][0][field]["quantiles"][key],4)+" → "+f(p["raw"][1][field][key] if key=="mean" else p["raw"][1][field]["quantiles"][key],4) for key in ["mean","50","5","95"]]] for label,field in [("전체","all"),("실제 양성","positive"),("실제 음성","negative")]])
        text += table(["확률 상태", "지표", "Baseline", "T8", "변화량"], [[kind,key,f(p[kind][0][key]),f(p[kind][1][key]),f(p[kind][1][key]-p[kind][0][key])] for kind in ["raw","calibrated"] for key in ["brier","ece_10","auroc_macro","auprc_macro"]])
        z=p["transitions"]
        text += f"0.5를 새로 넘은 위치 **{z['new_positive']:,}개**, 정답 {z['new_positive_true']:,}개 / 오탐 {z['new_positive_false']:,}개. 양성에서 음성으로 바뀐 위치는 {z['removed_positive']:,}개다.\n\n"
    text += "**관찰:** T8 단독 E002는 실제 음성 평균 확률이 약 0.506으로 올라가고 평균 5.37개 증상을 예측한다. 단독 고정 Threshold 분류는 크게 악화됐다. 고정된 다른 두 모델이 이 변화를 평균 과정에서 완화하여 앙상블 Score 하락을 막은 형태다. 이는 개선된 새 단독 분류기가 앙상블을 끌어올렸다는 설명과 다르다.\n\nAUROC는 거의 유지되고 AUPRC 상승은 작아, 분류 경계의 대대적 개선보다 확률 척도·위치 변화의 증거가 강하다. Brier/ECE는 보정 전후 모두 악화된다. 단, E002 단독의 calibrated 수치는 E026과 같은 고정 shift를 진단용으로 적용한 결과이며 실제 운영에서 E002에 먼저 보정을 적용한다는 뜻은 아니다. E026은 앙상블 후 보정값을 특징에 사용하고 실제 증상 판정은 보정 전 0.5로 수행한다.\n\n"
    text += "**원인 가설:** ASL의 쉬운 음성 Gradient 억제와 clipping은 일부 낮은 확률 음성에 대한 하향 학습 압력을 약하게 할 수 있다. 반면 Pos Weight 제거는 Weighted BCE의 양성 압력을 줄이는 방향이어서, 관측된 확률 상승을 Pos Weight 제거 하나로 설명할 수 없다. 두 효과가 결합된 최적화 결과로 확률이 재배치되었을 가능성이 있다. 실제 학습 Gradient를 측정하지 않았고 개별 요소 분리 실험도 없으므로 원인으로 확정하지 않는다.\n\n"
    text += "![예측 확률 분포](예측_확률_분포.png)\n\n## 5. 분류와 중단 위치의 반사실적 분리\n\nA=Baseline 분류+Baseline 중단, B=T8 분류+Baseline 중단, C=Baseline 분류+T8 중단, D=T8 분류+T8 중단이다. **저장 배열을 교차 선택한 사후 진단이며 실제 정책 또는 제출 가능한 새 정책이 아니다.**\n"
    text += table(["조합", "Macro-F1", "Earliness", "4-2 Score", "평균 중단", "중앙 중단"], [[key,*[f(v[k]) for k in ["macro_f1","earliness","score","halt_mean","halt_median"]]] for key,v in cf.items()])
    text += f"같은 Baseline 중단에서 분류만 교체(A→B)하면 Macro-F1 **{cf['B']['macro_f1']-cf['A']['macro_f1']:+.6f}**, 같은 T8 중단에서 교체(C→D)해도 **{cf['D']['macro_f1']-cf['C']['macro_f1']:+.6f}**다. 반대로 Baseline 모델도 늦게 듣게 하면(A→C) **{cf['C']['macro_f1']-cf['A']['macro_f1']:+.6f}** 상승한다. T8 자체의 추가 문맥 효과(B→D)는 **{cf['D']['macro_f1']-cf['B']['macro_f1']:+.6f}**다.\n\n"
    last=r["segments"]["마지막"]
    text += f"동일 마지막 발화의 Macro-F1도 **{last['baseline']['macro_f1']:.6f} → {last['t8']['macro_f1']:.6f}**다. 따라서 이번 Dev의 고정 판정 기준에서는 실제 중단 Macro-F1 상승을 같은 문맥의 전반적 분류 개선으로 설명하기 어렵다.\n\n"
    text += table(["4-2 변화의 경로별 계산", "Score 변화"], [[key,f(value)] for key,value in s["score42_decomposition"].items()])
    text += "두 경로를 평균하는 Shapley식 회계 분해에서는 약 −0.004734가 분류 교체, −0.067588이 중단 위치 교체에 대응한다. 중단 관련 몫은 총 하락의 약 93.5%지만 **인과적 기여율이 아니다**. F×E 항등식도 F 상승 기여보다 E 감소 항이 훨씬 큼을 보여준다.\n\n"
    text += "### Sequential 확인\n"
    text += table(["조합", "Sequential Macro-F1", "Earliness", "Score"], [[k,*[f(v[x]) for x in ["macro_f1","earliness","score"]]] for k,v in r["sequential_counterfactual"].items()])
    text += table(["실험", "Batch/Sequential 4-1 다른 라벨 위치", "다른 중단 통화", "Sequential 4-1에서 뽑은 중단 예측과 실제 4-2 차이"], [[name,*r["sequential_checks"][i].values()] for i,name in enumerate(["Baseline","T8"])])
    text += "Batch/Sequential은 같은 checkpoint를 사용했지만 수치 계산 경로가 달라 일부 0.5 판정과 중단이 다르다. 저장 Sequential 4-1을 실제 Sequential 4-2 중단에 맞춰 뽑은 예측은 양쪽 모두 실제 4-2 배열과 정확히 일치한다. 따라서 Sequential 반사실적 표도 저장 결과에 근거하며 Batch와 같은 결론을 보인다.\n\n"
    text += "![성능 및 반사실적 비교](성능과_반사실적_비교.png)\n\n## 6. 중단 지연 및 E026 입력 변화\n"
    h=r["halting"]
    text += table(["항목", "결과"], [["빨라짐 / 동일 / 늦어짐",f"{h['earlier']} / {h['same']} / {h['later']}통화"], ["평균 / 중앙 지연",f"{h['delta']['mean']:.3f} / {h['delta']['quantiles']['50']:.0f}발화"], ["지연 25% / 75% / 95% / 최대", " / ".join(f(h['delta']['quantiles'][k],1) for k in ["25","75","95","100"])], ["마지막까지 수신 B→T8",f"{h['at_final'][0]} → {h['at_final'][1]}통화 ({100*h['at_final_rate'][0]:.2f}% → {100*h['at_final_rate'][1]:.2f}%)"]])
    text += table(["중단 변화량", "통화 수"],h["delay_bins"].items())
    text += "**조건 미충족 비율은 정확히 계산할 수 없다.** 마지막까지 수신에는 마지막 발화에서 처음 조건을 만족한 통화와 끝까지 조건을 만족하지 못한 fallback이 섞인다. 기존 `never_halted`도 `halt==n`으로 계산하므로 이를 엄밀한 조건 미충족률로 읽으면 안 된다. 확실한 조건 미충족률의 상한은 각각 2.19%, 35.00%이며 비정보적 하한은 0%다.\n\n"
    text += "E026은 `step>=5`에서 `value + 128*delta_hat >= 0.5`를 처음 만족할 때 멈춘다. 저장 확률에서 Calibration·기대 점수·57개 입력 Feature만 계산했다. `delta_hat`과 최종 signal은 저장되지 않아 새 LightGBM 추론 없이 알 수 없다. 숫자 signal·변화량·전체 조건 충족 횟수를 계산했다고 주장하지 않는다.\n\n"
    text += "중단이 늦어진 2,480통화에서는 Baseline 중단 시점이 T8 종료보다 이르므로, 저장된 결정과 동일 정책 정의에 근거해 Baseline은 그 시점 조건 충족, T8은 미충족이었다고 **간접 추론**할 수 있다. 이는 수치 signal을 직접 확인한 결과가 아니다.\n\n"
    selected=["top1","top2","top3","cal_sum","n_pred","same_pred_run","value","value_delta","value_max","cal_rise_max","cal_change_abs"]
    text += table(["Feature", "동일 Baseline 중단 B", "동일 Baseline 중단 T8", "평균 변화"], [[name,*[f(v['mean']) for v in r['features'][name]['at_baseline_halt']],f(r['features'][name]['at_baseline_halt'][1]['mean']-r['features'][name]['at_baseline_halt'][0]['mean'])] for name in selected])
    text += "같은 발화 시점에서 top1은 오히려 조금 높지만 top2/top3와 예상 증상 수가 더 크게 늘고 기대 순간 Score `value`는 낮아진다. 여러 증상을 동시에 그럴듯하게 보는 확률이 늘면 기대 Jaccard가 낮아질 수 있다. 이는 고정된 `expected_rule` 계산에서 직접 관찰한 관계다. 높은 최고 확률만으로 E026이 빨리 멈출 것이라고 기대할 수 없다.\n\n"
    text += table(["동일 Baseline 중단의 Feature 변화", "중단 지연과 Spearman ρ"], [[v['feature'],f(v['rho_delta_feature_vs_delay'])] for v in h['feature_correlations'][:10]])
    text += f"동일 발화의 전사·화자·시간 기반 35개 Feature는 완전히 같다. 변화는 확률 기반 22개 Feature에 있다. 다만 개별 변화량과 지연의 상관은 크지 않으며, 가장 큰 절댓값도 약 0.161이다. 전체 길이와 지연의 ρ는 {s['delay_vs_length_spearman']:.3f}다. 통화 난도·길이·정책의 비선형 상호작용을 통제한 인과 분석이 아니다.\n\n"
    text += table(["Baseline 중단에서의 예측 수 변화", "통화 수", "평균 지연", "평균 통화 길이"], [[name,v['calls'],f(v['delay_mean'],3),f(v['length_mean'],3)] for name,v in s['delay_by_prediction_count'].items()])
    text += "예측 증상 수가 같은 2,228통화에서도 평균 지연이 크다. 따라서 정수 예측 개수 증가만으로 지연을 설명할 수 없다. 0.5 아래 확률의 상승도 Calibration 이후 expected value와 회귀 입력을 바꾼다. 예측 안정성 `same_pred_run`도 평균 0.15 낮아졌지만 지연과의 상관은 약 −0.019로 약하다.\n\n![중단 위치 변화](중단_위치_변화.png)\n\n"
    text += "## 7. 대표 사례: 관찰과 해석의 한계\n\n다음은 극단적인 개선/실패/지연 사례를 목적에 맞게 선택한 예시다. 빈도 대표 표본이 아니며 전체 데이터로 일반화하지 않는다. 정답은 통화 단위 라벨이다. 원문 발화 자체에도 질문·부정·다른 증상이 포함될 수 있어 단어 등장만으로 그 시점 확정 진단을 판단하지 않는다.\n"
    for case in s['explicit_early_symptom_examples'][:2]:
        text += f"- **명시적 증상 뒤 추가 검출 {case['case']}**: {case['step']}번째 발화 “{case['text'].strip()}”. 정답 {', '.join(case['gold'])}. 예측은 {case['predictions'][0]} → {case['predictions'][1]}, 순간 Score {case['moment_scores'][0]:.3f} → {case['moment_scores'][1]:.3f}. 추가 TP를 얻지만 불필요한 기존 예측도 남는 사례다. 중단은 {case['halts'][0]} → {case['halts'][1]}발화.\n"
    for case in [r['cases'][1],r['cases'][2],r['cases'][-1]]:
        text += f"\n- **{case['category']} {case['case']}**: 정답 {', '.join(case['gold'])}, 길이 {case['n']}, 4-1 {case['score41'][0]:.4f} → {case['score41'][1]:.4f}, 중단 {case['halts'][0]} → {case['halts'][1]}. Baseline 중단에서 예측 {case['at_baseline_halt'][0]} → {case['at_baseline_halt'][1]}, 마지막에는 {case['at_final'][0]} → {case['at_final'][1]}.\n"
        for snip in case['snippets'][:3]:
            text += f"  - 발화 {snip['step']}: “{snip['text'].strip()}”\n"
    text += "\n초반 개선 사례 `a043b8002d`는 2번째 “여보세요 혹시 구급차 혹시 부를 수 있나요?”에서 T8이 이미 복통을 내지만 구체적인 복통 진술은 19번째다. Prefix 정답이 통화 전체 라벨인 평가에서는 이런 선제 정답도 점수를 얻는다. 이를 곧바로 증상 근거를 더 잘 이해했다는 증거로 해석할 수 없다. 반대로 호흡곤란 명시 뒤 추가 TP 사례도 실제로 존재한다.\n\n실패 사례 `49196e37e8`는 어지러움에 구토·오심을 함께 덧붙여 1개 정답에 3개를 예측한다. 중단 지연 사례 `c5d188e27e`는 Baseline이 10번째에 맞힌 호흡곤란을 T8도 마지막에는 맞히지만 71번째까지 수신한다. 분류가 마지막에 맞는 것과 조기 중단의 성공은 다르다.\n\n"
    text += "## 8. 통화 단위 Paired Bootstrap\n\n동일 Dev 통화를 복원 추출해 Baseline/T8에 같은 재표집을 적용했다. **2,000회, seed20261116, percentile 95% CI**다. 4-2는 매 재표집마다 통화 혼동행렬을 합쳐 Macro-F1을 다시 계산하고 해당 표본의 평균 Earliness를 곱한다. 통화별 F1 또는 통화별 4-2 Score를 평균하지 않는다. 상대 구간은 해당 재표집의 발화 Score 합/발화 수로 계산한다. 기존 T3 분석의 별도 코드/Bootstrap 영수증은 확보하지 못했으므로 요청된 방식·반복수·seed를 따랐고 과거 난수 배열까지 같다고 주장하지 않는다.\n"
    text += table(["지표", "변화량", "95% CI 하한", "상한", "0 포함"], [[name,f(v['delta']),f(v['lo']),f(v['hi']),"포함" if v['includes_zero'] else "미포함"] for name,v in r['bootstrap'].items()])
    text += "초반 개선, 후반 하락, 중단 지연에 의한 Earliness 및 4-2 하락은 이 Dev의 표집 변동만으로 설명하기 어렵다. 전체 4-1 변화는 0을 포함한다. 동일 Baseline 중단 Macro-F1 하락의 상한은 0에 가깝기 때문에 특히 다중 비교에 주의해야 한다.\n\n단일 학습 seed 결과다. 이 Bootstrap은 Dev 통화 표집 불확실성만 반영하며 학습 seed·환경·모델 선택 변동을 반영하지 않는다. 여러 지표와 구간을 비교했고 CI에 다중 검정 보정을 하지 않았다. 양성 Bootstrap 비율은 유의확률로 해석하지 않는다.\n\n"
    text += "## 9. 최종 판단\n\n| 질문 | 판단 및 근거 |\n|---|---|\n| 초반 개선은 의미가 있는가? | 이 Dev의 초반 Score 개선과 빈 예측 감소는 명확하다. 그러나 순위 지표는 거의 유지되고 기존 비어 있지 않은 예측 집단은 악화돼 전반적 증상 이해 향상으로 볼 근거는 약하다. |\n| Macro-F1 상승의 원인은? | 동일 중단 위치 두 곳과 마지막 시점에서 모두 T8 F1이 낮다. 관측된 실제 중단 F1 상승은 더 많은 발화를 받은 효과가 우세하다. 일부 클래스·문맥 개선은 별도로 존재한다. |\n| 4-2 하락의 주요 경로는? | 2,480통화의 지연과 Earliness 0.829554→0.675213이다. 확률 기반 입력 분포 변화와 value 하락이 함께 관찰되지만 회귀 출력 기여도는 미확인이다. |\n| Prefix Learning의 개선/악화는? | 초반의 과도한 빈 예측·FN은 완화됐다. 확률 과대화, 다중 증상 FP, 후반 Precision 및 고정 E026 호환성은 악화됐다. |\n| 현재 최종 개선 후보로 유지? | 고정 E026 포함 최종 조합으로 채택할 근거는 없다. 4-1 순증가도 CI가 0을 포함하고 E002 단독 고정 판정은 크게 악화됐다. Baseline을 유지하는 판단이 타당하다. |\n| 다른 보정·중단 연구와 결합할 가치? | checkpoint와 확률표를 연구 후보로 보존할 가치는 있다. 순위 정보는 대체로 유지된다. 다만 보정만으로 개선될지, 다른 정책에서 Baseline을 이길지는 이번 결과로 알 수 없고 현재 채택 근거는 아니다. |\n| 추가 Loss 실험 필요? | ASL을 계속 연구한다면 Pos Weight 제거만의 대조군과 clipping/focusing의 분리 실험이 있어야 원인 규명이 가능하다. 이는 향후 검토안이며 이번 작업에서 학습·탐색·보정 최적화를 실행하지 않았다. 기대 효과가 입증된 것은 아니다. |\n\n"
    text += "**확인된 사실:** 기록된 ASL 적용, 동일 입력·고정 멤버·정책, 실제 1회 AMP Skip, 확률/양성 확대, 동일 문맥 F1 저하, 중단 지연 및 점수 변화.\n\n**관찰된 관계:** 낮아진 expected value와 확률 Feature 변화가 지연과 함께 나타난다. 단독 순위 정보는 대체로 보존됐지만 확률 보정 지표는 나빠졌다.\n\n**가설:** 음성 억제 약화와 비대칭 Loss에 따른 확률 척도 재배치가 고정 E026의 학습 분포와 맞지 않을 가능성이 있다. 실제 Gradient·회귀 signal·요소별 대조군이 없어 원인으로 확정하지 않는다.\n\n"
    text += "## 10. 추론 시간·재현성·제한\n"
    text += table(["실행", "Baseline 초", "T8 초", "변화량", "Baseline 초/발화", "T8 초/발화"], [["E002 Batch 추출",r['timing']['batch_extraction'][0]['seconds'],r['timing']['batch_extraction'][1]['seconds'],f(r['timing']['batch_extraction'][1]['seconds']-r['timing']['batch_extraction'][0]['seconds'],3),f(r['timing']['batch_extraction'][0]['seconds']/r['n_moments']),f(r['timing']['batch_extraction'][1]['seconds']/r['n_moments'])], *[["Sequential "+mission,f(r['timing']['sequential'][0][mission]['seconds'],3),f(r['timing']['sequential'][1][mission]['seconds'],3),f(r['timing']['sequential'][1][mission]['seconds']-r['timing']['sequential'][0][mission]['seconds'],3),f(r['timing']['sequential'][0][mission]['seconds_per_utterance']),f(r['timing']['sequential'][1][mission]['seconds_per_utterance'])] for mission in ['4-1','4-2']]])
    text += "Batch 시간은 교체 멤버 추출만이고 Sequential은 3개 모델의 실제 기존 실행 시간이다. 로딩·파일 IO 범위는 기존 기록 정의를 유지한다. 새로 시간을 측정한 추론 결과가 아니다. Sequential 4-2 총 시간 증가는 중단 지연으로 처리 발화 수가 늘어난 영향과 발화당 처리 시간 변동이 섞인다.\n\n"
    text += "- 미확보: Baseline 실제 AMP Skip/업데이트 영수증, 과거 학습 당시 Sampling/토큰 독립 기록, 실제 초기 헤드 Tensor, 실제 학습 Gradient, 시점별 E026 delta_hat/signal/조건 충족 배열, 이전 T3 Bootstrap 실행 코드.\n- 기존 정책 결과의 `halt==n`과 실제 조건 미충족을 구분할 수 없다. 마지막 중단율을 그대로 실패율이라 부르지 않았다.\n- 새 모델 추론을 금지했으므로 E026 delta_hat도 재생성하지 않았다. 이번 Feature 계산은 저장 확률·수식·기존 발화 데이터만 이용했다.\n- 공식 Validation·외부 데이터는 사용하지 않았다. 모든 해석은 같은 Dev의 사후 진단이며 별도 검증셋 확인이 아니다.\n- 원본 모델·결과·로그를 수정하지 않았다. Commit/Push/Merge를 하지 않았다. 이전 T8 구현의 미커밋 변경도 유지했다.\n\n"
    text += "산출물: [세부 증상·확률·Feature 표](세부_표.md), `analysis.json`, `supplement.json`, `input_manifest.json`, `diagnostic_arrays.npz`, 그림 PNG 3개. 분석 집계와 Bootstrap은 CPU 단위 테스트 3개로 독립 확인했다.\n\n저장소 루트에서 재현할 때는 아래 순서로 동일한 새 분석 폴더를 지정한다. 기본 폴더는 이미 생성되어 있으므로 재실행에는 새 이름을 사용한다. 기존 모델·결과 폴더를 출력으로 지정하면 거부한다.\n\n```powershell\n$py = (Resolve-Path .venv/Scripts/python.exe).Path\n& $py -B mission4_symptom/scripts/analyze_t8.py --out outputs/mission4_prefix/t8_analysis_recheck\nif ($LASTEXITCODE -ne 0) { throw '기본 분석 실패' }\n& $py -B mission4_symptom/scripts/supplement_t8_analysis.py --out outputs/mission4_prefix/t8_analysis_recheck\nif ($LASTEXITCODE -ne 0) { throw '보충 분석 실패' }\n& $py -B mission4_symptom/scripts/write_t8_report.py --out outputs/mission4_prefix/t8_analysis_recheck\n```\n"
    (OUT / "T8_심층분석_보고서.md").write_text(text,encoding="utf-8")
    details = "# T8 세부 통계 표\n\n혼동행렬은 집계 기준별로 구분했다. 0 분모의 Precision/Recall/F1은 이 분석에서 0으로 표시한다. 원본 평가의 null과 다른 표시 규칙이며 TP/FP/FN/TN은 동일하다.\n\n"
    for name in ["초반","중반","후반","마지막"]:
        v=r['segments'][name]
        details += perclass(v['baseline']['per_class'],v['t8']['per_class'],name)
    details += perclass(cf['A']['per_class'],cf['B']['per_class'],"동일 Baseline 중단 위치 A→B")
    details += perclass(cf['C']['per_class'],cf['D']['per_class'],"동일 T8 중단 위치 C→D")
    for name,p in r['probabilities'].items():
        details += perclass(p['classification'][0]['per_class'],p['classification'][1]['per_class'],name+" 고정 0.5 분류")
        details += "\n## "+name+" 증상별 확률\n"
        details += table(["증상","양성 평균 B→T8","음성 평균 B→T8","Brier B→T8","AUROC B→T8","AUPRC B→T8","확률 5% B→T8","확률 50% B→T8","확률 95% B→T8"], [[c,*[f(p['raw'][0]['per_class'][c][k],4)+" → "+f(p['raw'][1]['per_class'][c][k],4) for k in ['positive_mean','negative_mean','brier','auroc','auprc']],*[f(p['raw'][0]['per_class'][c]['quantiles'][k],4)+" → "+f(p['raw'][1]['per_class'][c]['quantiles'][k],4) for k in ['5','50','95']]] for c in CLASSES])
    details += "\n## E026 전체 Feature 분포\n\n동일 시점의 평균·표준편차·5/50/95 분위수다. own_halt와 전체 분포 전체 수치는 analysis.json에 있다. 회귀 출력/판단 signal은 포함하지 않는다.\n"
    for scope,title in [('same_step_5','동일 5번째 발화'),('at_baseline_halt','동일 Baseline 중단 위치')]:
        details += "### "+title+"\n"
        details += table(['Feature','평균 B→T8','표준편차 B→T8','5% B→T8','50% B→T8','95% B→T8'],[[name,f(v[scope][0]['mean'],4)+" → "+f(v[scope][1]['mean'],4),f(v[scope][0]['std'],4)+" → "+f(v[scope][1]['std'],4),*[f(v[scope][0]['quantiles'][k],4)+" → "+f(v[scope][1]['quantiles'][k],4) for k in ['5','50','95']]] for name,v in r['features'].items()])
    (OUT / "세부_표.md").write_text(details,encoding='utf-8')
    figures(r,s)
    print("한국어 보고서·세부 표·그림 3개 작성 완료")


if __name__ == "__main__":
    if hasattr(sys.stdout,"reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
