"""Display recorded aggregate results without importing training/inference modules."""
from __future__ import annotations

import hashlib
import html
import json
import math
from pathlib import Path


def load_snapshot(results_dir):
    """Verify bundled result files before displaying their recorded values."""
    root = Path(results_dir).resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["files"]:
        path = (root / entry["path"]).resolve()
        if root not in path.parents:
            raise ValueError("Result path must be inside results_dir")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != entry["sha256"]:
            raise ValueError(f"Recorded result checksum mismatch: {entry['path']}")
    snapshot = json.loads((root / "recorded_summary.json").read_text(encoding="utf-8"))
    for key in ("batch", "cli"):
        result = snapshot["final"][key]
        cm = result["confusion_matrix"]
        total = sum(sum(row) for row in cm)
        correct = cm[0][0] + cm[1][1]
        if total != result["n"] or correct != result["correct"]:
            raise ValueError(f"Inconsistent recorded confusion matrix: {key}")
        if not math.isclose(correct / total, result["accuracy"], abs_tol=1e-10):
            raise ValueError(f"Inconsistent recorded accuracy: {key}")
    return snapshot


def _table(headers, rows):
    return headers, [[str(value) for value in row] for row in rows]


def _section(snapshot, section):
    if section in snapshot.get("experiment_tables", {}):
        table = snapshot["experiment_tables"][section]
        return table["title"], _table(table["headers"], table["rows"]), table.get("notes", [])
    if section == "hypothesis_seeds":
        rows = []
        for h in snapshot["hypotheses"]:
            for seed, r in zip((42, 43), h["seed_results"]):
                metrics = r["metrics"]
                baseline = r.get("baseline_metrics")
                bacc = baseline["accuracy"] if baseline else metrics["accuracy"] - r["delta_accuracy_pp"] / 100
                ci = r["ci_95_pp"]
                d0 = r.get("delta_recall_0_pp", 100 * (metrics["recall_0"] - baseline["recall_0"]) if baseline else 0)
                d1 = r.get("delta_recall_1_pp", 100 * (metrics["recall_1"] - baseline["recall_1"]) if baseline else 0)
                rows.append([h["id"], seed, f"{bacc * 100:.4f}%", f"{metrics['accuracy'] * 100:.4f}%",
                             f"{r['delta_accuracy_pp']:+.4f}", f"{metrics['macro_f1']:.4f}",
                             f"{d0:+.4f}", f"{d1:+.4f}", f"[{ci[0]:+.4f}, {ci[1]:+.4f}]",
                             r["ci_repetitions"]])
        return "500통화: 가설별 두 seed의 대조 결과", _table(
            ["가설", "Seed", "기준선 Accuracy", "조건 Accuracy", "ΔAcc %p", "Macro F1",
             "ΔRecall 0 %p", "ΔRecall 1 %p", "95% 통화 CI %p", "재표집 횟수"], rows), [
            "각 실행 안의 기준선과 비교합니다. Centroid 별도 실행과 통합 실행의 기준선은 구분합니다.",
            snapshot.get("hypothesis_ci_provenance", "CI 출처는 집계 파일을 참고합니다."),
            "학습 초기화뿐 아니라 DataLoader 셔플 순서도 결과에 영향을 주므로 같은 seed 숫자만으로 실행 간 동일성을 보장하지 않습니다."]
    if section == "final":
        final = snapshot["final"]
        rows = []
        for title, key in (("배치 평가 엔진", "batch"), ("제출 CLI CSV", "cli")):
            r = final[key]
            cm = r["confusion_matrix"]
            rows.append([title, f"{r['accuracy'] * 100:.4f}%", f"{r['macro_f1']:.4f}",
                         f"{r['correct']:,} / {r['n']:,}",
                         f"{cm[0][0] / sum(cm[0]) * 100:.2f}%",
                         f"{cm[1][1] / sum(cm[1]) * 100:.2f}%", r["batch_size"]])
        notes = ["3모델 균등 평균(각 1/3), 임계값 0.50; 클래스 0=상황실, 1=신고자.",
                 f"평가 {final['batch']['n']:,}발화 / CSV {final['csv_rows']:,}행 / 음원 누락 대체 {final['fallback_rows']}행.",
                 f"예측 일치 {final['identical_predictions']:,}건, 차이 {final['differing_predictions']}건. 차이 원인은 단계별 수치 대조가 없어 확정하지 않았습니다.",
                 f"혼동행렬(행=정답, 열=예측; 순서 0,1): 배치 {final['batch']['confusion_matrix']} / CLI {final['cli']['confusion_matrix']}."]
        return "최종 제출 모델 성적", _table(
            ["실행", "Accuracy", "Macro F1", "정답 / 분모", "상황실 Recall", "신고자 Recall", "Batch"], rows), notes
    if section == "hypotheses":
        rows = [[h["id"], h["change"],
                 (f"{h['mean_delta_accuracy_pp']:+.4f} → {h['confirmation_mean_delta_accuracy_pp']:+.4f}%p"
                  if "confirmation_mean_delta_accuracy_pp" in h else f"{h['mean_delta_accuracy_pp']:+.4f}%p"),
                 h["scope"], h["decision"], h["reason"]] for h in snapshot["hypotheses"]]
        notes = ["H1은 초기 3초 설정의 후속 재확인입니다. 200통화 Tiny CNN의 짧은 입력 개선 가능성을 128-Mel ResNet 전체 미세조정에서 다시 비교했습니다.",
                 "500통화 기준: 두 seed 모두 ΔAccuracy>0, 평균 ≥+0.5%p, 평균 ΔMacro F1≥0, 각 클래스 Recall 하락 <1.0%p. Centroid 별도 실험은 양쪽 CI 하한>0도 요구했습니다.",
                 "1,000통화 재확인은 마지막 epoch 5를 고정하고 양쪽 Accuracy CI 하한>0도 요구했습니다. 이 기준은 팀의 채택 정책입니다.",
                 "RMS·Bandpass는 정확도가 상승했습니다. 미채택은 효과 부재를 뜻하지 않으며, 감정·채널·정보 손실을 실패 원인으로 직접 증명한 실험도 아닙니다."]
        return "다섯 음성 특화 조건과 채택 판단", _table(
            ["가설", "변경", "평균 ΔAccuracy", "비교 범위", "결정", "미충족 기준"], rows), notes
    if section == "zcr":
        rows = [[r["seed"], f"{r['baseline_accuracy'] * 100:.4f}%",
                 f"{r['candidate_accuracy'] * 100:.4f}%", f"{r['delta_accuracy_pp']:+.4f}%p",
                 f"[{r['ci_95_pp'][0]:+.4f}, {r['ci_95_pp'][1]:+.4f}]%p",
                 f"{r['delta_recall_0_pp']:+.4f}%p", f"{r['delta_recall_1_pp']:+.4f}%p"]
                for r in snapshot["zcr_confirmation"]["seeds"]]
        notes = ["신규 1,000통화: Train 784통화·24,144발화 / Dev 216통화·6,436발화. 이전 500통화와 중복 0.",
                 f"두 seed 평균 ΔAccuracy: {sum(r['delta_accuracy_pp'] for r in snapshot['zcr_confirmation']['seeds']) / 2:+.4f}%p.",
                 "같은 분할에서 학습 seed 42·43을 반복했습니다. 2,000회 bootstrap은 통화 재표집 횟수이며 학습 횟수가 아닙니다.",
                 "판정: seed별 개선 방향이 다르고 Recall·CI 기준을 충족하지 않아 미채택. 새 특징을 적용한 전체 데이터 본 학습은 진행하지 않았습니다."]
        return "ZCR 확대 재확인: 고정 epoch 5", _table(
            ["Seed", "기준선", "ZCR", "ΔAccuracy", "95% 통화 CI", "ΔRecall 0", "ΔRecall 1"], rows), notes
    if section == "checkpoints":
        rows = [[m["name"], f"{m['parameters']:,}", m["filename"], m["sha256"]]
                for m in snapshot["models"]]
        return "최종 모델 식별 정보", _table(["모델", "Total / Active", "파일", "SHA-256 (기록)"], rows), [
            "Total과 Active는 이 밀집 모델에서 같습니다. 합계 31,866,563개.",
            "가중치는 별도 제출하며, 실행 전에 이 SHA-256과 대조합니다."]
    raise ValueError(f"Unknown result section: {section}")


class RecordedView:
    """Plain-text output in Python; HTML table as the last expression in Jupyter."""
    def __init__(self, snapshot, section):
        self.title, (self.headers, self.rows), self.notes = _section(snapshot, section)

    def __repr__(self):
        lines = [self.title, " | ".join(self.headers)]
        lines.extend(" | ".join(row) for row in self.rows)
        lines.extend(f"- {note}" for note in self.notes)
        return "\n".join(lines)

    def _repr_html_(self):
        esc = html.escape
        head = "".join(f"<th style='padding:8px;background:#eaf4fb;border:1px solid #d7e2ea'>{esc(v)}</th>" for v in self.headers)
        rows = "".join("<tr>" + "".join(f"<td style='padding:8px;border:1px solid #d7e2ea'>{esc(v)}</td>" for v in row) + "</tr>" for row in self.rows)
        notes = "".join(f"<li>{esc(v)}</li>" for v in self.notes)
        return ("<div style='font-family:system-ui,sans-serif;line-height:1.5'>"
                f"<h3>{esc(self.title)}</h3><div style='overflow-x:auto'>"
                "<table style='border-collapse:collapse;text-align:left'>"
                f"<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"
                f"<ul>{notes}</ul></div>")


def view(snapshot, section):
    return RecordedView(snapshot, section)
