"""T1: Fit 수동 검토 허용 목록에 한정한 오심 음성 Loss 마스킹.

입력/정답/샘플링은 변경하지 않는다. 추론에서는 이 모듈을 사용하지 않는다.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import re
from pathlib import Path

import torch

from .data import prefix_texts

NAUSEA = 6
VERSION = "t1-fit-reviewed-v1"
# 이전 탐색의 255건을 재구성하는 용도로만 사용. 마스킹 규칙이 아니다.
SCREEN = re.compile(r"메스껍|메스꺼|메슥|미식거|울렁|구역질|토할\s*(?:것|거|꺼)|토\s*나올\s*(?:것|거)")
SCREEN_NEG = re.compile(r"안\s*(?:아프|아파|어지|토하|울렁)|아프지\s*않|없(?:어요|습니다|고|는데|다)|아니|안\s*했|안\s*해|괜찮|없대|안\s*나|안\s*그")
# 실제 구토/구역질만으로는 선택하지 않는다.
EVIDENCE = re.compile(r"메스껍|메스꺼|메슥|미식거|울렁|토할\s*(?:것|거|꺼)|토\s*나올\s*(?:것|거)")
DENIAL = re.compile(r"(?:울렁\w*|메스\w*|미식\w*|오심|토할.{0,8})(?:.{0,6})(?:않|없|아니|괜찮|가라앉|해소)|안\s*(?:울렁|메스|미식)|(?:이제|지금은).{0,10}(?:괜찮|가라앉|해소)")
PAST = re.compile(r"그랬|했었|예전에|지난번|그때|어제는|나았|가라앉|해소|괜찮아졌")
OTHER = re.compile(r"다른\s*(?:사람|환자)|저\s*말고|제가\s*(?:아니라|아닌)|환자.{0,6}아니")

# 255건 관련 발화 선별검토 중 30건을 추가 문맥 검토. 5건은 환자/의미가 불명확하여 제외.
DEEP_REVIEW = [28, 33, 56, 67, 69, 78, 79, 104, 106, 113, 130, 140, 141, 144,
               155, 156, 161, 164, 168, 174, 176, 178, 181, 182, 184, 188, 194, 196, 198, 245]
EXCLUDED = {113: "울렁거림의 의미가 호흡/가슴 불편과 구분 불명확",
            164: "증상 최초 발화 당시 대상 환자 확인이 불충분",
            178: "신고자/보호자/환자 사이 발화 주체 전환이 불명확",
            182: "신고자와 다른 대상 환자의 진술이 섞여 주체 불명확",
            184: "환자 소개와 일인칭 호소 사이 주체 불명확"}


def sha(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def text_sha(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def valid_evidence(text, speaker, target_confirmed):
    """수동 검토된 환자 귀속 + 현재의 평서형 증상 근거를 모두 요구한다."""
    if not target_confirmed or speaker != 1 or DENIAL.search(text) or PAST.search(text) or OTHER.search(text):
        return False
    return any(EVIDENCE.search(m.group()) and "?" not in m.group()
               and not re.search(r"나요|습니까|거야|거냐", m.group())
               for m in re.finditer(r"[^.!?]+[.!?]?", text))


def screening(calls):
    return [c for c in calls if not c.labels[NAUSEA] and any(
        sp == 1 and SCREEN.search(s) and not SCREEN_NEG.search(s) for s, sp in zip(c.texts, c.speakers))]


def make_review(calls):
    """Fit만 전달받는다. 후보 순서는 동결된 캐시의 순서를 따른다."""
    candidates = screening(calls)
    if len(candidates) != 255:
        raise ValueError("Fit 후보가 동결된 255건과 다릅니다")
    records = []
    for i, call in enumerate(candidates):
        decision = "not_deep_reviewed"
        reason = "관련 발화 선별검토만 완료: 주체/시제 전체 문맥 미확정으로 보수적 제외"
        evidence = []
        if i in EXCLUDED:
            decision, reason = "excluded_ambiguous", EXCLUDED[i]
        elif i in DEEP_REVIEW:
            decision, reason = "accepted", "동일 환자의 현재 오심성 평서 진술을 문맥 검토"
            # 전체 후보 자동확장이 아닌 검토 완료된 통화에만 적용한다.
            evidence = [{"step": j, "text_sha256": text_sha(s), "target_confirmed": True}
                        for j, (s, sp) in enumerate(zip(call.texts, call.speakers), 1)
                        if valid_evidence(s, sp, True)]
            if not evidence:
                raise ValueError(f"검토된 증상 근거가 없음: 후보 {i}")
        records.append({"candidate_index": i, "call_id": call.call_id, "decision": decision,
                        "reason": reason, "evidence": evidence})
    return records


def maskable_prefixes(call, record, tokenizer):
    """현재 Prefix까지 관측한 검토 근거만 사용. 512토큰에서 잘린 근거는 제외."""
    if call.labels[NAUSEA] or record.get("decision") != "accepted":
        return set()
    reviewed = {e["step"]: e for e in record["evidence"]}
    prefixes = prefix_texts(call, "space")
    active_end, allowed = None, set()
    for step, (text, speaker) in enumerate(zip(call.texts, call.speakers), 1):
        if speaker == 1 and (DENIAL.search(text) or OTHER.search(text)):
            active_end = None
        if step in reviewed:
            evidence = reviewed[step]
            if evidence["text_sha256"] != text_sha(text):
                raise ValueError("검토 근거 발화가 변경됨")
            if valid_evidence(text, speaker, evidence["target_confirmed"]):
                end = max(m.end() for m in EVIDENCE.finditer(text.strip()))
                # prefix_texts는 각 발화를 strip 후 공백 하나로 연결한다.
                active_end = len(prefixes[step - 1]) - len(text.strip()) + end
        if active_end is not None:
            encoded = tokenizer(prefixes[step - 1], truncation=True, max_length=512,
                                return_offsets_mapping=True)
            visible_end = max((b for a, b in encoded["offset_mapping"]), default=0)
            if active_end <= visible_end:
                allowed.add(step)
    return allowed


def load_plan(path, calls, tokenizer):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest["version"] != VERSION:
        raise ValueError("지원하지 않는 T1 규칙 버전")
    call_map = {c.call_id: c for c in calls}
    if sorted(call_map) != sorted(manifest["fit_ids"]):
        raise ValueError("마스킹 파일은 기존 Fit 전체에만 적용 가능")
    if manifest["review"] != make_review(calls):
        raise ValueError("Fit 수동 검토 허용 목록 또는 근거 발화가 동결본과 다릅니다")
    plan = {}
    for record in manifest["review"]:
        if record["call_id"] not in call_map:
            raise ValueError("Fit 이외 통화가 마스킹 후보에 포함됨")
        plan[record["call_id"]] = maskable_prefixes(call_map[record["call_id"]], record, tokenizer)
    return plan


def loss_mask(labels, call_ids, lengths, plan):
    mask = torch.ones_like(labels)
    for i, (call_id, length) in enumerate(zip(call_ids, lengths)):
        if labels[i, NAUSEA].item() == 0 and length in plan.get(call_id, set()):
            mask[i, NAUSEA] = 0
    return mask


class MaskStats:
    def __init__(self):
        self.epochs = []
        self.unique = set()

    def start(self, epoch):
        self.current = {"epoch": epoch, "exposures": 0, "negative_nausea": 0,
                        "masked": 0, "length_distribution": Counter()}

    def update(self, labels, ids, lengths, mask):
        marked = mask[:, NAUSEA].detach().cpu().numpy() == 0
        self.current["exposures"] += len(ids)
        self.current["negative_nausea"] += int((labels[:, NAUSEA] == 0).sum().item())
        self.current["masked"] += int(marked.sum())
        for call_id, length, on in zip(ids, lengths, marked):
            if on:
                self.unique.add(call_id)
                self.current["length_distribution"][int(length)] += 1

    def finish(self):
        self.current.update({
            "masked_prefix_fraction": self.current["masked"] / max(self.current["exposures"], 1),
            "masked_negative_nausea_fraction": self.current["masked"] / max(self.current["negative_nausea"], 1),
            "masked_all_loss_elements_fraction": self.current["masked"] / max(9 * self.current["exposures"], 1)})
        self.epochs.append(dict(self.current))

    def report(self):
        exposures = sum(e["exposures"] for e in self.epochs)
        negative = sum(e["negative_nausea"] for e in self.epochs)
        masked = sum(e["masked"] for e in self.epochs)
        return {"epochs": self.epochs, "total_exposures": exposures, "masked_prefix_exposures": masked,
                "masked_prefix_fraction": masked / max(exposures, 1),
                "masked_negative_nausea_fraction": masked / max(negative, 1),
                "masked_all_loss_elements_fraction": masked / max(9 * exposures, 1),
                "unique_masked_calls": len(self.unique),
                "normalization": "masked element losses mean over original B*9"}
