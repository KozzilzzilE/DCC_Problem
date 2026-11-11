"""발화를 하나씩 받아 증상을 예측하는 스트리밍 추론기. inference.py 의 Predictor 가 이 클래스를 쓴다.

예측은 그 순간까지 받은 발화의 전사와 화자에만 의존한다. 통화 전체 길이나 뒤 발화는 알 수 없다.
`reset()` 은 누적 상태를 모두 지우므로 통화 순서를 바꿔도 결과가 같다.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from . import SYMPTOM_CLASSES
from .bundle import load_bundle
from .data import format_utterance
from .decision import calibrate, expected_rule, make_rule


class StreamingSymptomModel:
    def __init__(self, ckpt_path: str, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        want_cuda = str(device).startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(device if want_cuda else "cpu")
        self.members, self.settings = load_bundle(ckpt_path, self.device)
        shift = self.settings.get("logit_shift")
        self._rule = make_rule(self.settings["rule"], shift)                                  # 4-1 결정 규칙
        self._halt_rule = make_rule(self.settings.get("halt_rule") or self.settings["rule"], shift)  # 4-2 중단 시점 규칙
        self.reset()

    def reset(self) -> None:
        self._parts: List[str] = []        # 지금까지 받은 발화 (형식 적용 뒤)
        self._n_seen = 0                   # 지금까지 받은 발화 수
        self._probs: Optional[np.ndarray] = None
        self._stable_pred: Optional[tuple] = None
        self._stable_run = 0

    # ------------------------------------------------------------------ 확률
    def push(self, text: str, speaker: int) -> np.ndarray:
        """발화 하나를 누적하고, 누적 전사 전체에 대한 9종 확률을 돌려준다 (멤버 평균)."""
        torch = self._torch
        piece = format_utterance(text, speaker, self.settings["fmt"])
        if piece:
            self._parts.append(piece)
        self._n_seen += 1
        joined = " ".join(self._parts)
        total = np.zeros(len(SYMPTOM_CLASSES), dtype=np.float64)
        with torch.inference_mode():
            for tokenizer, model in self.members:
                enc = tokenizer(joined, truncation=True, max_length=int(self.settings["max_length"]),
                                return_tensors="pt").to(self.device)
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
                total += torch.sigmoid(logits.float())[0].cpu().numpy()
        self._probs = (total / len(self.members)).astype(np.float32)
        return self._probs

    # ------------------------------------------------------------------ 결정
    def symptoms(self, at_halt: bool = False) -> List[str]:
        """현재 확률에 결정 규칙을 적용한 증상 리스트. at_halt=True 면 4-2 중단 시점용 규칙을 쓴다."""
        if self._probs is None:
            return []
        rule = self._halt_rule if at_halt else self._rule
        row = rule(self._probs[None, :].copy())[0]
        return [c for c, on in zip(SYMPTOM_CLASSES, row) if on]

    def should_halt(self) -> bool:
        """4-2: 지금 중단할지. 현재와 과거 확률만 본다."""
        halt: Dict[str, object] = self.settings["halt"]
        min_step = int(halt.get("min_step", 5))
        if halt["type"] == "confidence":
            return self._n_seen >= min_step and float(self._probs.max()) >= float(halt["tau"])
        if halt["type"] == "expected":
            # 지금 답했을 때의 기대 순간 점수(보정 확률 기준)가 tau 이상이면 중단한다
            value = expected_rule(calibrate(self._probs[None, :], self.settings.get("logit_shift")))[1][0]
            return self._n_seen >= min_step and float(value) >= float(halt["tau"])
        if halt["type"] == "stable":
            pred = tuple(self._probs >= float(halt.get("threshold", 0.5)))
            if any(pred) and pred == self._stable_pred:
                self._stable_run += 1
            else:
                self._stable_run = 1 if any(pred) else 0
            self._stable_pred = pred
            return self._n_seen >= min_step and self._stable_run >= int(halt.get("patience", 2))
        if halt["type"] == "fixed":
            return self._n_seen >= int(halt["k"])
        if halt["type"] == "never":
            return False
        raise ValueError(f"알 수 없는 중단 기준: {halt['type']}")
