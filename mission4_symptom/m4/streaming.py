"""발화를 하나씩 받아 증상을 예측하는 스트리밍 추론기. inference.py 의 Predictor 가 이 클래스를 쓴다.

예측은 그 순간까지 받은 발화의 전사와 화자에만 의존한다. 통화 전체 길이나 뒤 발화는 알 수 없다.
4-2 의 학습형 중단 기준(halt type "learned")은 여기에 그 순간까지 받은 발화의 시각(start_ms, end_ms)을 더 쓴다.
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
        self._configure()

    def _configure(self) -> None:
        """추론 설정(self.settings)에서 결정 규칙과 중단 기준을 만든다."""
        shift = self.settings.get("logit_shift")
        prior = self.settings.get("label_prior")           # maxent 규칙용 (예전 번들에는 없다)
        self._rule = make_rule(self.settings["rule"], shift, prior)                                  # 4-1 결정 규칙
        self._halt_rule = make_rule(self.settings.get("halt_rule") or self.settings["rule"], shift, prior)  # 4-2 중단 시점 규칙
        # 학습형 중단 기준이면 번들에 든 정책(LightGBM 모델 문자열 포함)을 복원한다. 다른 기준에서는 lightgbm 을 읽지 않는다.
        self._policy = None
        self._halt_state = None
        self._halt_x: Optional[np.ndarray] = None
        halt = self.settings["halt"]
        if halt["type"] == "learned":
            from .halting import HaltFeatureState, HaltPolicy

            try:
                self._policy = HaltPolicy.from_dict(halt)
                self._policy.booster()                      # 특징 목록 대조와 모델 복원을 첫 통화 전에 끝낸다
                # 특징의 보정 확률은 정책을 학습할 때 쓴 보정값으로 만든다 (번들 보정값과 같은지는 묶을 때 대조한다)
                self._halt_state = HaltFeatureState(self._policy.logit_shift)
            except ImportError as err:
                # lightgbm 이 없는 환경: 번들에 함께 넣어 둔 대체 기준(기대 점수 >= tau)으로 내려간다.
                # 4-1 은 중단 기준을 쓰지 않으므로 이 경우에도 결과가 같다.
                fallback = halt.get("fallback")
                if not fallback:
                    raise
                import sys

                print(f"[경고] 학습형 중단 정책을 읽지 못했습니다 ({err}). 대체 기준 {fallback} 을 씁니다.", file=sys.stderr)
                self._policy = None
                self.settings["halt"] = dict(fallback)
        self.reset()

    def reset(self) -> None:
        self._parts: List[str] = []        # 지금까지 받은 발화 (형식 적용 뒤)
        self._n_seen = 0                   # 지금까지 받은 발화 수
        self._probs: Optional[np.ndarray] = None
        self._stable_pred: Optional[tuple] = None
        self._stable_run = 0
        self._halt_x = None                # 학습형 중단 기준의 이번 순간 특징
        if self._halt_state is not None:
            self._halt_state.reset()

    # ------------------------------------------------------------------ 확률
    def push(self, text: str, speaker: int, start_ms: int = 0, end_ms: int = 0) -> np.ndarray:
        """발화 하나를 누적하고, 누적 전사 전체에 대한 9종 확률을 돌려준다 (멤버 평균).

        start_ms / end_ms 는 통화 시작 기준 발화 구간(밀리초)이고 학습형 중단 기준의 특징에만 쓴다.
        """
        piece = format_utterance(text, speaker, self.settings["fmt"])
        if piece:
            self._parts.append(piece)
        self._n_seen += 1
        self._probs = self._infer(" ".join(self._parts))
        if self._halt_state is not None:
            self._halt_x = self._halt_state.update(text, speaker, start_ms, end_ms, self._probs)
        return self._probs

    def _infer(self, joined: str) -> np.ndarray:
        """누적 전사 하나에 대한 9종 확률 (멤버 평균, 보정 전)."""
        torch = self._torch
        total = np.zeros(len(SYMPTOM_CLASSES), dtype=np.float64)
        with torch.inference_mode():
            for tokenizer, model in self.members:
                enc = tokenizer(joined, truncation=True, max_length=int(self.settings["max_length"]),
                                return_tensors="pt").to(self.device)
                logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).logits
                total += torch.sigmoid(logits.float())[0].cpu().numpy()
        return (total / len(self.members)).astype(np.float32)

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
        if halt["type"] == "learned":
            # 5번째 발화부터 "기대 점수 + beta * (지금 끊는 쪽의 예상 이득) >= tau" 이면 중단한다 (m4/halting.py)
            return self._policy.should_halt(self._halt_state, self._halt_x)
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
