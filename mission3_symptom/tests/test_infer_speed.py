"""추론 속도·메모리 안전장치 테스트.

- 긴 본문부터 처리해 GPU 캐시 재사용을 늘린다 (짧은 것부터 하면 배치 모양이 커질 때마다 새 블록을
  잡아 8GB 카드에서 예약 메모리가 4.5GB 까지 부푼다).
- fp16 은 번들(`.pt` 의 `precision`)이나 `--precision` 이 켤 때만, CUDA 에서만 쓴다. 기본은 fp32.
- GPU 메모리가 부족하면 배치를 반으로 나눠 다시 하고(실패한 시도의 메모리는 풀고 나서), 한 건도 안 되면
  또는 CUDA 실행 오류가 나면 CPU 로 내려 끝까지 채운다.
"""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

MISSION3_DIR = Path(__file__).resolve().parents[1]
TESTS_DIR = Path(__file__).resolve().parent
for _path in (MISSION3_DIR, TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from m3 import infer
from m3.bundle import load_bundle
from m3.config import NUM_CLASSES
from test_ensemble_inference import write_fake_bundle


class FakeOutput:
    def __init__(self, logits):
        self.logits = logits


class FakeModel:
    """입력 행마다 결정적인 로짓을 낸다. 배치가 max_rows 를 넘거나 GPU 에 있으면 OOM 을 흉내 낸다.

    실패하기 전에 활성값 텐서를 하나 만들어 두고, 다음 호출 때 이전 실패의 텐서가 아직 살아 있는지 기록한다
    (except 블록 안에서 재시도하면 traceback 이 실패한 forward 의 프레임을 붙잡아 메모리가 풀리지 않는다).
    """

    def __init__(self, max_rows=None, oom_on_gpu=False, error=None):
        self.max_rows = max_rows
        self.oom_on_gpu = oom_on_gpu
        self.on_gpu = oom_on_gpu
        self.error = error or torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        self.calls = []
        self.ok_calls = []
        self.failed_refs = []
        self.alive_at_retry = []

    def to(self, device):
        self.on_gpu = torch.device(device).type == "cuda"
        return self

    def __call__(self, input_ids):
        import gc
        import weakref

        gc.collect()
        self.alive_at_retry.append(sum(ref() is not None for ref in self.failed_refs))
        rows = int(input_ids.shape[0])
        self.calls.append(rows)
        activation = torch.zeros(rows, 64)
        if (self.max_rows is not None and rows > self.max_rows) or (self.oom_on_gpu and self.on_gpu):
            self.failed_refs.append(weakref.ref(activation))
            # 매번 새 예외를 만든다. 같은 객체를 다시 던지면 그 객체의 __traceback__ 이 프레임을 붙잡는다.
            raise type(self.error)(*self.error.args)
        self.ok_calls.append(rows)
        base = input_ids.float().sum(dim=1, keepdim=True)
        return FakeOutput(base / 10.0 - torch.arange(NUM_CLASSES, dtype=torch.float32))


def make_batch(indices, device):
    return {"input_ids": torch.tensor([[i, i + 1, 2 * i] for i in indices], dtype=torch.long)}


def reference(n):
    ids = torch.tensor([[i, i + 1, 2 * i] for i in range(n)], dtype=torch.float32)
    return torch.sigmoid(ids.sum(dim=1, keepdim=True) / 10.0 - torch.arange(NUM_CLASSES)).numpy()


class InferenceOrderTest(unittest.TestCase):
    def test_longest_text_first_and_ties_keep_input_order(self) -> None:
        texts = ["aa", "aaaa", "a", "bbbb", ""]

        self.assertEqual(infer.inference_order(texts), [1, 3, 0, 2, 4])


class PrecisionTest(unittest.TestCase):
    def test_fp16_only_on_cuda(self) -> None:
        self.assertIs(infer.autocast_dtype("fp16", torch.device("cuda")), torch.float16)
        self.assertIsNone(infer.autocast_dtype("fp16", torch.device("cpu")))
        self.assertIsNone(infer.autocast_dtype("fp32", torch.device("cuda")))

    def test_unknown_precision_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "precision"):
            infer.autocast_dtype("int8", torch.device("cuda"))

    def _precision_of(self, precision) -> str:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            return load_bundle(write_fake_bundle(Path(tmp) / "b.pt", ["m"], precision=precision)).precision

    def test_bundle_precision_defaults_to_fp32(self) -> None:
        self.assertEqual(self._precision_of(None), "fp32")   # precision 키가 없는 번들

    def test_bundle_can_opt_into_fp16(self) -> None:
        self.assertEqual(self._precision_of("fp16"), "fp16")

    def test_bundle_rejects_unknown_precision(self) -> None:
        for bad in ("int8", 16, ""):
            with self.subTest(value=bad):
                with self.assertRaisesRegex(ValueError, "precision"):
                    self._precision_of(bad)


class PrecisionPassThroughTest(unittest.TestCase):
    def _captured_precisions(self, bundle_precision, override=None, members=("a", "b")):
        from unittest.mock import patch

        seen = []

        def fake_transformer(label_dir, ckpt_path, batch_size=16, device=None, precision="fp32", loader=None):
            seen.append(precision)
            return ["a.json"], np.zeros((1, NUM_CLASSES))

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            pt = write_fake_bundle(Path(tmp) / "b.pt", list(members), precision=bundle_precision)
            with patch.object(infer, "transformer_probabilities", side_effect=fake_transformer):
                infer.predict_directory(Path(tmp), pt, precision=override)
        return seen

    def test_bundle_precision_reaches_every_member(self) -> None:
        self.assertEqual(self._captured_precisions("fp16"), ["fp16", "fp16"])
        self.assertEqual(self._captured_precisions(None), ["fp32", "fp32"])

    def test_explicit_override_wins_over_bundle(self) -> None:
        self.assertEqual(self._captured_precisions("fp16", override="fp32"), ["fp32", "fp32"])

    def test_empty_override_is_rejected_not_ignored(self) -> None:
        with self.assertRaisesRegex(ValueError, "precision"):
            self._captured_precisions("fp16", override="")

    def test_single_run_directory_defaults_to_fp32(self) -> None:
        from unittest.mock import patch

        seen = []

        def fake_transformer(label_dir, ckpt_path, batch_size=16, device=None, precision="fp32", loader=None):
            seen.append(precision)
            return ["a.json"], np.zeros((1, NUM_CLASSES))

        with patch.object(infer, "transformer_probabilities", side_effect=fake_transformer):
            infer.predict_directory(".", "runs/some_run")
        self.assertEqual(seen, ["fp32"])


class OutOfMemoryFallbackTest(unittest.TestCase):
    def test_full_batch_path_matches_reference(self) -> None:
        model = FakeModel()
        probs, device = infer.predict_all(model, make_batch, list(range(10)), 4, torch.device("cpu"), None)

        np.testing.assert_allclose(probs, reference(10), rtol=1e-6)
        self.assertEqual(device.type, "cpu")

    def test_oom_batches_are_split_and_every_row_is_filled(self) -> None:
        model = FakeModel(max_rows=2)
        order = infer.inference_order(["x" * (i % 5) for i in range(11)])
        probs, _ = infer.predict_all(model, make_batch, order, 8, torch.device("cpu"), None)

        np.testing.assert_allclose(probs, reference(11), rtol=1e-6)
        self.assertTrue(all(rows <= 2 for rows in model.ok_calls))
        self.assertEqual(sum(model.ok_calls), 11)
        self.assertEqual(model.calls[0], 8)

    def test_failed_attempt_memory_is_released_before_retry(self) -> None:
        model = FakeModel(max_rows=2)
        infer.predict_all(model, make_batch, list(range(8)), 8, torch.device("cpu"), None)

        self.assertGreater(len(model.failed_refs), 0)
        self.assertEqual(max(model.alive_at_retry), 0)

    def test_cuda_runtime_errors_fall_back_to_cpu(self) -> None:
        for message in ("CUDA error: out of memory", "CUBLAS_STATUS_ALLOC_FAILED when calling cublasCreate",
                        "CUDA error: no kernel image is available for execution on the device"):
            with self.subTest(message=message):
                model = FakeModel(oom_on_gpu=True, error=RuntimeError(message))
                probs, device = infer.predict_all(model, make_batch, list(range(3)), 2, torch.device("cuda"), None)
                np.testing.assert_allclose(probs, reference(3), rtol=1e-6)
                self.assertEqual(device.type, "cpu")

    def test_other_errors_are_not_swallowed(self) -> None:
        model = FakeModel(max_rows=0, error=RuntimeError("mat1 and mat2 shapes cannot be multiplied"))
        with self.assertRaisesRegex(RuntimeError, "shapes"):
            infer.predict_all(model, make_batch, list(range(3)), 2, torch.device("cpu"), None)

    def test_wrong_row_count_from_model_is_an_error(self) -> None:
        def one_row_batch(indices, device):
            return {"input_ids": torch.tensor([[0, 1, 0]], dtype=torch.long)}

        with self.assertRaisesRegex(RuntimeError, "행 수"):
            infer.predict_all(FakeModel(), one_row_batch, list(range(4)), 4, torch.device("cpu"), None)

    def test_single_row_oom_moves_the_model_to_cpu_and_finishes(self) -> None:
        model = FakeModel(oom_on_gpu=True)
        probs, device = infer.predict_all(model, make_batch, list(range(5)), 2, torch.device("cuda"), torch.float16)

        np.testing.assert_allclose(probs, reference(5), rtol=1e-6)
        self.assertEqual(device.type, "cpu")
        self.assertFalse(model.on_gpu)


if __name__ == "__main__":
    unittest.main()
