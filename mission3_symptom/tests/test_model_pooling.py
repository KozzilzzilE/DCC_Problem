"""CLS baseline 보존과 RoBERTa label-wise attention pooling 단위 테스트."""

from __future__ import annotations

import gc
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch
from transformers import RobertaConfig
from transformers.models.roberta.modeling_roberta import RobertaForSequenceClassification

MISSION3_DIR = Path(__file__).resolve().parents[1]
if str(MISSION3_DIR) not in sys.path:
    sys.path.insert(0, str(MISSION3_DIR))

from m3.config import NUM_CLASSES
from m3.losses import build_loss
from m3.model import (
    CLS_POOLING,
    LABEL_ATTENTION_POOLING,
    RobertaForLabelWiseAttentionClassification,
    build_tokenizer_and_model,
    load_saved_model,
    validate_pooling_type,
)
from m3.training import TrainingConfig, _validate_config
import train as train_entry


def make_config() -> RobertaConfig:
    return RobertaConfig(
        vocab_size=64,
        hidden_size=24,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        max_position_embeddings=32,
        num_labels=NUM_CLASSES,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        classifier_dropout=0.0,
        problem_type="multi_label_classification",
    )


def make_inputs(batch_size: int = 2, sequence_length: int = 7):
    input_ids = torch.randint(3, 63, (batch_size, sequence_length))
    attention_mask = torch.tensor(
        [[1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0]],
        dtype=torch.long,
    )
    return input_ids, attention_mask


class ModelPoolingTest(unittest.TestCase):
    def test_default_pooling_is_cls_in_config_and_cli(self) -> None:
        config = TrainingConfig("train.csv", "val.csv", "output")
        self.assertEqual(config.pooling_type, CLS_POOLING)

        argv = [
            "train.py",
            "--train-csv",
            "train.csv",
            "--val-csv",
            "val.csv",
            "--output-dir",
            "output",
        ]
        with patch.object(sys, "argv", argv):
            default_cli = train_entry.build_config(train_entry.parse_args())
        with patch.object(
            sys,
            "argv",
            [*argv, "--pooling-type", LABEL_ATTENTION_POOLING],
        ):
            attention_cli = train_entry.build_config(train_entry.parse_args())

        self.assertEqual(default_cli.pooling_type, CLS_POOLING)
        self.assertEqual(attention_cli.pooling_type, LABEL_ATTENTION_POOLING)

    def test_cls_forward_logits_shape_is_unchanged(self) -> None:
        model = RobertaForSequenceClassification(make_config()).eval()
        input_ids, attention_mask = make_inputs()
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
        self.assertEqual(output.logits.shape, (2, NUM_CLASSES))

    def test_label_attention_shapes_and_padding_mask(self) -> None:
        model = RobertaForLabelWiseAttentionClassification(make_config()).eval()
        input_ids, attention_mask = make_inputs()
        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)

        weights = output.label_attention_weights
        self.assertEqual(output.logits.shape, (2, NUM_CLASSES))
        self.assertEqual(weights.shape, (2, NUM_CLASSES, input_ids.shape[1]))
        self.assertTrue(
            torch.allclose(weights.sum(dim=-1), torch.ones(2, NUM_CLASSES), atol=1e-6)
        )
        padding_weights = weights.masked_select(~attention_mask[:, None, :].bool())
        self.assertTrue(torch.allclose(padding_weights, torch.zeros_like(padding_weights)))

    def test_padding_hidden_values_do_not_change_logits(self) -> None:
        model = RobertaForLabelWiseAttentionClassification(make_config()).eval()
        _, attention_mask = make_inputs()
        inputs_embeds = torch.randn(2, attention_mask.shape[1], model.config.hidden_size)
        changed_embeds = inputs_embeds.clone()
        changed_embeds[~attention_mask.bool()] = 10_000.0

        with torch.no_grad():
            original = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            ).logits
            changed = model(
                inputs_embeds=changed_embeds,
                attention_mask=attention_mask,
            ).logits
        self.assertTrue(torch.allclose(original, changed, atol=1e-6))

    def test_all_padding_input_fails_fast(self) -> None:
        model = RobertaForLabelWiseAttentionClassification(make_config())
        input_ids = torch.ones((1, 4), dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        with self.assertRaisesRegex(ValueError, "모든 token이 padding"):
            model(input_ids=input_ids, attention_mask=attention_mask)

    def test_queries_and_existing_classifier_receive_gradients(self) -> None:
        model = RobertaForLabelWiseAttentionClassification(make_config())
        input_ids, attention_mask = make_inputs()
        targets = torch.randint(0, 2, (2, NUM_CLASSES), dtype=torch.float32)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        loss = build_loss("bce")(logits, targets)
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertIsNotNone(model.label_queries.weight.grad)
        self.assertIsNotNone(model.classifier.dense.weight.grad)
        self.assertIsNotNone(model.classifier.out_proj.weight.grad)
        self.assertGreater(model.label_queries.weight.grad.abs().sum().item(), 0.0)

    def test_bce_pos_weight_and_asl_backward(self) -> None:
        input_ids, attention_mask = make_inputs()
        targets = torch.randint(0, 2, (2, NUM_CLASSES), dtype=torch.float32)
        losses = (
            build_loss("bce"),
            build_loss("bce", pos_weight=torch.full((NUM_CLASSES,), 2.0)),
            build_loss("asl"),
        )
        for loss_fn in losses:
            with self.subTest(loss=type(loss_fn).__name__):
                model = RobertaForLabelWiseAttentionClassification(make_config())
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                loss = loss_fn(logits, targets)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertIsNotNone(model.label_queries.weight.grad)

    def test_parameter_delta_is_only_label_queries(self) -> None:
        cls_model = RobertaForSequenceClassification(make_config())
        attention_model = RobertaForLabelWiseAttentionClassification(make_config())
        cls_parameters = sum(parameter.numel() for parameter in cls_model.parameters())
        attention_parameters = sum(
            parameter.numel() for parameter in attention_model.parameters()
        )
        self.assertEqual(
            attention_parameters - cls_parameters,
            NUM_CLASSES * make_config().hidden_size,
        )

    def test_label_attention_save_load_preserves_config_weights_and_logits(self) -> None:
        model = RobertaForLabelWiseAttentionClassification(make_config()).eval()
        input_ids, attention_mask = make_inputs()
        with torch.no_grad():
            expected = model(input_ids=input_ids, attention_mask=attention_mask).logits

        with tempfile.TemporaryDirectory() as directory:
            model.save_pretrained(directory, safe_serialization=True)
            reloaded = RobertaForLabelWiseAttentionClassification.from_pretrained(
                directory,
                local_files_only=True,
            ).eval()
            with torch.no_grad():
                actual = reloaded(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits

            self.assertEqual(reloaded.config.pooling_type, LABEL_ATTENTION_POOLING)
            self.assertIn("label_queries.weight", reloaded.state_dict())
            self.assertTrue(torch.equal(expected, actual))
            del reloaded
            gc.collect()

    def test_old_config_without_pooling_field_uses_cls_loader(self) -> None:
        config = make_config()
        self.assertFalse(hasattr(config, "pooling_type"))
        tokenizer = MagicMock()
        cls_model = RobertaForSequenceClassification(config)

        with tempfile.TemporaryDirectory() as directory, patch(
            "m3.model.AutoConfig.from_pretrained", return_value=config
        ), patch(
            "m3.model.AutoTokenizer.from_pretrained", return_value=tokenizer
        ), patch(
            "m3.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=cls_model,
        ) as cls_loader, patch(
            "m3.model.validate_tokenizer_model_compatibility"
        ):
            _, loaded = load_saved_model(directory)

        cls_loader.assert_called_once()
        self.assertIs(loaded, cls_model)

    def test_saved_label_attention_config_uses_custom_loader(self) -> None:
        config = make_config()
        config.pooling_type = LABEL_ATTENTION_POOLING
        tokenizer = MagicMock()
        attention_model = RobertaForLabelWiseAttentionClassification(config)

        with tempfile.TemporaryDirectory() as directory, patch(
            "m3.model.AutoConfig.from_pretrained", return_value=config
        ), patch(
            "m3.model.AutoTokenizer.from_pretrained", return_value=tokenizer
        ), patch(
            "m3.model.RobertaForLabelWiseAttentionClassification.from_pretrained",
            return_value=attention_model,
        ) as attention_loader, patch(
            "m3.model.validate_tokenizer_model_compatibility"
        ):
            _, loaded = load_saved_model(directory)

        attention_loader.assert_called_once()
        self.assertIs(loaded, attention_model)

    def test_kobert_and_koelectra_cls_still_use_auto_sequence_classifier(self) -> None:
        dummy_model = MagicMock()
        # KoBERT tokenizer 는 sentencepiece 의존성을 피하려고 지연 import 한다.
        # 제출 모델(KLUE-RoBERTa)이 없는 패키지 때문에 죽지 않아야 하므로 로더를 대신 patch 한다.
        with patch(
            "m3.model._kobert_tokenizer_class", return_value=MagicMock()
        ), patch(
            "m3.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=dummy_model,
        ) as cls_loader, patch(
            "m3.model.validate_tokenizer_model_compatibility"
        ):
            _, loaded = build_tokenizer_and_model(
                "skt/kobert-base-v1",
                pooling_type=CLS_POOLING,
            )
            self.assertIs(loaded, dummy_model)
            cls_loader.assert_called_once()

        with patch(
            "m3.model.AutoTokenizer.from_pretrained", return_value=MagicMock()
        ), patch(
            "m3.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=dummy_model,
        ) as cls_loader, patch(
            "m3.model.validate_tokenizer_model_compatibility"
        ):
            _, loaded = build_tokenizer_and_model(
                "monologg/koelectra-base-v3-discriminator",
                pooling_type=CLS_POOLING,
            )
            self.assertIs(loaded, dummy_model)
            cls_loader.assert_called_once()

    def test_invalid_backbone_with_label_attention_fails_fast(self) -> None:
        with patch(
            "m3.model.AutoTokenizer.from_pretrained", return_value=MagicMock()
        ), patch(
            "m3.model.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(model_type="electra"),
        ), patch(
            "m3.model.RobertaForLabelWiseAttentionClassification.from_pretrained"
        ) as attention_loader:
            with self.assertRaisesRegex(ValueError, "RoBERTa 계열만 지원"):
                build_tokenizer_and_model(
                    "monologg/koelectra-base-v3-discriminator",
                    pooling_type=LABEL_ATTENTION_POOLING,
                )
        attention_loader.assert_not_called()

    def test_kf_deberta_with_label_attention_fails_fast(self) -> None:
        with patch(
            "m3.model.AutoConfig.from_pretrained",
            return_value=SimpleNamespace(model_type="deberta-v2"),
        ), patch(
            "m3.model.AutoTokenizer.from_pretrained"
        ) as tokenizer_loader, patch(
            "m3.model.RobertaForLabelWiseAttentionClassification.from_pretrained"
        ) as attention_loader:
            with self.assertRaisesRegex(ValueError, "RoBERTa 계열만 지원"):
                build_tokenizer_and_model(
                    "kakaobank/kf-deberta-base",
                    pooling_type=LABEL_ATTENTION_POOLING,
                )
        tokenizer_loader.assert_not_called()
        attention_loader.assert_not_called()

    def test_kf_deberta_cls_uses_generic_auto_path(self) -> None:
        tokenizer = MagicMock()
        model = MagicMock()
        with patch(
            "m3.model.AutoTokenizer.from_pretrained", return_value=tokenizer
        ) as tokenizer_loader, patch(
            "m3.model.AutoModelForSequenceClassification.from_pretrained",
            return_value=model,
        ) as model_loader, patch(
            "m3.model.validate_tokenizer_model_compatibility"
        ):
            loaded_tokenizer, loaded_model = build_tokenizer_and_model(
                "kakaobank/kf-deberta-base",
                pooling_type=CLS_POOLING,
            )

        self.assertIs(loaded_tokenizer, tokenizer)
        self.assertIs(loaded_model, model)
        tokenizer_loader.assert_called_once()
        model_loader.assert_called_once()
        self.assertEqual(model_loader.call_args.kwargs["num_labels"], NUM_CLASSES)
        self.assertEqual(
            model_loader.call_args.kwargs["problem_type"],
            "multi_label_classification",
        )

    def test_invalid_pooling_type_fails_fast(self) -> None:
        with self.assertRaisesRegex(ValueError, "지원하지 않는 pooling_type"):
            validate_pooling_type("mean")
        with self.assertRaisesRegex(ValueError, "지원하지 않는 pooling_type"):
            _validate_config(
                TrainingConfig(
                    "train.csv",
                    "val.csv",
                    "output",
                    pooling_type="mean",
                )
            )


if __name__ == "__main__":
    unittest.main()
