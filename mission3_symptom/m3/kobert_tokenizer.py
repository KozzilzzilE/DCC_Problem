"""SentencePiece 기반 KoBERT tokenizer."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sentencepiece as spm
from transformers import PreTrainedTokenizer


VOCAB_FILES_NAMES = {"vocab_file": "spiece.model"}


class KoBertTokenizer(PreTrainedTokenizer):
    """KoBERT의 SentencePiece vocabulary를 BERT 입력 형식으로 변환."""

    vocab_files_names = VOCAB_FILES_NAMES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file: str,
        unk_token: str = "[UNK]",
        sep_token: str = "[SEP]",
        pad_token: str = "[PAD]",
        cls_token: str = "[CLS]",
        mask_token: str = "[MASK]",
        **kwargs,
    ) -> None:
        self.vocab_file = str(vocab_file)
        self.sp_model = spm.SentencePieceProcessor(model_file=self.vocab_file)

        # KoBERT는 NFC 한국어를 그대로 SentencePiece에 전달해야 한다.
        kwargs.pop("keep_accents", None)
        kwargs.pop("bos_token", None)
        kwargs.pop("eos_token", None)
        kwargs.setdefault("model_max_length", 512)
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )

    @property
    def vocab_size(self) -> int:
        return int(self.sp_model.vocab_size())

    def get_vocab(self) -> Dict[str, int]:
        vocab = {
            self.sp_model.id_to_piece(index): index
            for index in range(self.vocab_size)
        }
        vocab.update(self.added_tokens_encoder)
        return vocab

    def _tokenize(self, text: str, **kwargs) -> List[str]:
        return list(self.sp_model.encode(text, out_type=str))

    def _convert_token_to_id(self, token: str) -> int:
        return int(self.sp_model.piece_to_id(token))

    def _convert_id_to_token(self, index: int) -> str:
        return str(self.sp_model.id_to_piece(index))

    def convert_tokens_to_string(self, tokens: List[str]) -> str:
        return str(self.sp_model.decode(tokens))

    def build_inputs_with_special_tokens(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        if token_ids_1 is None:
            return [self.cls_token_id, *token_ids_0, self.sep_token_id]
        return [
            self.cls_token_id,
            *token_ids_0,
            self.sep_token_id,
            *token_ids_1,
            self.sep_token_id,
        ]

    def get_special_tokens_mask(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> List[int]:
        if already_has_special_tokens:
            return super().get_special_tokens_mask(
                token_ids_0,
                token_ids_1=token_ids_1,
                already_has_special_tokens=True,
            )
        if token_ids_1 is None:
            return [1, *([0] * len(token_ids_0)), 1]
        return [
            1,
            *([0] * len(token_ids_0)),
            1,
            *([0] * len(token_ids_1)),
            1,
        ]

    def create_token_type_ids_from_sequences(
        self,
        token_ids_0: List[int],
        token_ids_1: Optional[List[int]] = None,
    ) -> List[int]:
        first_segment_length = len(token_ids_0) + 2
        if token_ids_1 is None:
            return [0] * first_segment_length
        return [0] * first_segment_length + [1] * (len(token_ids_1) + 1)

    def save_vocabulary(
        self,
        save_directory: str,
        filename_prefix: Optional[str] = None,
    ) -> Tuple[str, ...]:
        directory = Path(save_directory)
        if not directory.is_dir():
            raise ValueError(f"tokenizer 저장 경로가 디렉터리가 아닙니다: {directory}")

        filename = VOCAB_FILES_NAMES["vocab_file"]
        if filename_prefix:
            filename = f"{filename_prefix}-{filename}"
        destination = directory / filename
        source = Path(self.vocab_file)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        return (str(destination),)
