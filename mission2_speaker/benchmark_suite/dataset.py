import os
import json
import glob
import random
import torch
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np

class UniversalSpeakerDataset(Dataset):
    """
    미션 2 화자 분류를 위한 고성능 통합 데이터셋.
    
    대회 규칙 준수:
    - 오직 startAt, endAt 음성 조각만 로드 (전사 텍스트 절대 사용 안함)
    - 0: 119대원, 1: 신고자
    
    입력 타입 지원:
    - 'mel_spec': 2D Mel-Spectrogram (1, n_mels, time) -> ResNet, SSAST용
    - 'fbank': Log Mel-Filterbank (n_mels, time) -> ECAPA, ReDimNet, CAM++용
    - 'waveform': 1D Raw Waveform (1, samples) -> HuBERT용
    """
    def __init__(self, data_dir, model_name="resnet50", config_dict=None, 
                 is_train=True, max_files=None, sample_rate=16000, target_duration=3.0):
        super().__init__()
        self.data_dir = data_dir
        self.is_train = is_train
        self.sample_rate = sample_rate
        self.target_samples = int(sample_rate * target_duration)
        
        # 모델별 설정 반영
        cfg = config_dict or {}
        self.input_type = cfg.get("input_type", "mel_spec")
        self.n_mels = cfg.get("n_mels", 128)
        self.n_fft = cfg.get("n_fft", 2048)
        self.hop_length = cfg.get("hop_length", 512)
        
        # 변환기 (Transform) 사전 초기화 (속도 최적화)
        if self.input_type == "mel_spec":
            self.transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0
            )
            self.amplitude_to_db = T.AmplitudeToDB(top_db=80.0)
        elif self.input_type == "fbank":
            self.transform = T.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0
            )
        else:
            self.transform = None

        # 데이터 증강 (Augmentation) 초기화
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

        # 발화(Utterance) 메타데이터 수집
        self.samples = self._load_utterance_metadata(max_files)

    def _load_utterance_metadata(self, max_files=None):
        # json 및 wav 파일 매핑 검색 (하위 폴더 재귀 검색)
        json_pattern = os.path.join(self.data_dir, "**/*.json")
        json_files = sorted(glob.glob(json_pattern, recursive=True))
        
        if max_files and len(json_files) > max_files:
            random.seed(42)
            json_files = random.sample(json_files, max_files)

        samples = []
        for j_path in json_files:
            # 대응되는 wav 파일 경로 탐색
            base_name = os.path.splitext(os.path.basename(j_path))[0]
            # 상위 폴더 구조가 2.라벨링데이터 <-> 1.원천데이터 형태인지 자동 확인
            w_candidate = j_path.replace("2.라벨링데이터", "1.원천데이터").replace("TL_", "TS_").replace(".json", ".wav")
            if not os.path.exists(w_candidate):
                # 일반적인 audio 폴더 또는 동일 폴더 검색 (운영체제별 슬래시 처리)
                w_candidate = j_path.replace("/label/", "/audio/").replace("\\label\\", "\\audio\\").replace(".json", ".wav")
                if not os.path.exists(w_candidate):
                    # 같은 폴더 내 .wav
                    w_candidate = os.path.splitext(j_path)[0] + ".wav"
                    if not os.path.exists(w_candidate):
                        continue

            try:
                with open(j_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                continue

            # JSON 내 발화 정보 파싱 (다양한 AI-Hub 포맷 방어)
            dialogs = []
            if "dialogs" in meta:
                dialogs = meta["dialogs"]
            elif "utterances" in meta:
                dialogs = meta["utterances"]
            elif "dialogue" in meta:
                dialogs = meta["dialogue"]
            elif "annotations" in meta:
                dialogs = meta["annotations"]

            for item in dialogs:
                # 시작, 종료 시간 추출
                start_sec = item.get("startAt") or item.get("start_time") or item.get("start")
                end_sec = item.get("endAt") or item.get("end_time") or item.get("end")
                speaker = item.get("speaker")

                if start_sec is None or end_sec is None or speaker is None:
                    continue

                try:
                    start_sec = float(start_sec)
                    end_sec = float(end_sec)
                except ValueError:
                    continue

                duration = end_sec - start_sec
                if duration <= 0.1:  # 0.1초 이하는 유효한 음성 정보 없음
                    continue

                # 화자 라벨 매핑 (0: 대원/Agent/기타, 1: 신고자/Caller)
                spk_str = str(speaker).strip().lower()
                if spk_str in ["1", "신고자", "caller", "c", "patient"]:
                    label = 1
                elif spk_str in ["0", "대원", "수신자", "agent", "a", "operator", "119대원"]:
                    label = 0
                else:
                    try:
                        label = int(speaker)
                    except ValueError:
                        continue

                samples.append({
                    "wav_path": w_candidate,
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "label": label
                })

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        wav_path = item["wav_path"]
        start_sec = item["start_sec"]
        end_sec = item["end_sec"]
        label = item["label"]

        # 1. 오디오 발화 구간 부분 로딩 (I/O 메모리 최적화)
        start_frame = int(start_sec * self.sample_rate)
        num_frames = int((end_sec - start_sec) * self.sample_rate)

        try:
            waveform, sr = torchaudio.load(wav_path, frame_offset=start_frame, num_frames=num_frames)
            if sr != self.sample_rate:
                resampler = T.Resample(sr, self.sample_rate)
                waveform = resampler(waveform)
        except Exception:
            # 로딩 실패 시 0 텐서 반환 (방어)
            waveform = torch.zeros(1, self.target_samples)

        # 모노 채널 변환
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # 2. 길이 표준화 (Random Crop / Center Crop / Zero Padding)
        cur_len = waveform.shape[-1]
        if cur_len < self.target_samples:
            # 짧은 발화 방어: Zero-padding
            pad_amount = self.target_samples - cur_len
            waveform = F.pad(waveform, (0, pad_amount), mode="constant", value=0.0)
        elif cur_len > self.target_samples:
            if self.is_train:
                max_offset = cur_len - self.target_samples
                offset = random.randint(0, max_offset)
            else:
                offset = (cur_len - self.target_samples) // 2
            waveform = waveform[:, offset:offset + self.target_samples]

        # 1D Waveform Data Augmentation (학습 시에만 적용)
        if self.is_train:
            # 1. Random Gain (0.5 ~ 1.5)
            if random.random() < 0.5:
                gain = random.uniform(0.5, 1.5)
                waveform = waveform * gain
            # 2. Gaussian Noise
            if random.random() < 0.5:
                noise = torch.randn_like(waveform) * random.uniform(0.001, 0.01)
                waveform = waveform + noise

        # 3. 모델별 입력 특징(Feature) 변환
        if self.input_type == "mel_spec":
            # 2D Mel-Spectrogram (dB 스케일) -> (1, n_mels, time)
            spec = self.transform(waveform)
            if self.is_train and random.random() < 0.5:
                spec = self.freq_mask(spec)
                spec = self.time_mask(spec)
            spec_db = self.amplitude_to_db(spec)
            # 인스턴스 정규화 (Mean-Std)
            mean = spec_db.mean()
            std = spec_db.std() + 1e-6
            features = (spec_db - mean) / std

        elif self.input_type == "fbank":
            # Log Mel-Filterbank -> (n_mels, time)
            spec = self.transform(waveform)
            if self.is_train and random.random() < 0.5:
                spec = self.freq_mask(spec)
                spec = self.time_mask(spec)
            fbank = torch.log(spec + 1e-6)
            # Cepstral Mean and Variance Normalization (CMVN)
            features = (fbank - fbank.mean(dim=-1, keepdim=True)) / (fbank.std(dim=-1, keepdim=True) + 1e-6)
            features = features.squeeze(0) # (n_mels, time)

        else: # "waveform"
            # 정규화된 1D Waveform (samples,)
            waveform = waveform.squeeze(0)
            features = (waveform - waveform.mean()) / (waveform.std() + 1e-6)

        return features, torch.tensor(label, dtype=torch.long)
