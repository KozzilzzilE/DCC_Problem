# DCC Mission 2 Experiment Source & Reproduction Map

- Date: 2026-10-11
- Scope: Mapping all acoustic preprocessing and feature hypothesis experiments (200-call exploration, 500-call contrast, 1,000-call confirmation) to exact executable code, hashes, configurations, and reproduction commands.

---

## 1. Experiment Progression Overview

| Stage | Calls | Utterances | Audio / Input Profile | Model / Budget | Primary Purpose | Result |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **200-Call Exploration** | 200 calls | 6,075 (Train 4,913 / Dev 1,162) | 16kHz, 80-Mel (FFT 512, hop 160), float16 | Tiny CNN / Frozen ResNet / Full FT (3-5 epochs) | Low-cost rapid screening of acoustic features | Identified H2(RMS), H3(Centroid), H4(ZCR) as pilot candidates |
| **500-Call Contrast (Centroid)** | 500 calls | 15,332 (Train 12,482 / Dev 2,850) | 16kHz, 128-Mel (FFT 2048, hop 512), float32 | ResNet-50 Full FT, seed 42 & 43, 5 epochs | Test H3 Centroid under official 128-Mel submission specification | Mean Delta Acc -0.4561%p, CI spans 0 -> Rejected H3 |
| **500-Call Contrast (Integrated)** | 500 calls | 15,332 (Train 12,482 / Dev 2,850) | 16kHz, 128-Mel (FFT 2048, hop 512), float32 | ResNet-50 Full FT, seed 42 & 43, 5 epochs | Compare Baseline, H1(Short), H2(RMS), H4(ZCR), H5(Bandpass) | H1 degraded; H2 & H5 failed Class 0 Recall limit; H4 passed gate |
| **1,000-Call Confirmation** | 1,000 fresh calls | 30,580 (Train 24,144 / Dev 6,436) | 16kHz, 128-Mel (FFT 2048, hop 512), float32 (deduplicated) | ResNet-50 Full FT, seed 42 & 43, 5 epochs | Final gate verification of H4 ZCR on fresh sample at fixed Epoch 5 | Seed 42: -1.51%p; Seed 43: +1.17%p (Recall 0 -2.23%p) -> Rejected H4 |

---

## 2. Specification & Design Controls

1. **Audio and Mel Profile**:
   - Sample Rate: 16,000 Hz.
   - Mel Specification (`resnet128`): 128 Mel filter banks, n_fft=2048, hop_length=512, fmin=0, fmax=8000, 3.0s window (94 time frames).
   - Storage format: `np.float32` memmap.
2. **Architecture and Parameter Count**:
   - Backbone: Torchvision `resnet50(weights=ResNet50_Weights.DEFAULT)` with 1-channel conv1 initialized by averaging ImageNet 3-channel weights.
   - Classification Head: `Dropout(0.3) -> Linear(2048 + 6, 1)`.
   - Total Parameters: Exactly **23,503,815** across baseline and all feature conditions. Feature slots for inactive conditions are fixed to zero.
3. **Training Controls**:
   - Standardization: Population mean and standard deviation computed strictly on internal Train utterances (`~dev`), applied to Dev.
   - Initial Weights: Seed-deterministic (`seed=42`, `seed=43`), bit-exact across conditions within the same seed.
   - Shuffling Isolation: `DataLoader` shuffle uses an isolated `torch.Generator().manual_seed(seed)` to guarantee reproducibility independent of preceding timing smoke executions.
4. **Cache Deduplication**:
   - Baseline and ZCR share the exact same 128-Mel array (`baseline.npy`, 1.47 GB). ZCR auxiliary statistics are loaded from `features.npz` (1.89 MB), eliminating redundant spectrogram storage.

---

## 3. Code Sources and Locations

| Script | Location | Purpose | Key CLI Flags |
| :--- | :--- | :--- | :--- |
| `quick_hypotheses.py` | `mission2_speaker/experiments/quick_hypotheses.py` | Universal cache extractor, audit, probe, and self-test runner | `prepare`, `probe`, `audit`, `self-test` |
| `run_all_audio_hypotheses.py` | `mission2_speaker/experiments/run_all_audio_hypotheses.py` | 500-call integrated 5-hypothesis suite runner | `--cache-dir`, `--output-dir`, `--epochs 5` |
| `run_zcr_confirmation.py` | `mission2_speaker/experiments/run_zcr_confirmation.py` | 1,000-call ZCR confirmation runner and gate synthesizer | `--cache-dir`, `--output-dir` |

---

## 4. Exact Execution Commands

### 4.1 Self-Test
```bash
python mission2_speaker/experiments/quick_hypotheses.py self-test
```

### 4.2 Deduplicated 1,000-Call Cache Preparation
```bash
python mission2_speaker/experiments/quick_hypotheses.py prepare \
    --training-root "data/train" \
    --cache-dir "D:/DCC_m2_zcr_confirmation/20261011/cache_1000_resnet128" \
    --exclude-calls "path/to/previous_500_calls.json" \
    --max-calls 1000 \
    --max-per-call 0 \
    --dev-fraction 0.2 \
    --seed 42 \
    --sample-seed 4243 \
    --mel-profile resnet128 \
    --dedup-mel
```

### 4.3 Probe Execution (Baseline vs ZCR)
```bash
python mission2_speaker/experiments/quick_hypotheses.py probe \
    --cache-dir "D:/DCC_m2_zcr_confirmation/20261011/cache_1000_resnet128" \
    --output-dir "D:/DCC_m2_zcr_confirmation/20261011/probe_s42" \
    --model resnet50 \
    --conditions baseline zcr \
    --epochs 5 \
    --batch-size 16 \
    --lr 0.0001 \
    --seed 42 \
    --device cuda \
    --head-dropout 0.3 \
    --bootstrap 2000
```

---

## 5. Offline Inspection (No Reruns Required)

All quantitative tables and figures can be reconstructed directly from the saved CSV and NPZ arrays without re-executing training:
- 500-call results: `ALL_HYPOTHESES_COMPARISON.csv`
- 1,000-call results: `ZCR_PILOT_COMPARISON_ALL.csv` and `ZCR_PILOT_DIFFERENCES_ALL.csv`
- Final decision records: `ADOPTION_DECISION.md` and `decision_summary.json`
