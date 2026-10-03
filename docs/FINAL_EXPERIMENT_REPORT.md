# [ ] 119            

- ****: DCC Problem Mission 2 -    (  vs )  
- ** **: 2026 9
- ** **: Local Workstation (NVIDIA GeForce RTX 3060 6GB GDDR6, 32GB RAM)  Google Colab (Tesla T4 15GB)
- ** **: AI-Hub 119     (    111,947, 15.6GB)

---

## 1.    

### 1.1  
119          ,  ,    .  (Dispatcher)          , (Caller)    ,   ,          .                      .

### 1.2     
1. ** **:    (11.2 )   F1-Score   92%  .
2. ** **:       RTF(Real-Time Factor) 0.01    20ms  .
3. **  **:   GPU(VRAM 6GB )   (OOM)          .

---

## 2.      

### 2.1   
     8kHz  (Mono)     (   16kHz  )  /  (JSON) .
-  (Train Set):  873,137  
-  (Validation Set):  111,947   ( 0(): 53,845 /  1(): 58,102)

### 2.2 5     
      355       (DSP)  ,    .

|   |  (Class 0) |  (Class 1) |      |
| :--- | :---: | :---: | :--- |
| **  ** | 2.11 | 2.38 |   .      12.8%   . |
| **   (RMS Std)** | 0.0376 | 0.0488 | **   29.8% .**        . |
| ** ** | 1,070.9 Hz | 891.8 Hz |      ,       . |
| ** (ZCR)** | 0.1048 | 0.0900 |         /  . |
| **  ** | 300 ~ 3,400 Hz | 300 ~ 3,400 Hz |  (PSTN/VoLTE)      200~4,000Hz  . |

### 2.3      
1. ** 1 ( )**:   (200~4,000Hz)       .
2. ** 2 (  )**:  0dB     30%     RMS  .
3. ** 3 (   )**:     Delta  Delta-Delta   3  .

---

## 3.        (Ablation Study)

### 3.1      
- ** (Window) **:         `[Batch, Channel, Mels, Time_Steps]`           .
- ** **:  1.5              .  5.0     1~2    0(Zero-Padding)   VRAM ,  GPU       .       .

### 3.2     
 (Train 30,313 / Val 9,138)  4    .
- Pre-1: 1.5  + Zero  (n_fft 2048, n_mels 128)
- Pre-2: 3.0  + Repeat  (n_fft 2048, n_mels 128)
- Pre-3: 3.0  + Zero  (n_fft 2048, n_mels 128) - 
- Pre-4: 3.0  + Zero  (n_fft 1024, n_mels 128)

### 3.3    

#### [  : model_train.ipynb Cell 12]
```text
[TRAIN]    : 30313  !
[VAL]    : 9138  !

=================== [ : Pre-1] ===================
: Window=1.5s, Padding=zero, n_fft=2048, n_mels=128
  [Epoch 1/3] Train Loss: 0.4456 | Val Acc: 83.79% | Macro F1: 0.8367
  [Epoch 2/3] Train Loss: 0.3250 | Val Acc: 84.92% | Macro F1: 0.8480
  [Epoch 3/3] Train Loss: 0.2784 | Val Acc: 84.17% | Macro F1: 0.8416

=================== [ : Pre-2] ===================
: Window=3.0s, Padding=repeat, n_fft=2048, n_mels=128
  [Epoch 1/3] Train Loss: 0.4026 | Val Acc: 84.33% | Macro F1: 0.8429
  [Epoch 2/3] Train Loss: 0.2844 | Val Acc: 83.97% | Macro F1: 0.8397
  [Epoch 3/3] Train Loss: 0.2289 | Val Acc: 85.14% | Macro F1: 0.8503

=================== [ : Pre-3] ===================
: Window=3.0s, Padding=zero, n_fft=2048, n_mels=128
  [Epoch 1/3] Train Loss: 0.4145 | Val Acc: 83.73% | Macro F1: 0.8372
  [Epoch 2/3] Train Loss: 0.2925 | Val Acc: 85.93% | Macro F1: 0.8584
  [Epoch 3/3] Train Loss: 0.2282 | Val Acc: 83.63% | Macro F1: 0.8363

=================== [ : Pre-4] ===================
: Window=3.0s, Padding=zero, n_fft=1024, n_mels=128
  [Epoch 1/3] Train Loss: 0.4107 | Val Acc: 84.14% | Macro F1: 0.8410
  [Epoch 2/3] Train Loss: 0.2876 | Val Acc: 85.48% | Macro F1: 0.8545
  [Epoch 3/3] Train Loss: 0.2263 | Val Acc: 85.19% | Macro F1: 0.8513

[Ablation Study     ]
    ID        n_fft  n_mels Val Accuracy (%) Macro F1                     
0  Pre-1   1.5    zero   2048     128           84.92%   0.8480          1.5s    
1  Pre-2   3.0  repeat   2048     128           85.14%   0.8503        3.0s     
2  Pre-3   3.0    zero   2048     128           85.93%   0.8584        3.0s  +   ()
3  Pre-4   3.0    zero   1024     128           85.48%   0.8545  3.0s  +  FFT
```
![ 1:        ](figures/fig1_ablation_window.png)
*< 1>           Macro F1-Score  (Ablation Study)*

![ 1:        ](figures/fig1_ablation_window.png)
*< 1>           Macro F1-Score  (Ablation Study)*


** **: Pre-3(3.0  + Zero )    85.93%, Macro F1 0.8584    . Pre-1(1.5)  1.01%p  , 6GB VRAM    32       .

---

## 4.         

### 4.1        

#### (1)  (Transformer)      
     AI      /         .
- **  **: Microsoft `WavLM-Base+` (95.0M), Meta `HuBERT-Base` (95.0M), Meta `Wav2Vec 2.0` (94.4M)
- **/ **: `SSAST` (Spectrogram Swin / Audio Spectrogram Transformer, 87.0M)

   (Train 87  / Val 11.2 )            :
1. ** GPU   (CUDA Out Of Memory, OOM)**:
   - Self-Attention      $O(T^2)$     , 3 (48,000 )     VRAM  14GB   RTX 3060 (6GB VRAM)   **CUDA OOM** .
   - OOM     4~8    ,      87   1  12       10    .
2. **   (Acoustic Domain Mismatch)**:
   -   GPU     , `Wav2Vec 2.0`   **89.72%**    36 1   (ReDimNet 91.89%, ECAPA-TDNN 92.10%)   .
   - 16kHz       8kHz  PSTN   119            .
3. **  **:
   -      45~60ms ,  119    ( RTF < 0.01)     .

#### (2)  (Heterogeneous)   
     ,        **   3  **  .
1. **ReDimNet2-B2 ( , 2.57M)**: 2D   Conv Formant/Pitch , 1D Dilated Conv  Multi-Head Attention      2.57M     .
2. **ECAPA-TDNN (1D CNN , 5.80M)**:       1D Res2Net , Squeeze-and-Excitation  ,  (Attentive Statistics Pooling)        .
3. **AudioResNet-50 (2D CNN , 23.50M)**: 2   -    1  CNN ,        (Anchor)  .

### 4.2  10 Epoch   
  (Train 873,137 / Val 111,947,   32, AMP )  10 Epoch  .

#### [ReDimNet2-B2  : Local_Light_Train.ipynb Cell 5]
```text
Epoch 01  | Train Acc: 87.83% | Val Loss: 0.2123 | Val Acc: 90.01%
Epoch 02  | Train Acc: 89.95% | Val Loss: 0.2025 | Val Acc: 90.59%
Epoch 03  | Train Acc: 90.72% | Val Loss: 0.1908 | Val Acc: 90.99%
Epoch 04  | Train Acc: 91.23% | Val Loss: 0.1900 | Val Acc: 90.79%
Epoch 05  | Train Acc: 91.64% | Val Loss: 0.1827 | Val Acc: 91.31%
Epoch 06  | Train Acc: 92.04% | Val Loss: 0.1765 | Val Acc: 91.63%
Epoch 07  | Train Acc: 92.40% | Val Loss: 0.1737 | Val Acc: 91.74%
Epoch 08  | Train Acc: 92.73% | Val Loss: 0.1755 | Val Acc: 91.75%
Epoch 09  | Train Acc: 93.07% | Val Loss: 0.1737 | Val Acc: 91.89% ( )
Epoch 10  | Train Acc: 93.26% | Val Loss: 0.1783 | Val Acc: 91.80%
```

#### [ECAPA-TDNN  : Local_Light_Train.ipynb Cell 7]
```text
[ECAPA-TDNN] Epoch 01 | Train Acc: 87.97% | Val Loss: 0.2160 | Val Acc: 90.01%
[ECAPA-TDNN] Epoch 02 | Train Acc: 90.12% | Val Loss: 0.2192 | Val Acc: 89.59%
[ECAPA-TDNN] Epoch 03 | Train Acc: 90.95% | Val Loss: 0.1886 | Val Acc: 90.99%
[ECAPA-TDNN] Epoch 04 | Train Acc: 91.46% | Val Loss: 0.1859 | Val Acc: 91.17%
[ECAPA-TDNN] Epoch 05 | Train Acc: 91.90% | Val Loss: 0.1767 | Val Acc: 91.68%
[ECAPA-TDNN] Epoch 06 | Train Acc: 92.30% | Val Loss: 0.1765 | Val Acc: 91.65%
[ECAPA-TDNN] Epoch 07 | Train Acc: 92.71% | Val Loss: 0.1699 | Val Acc: 92.00%
[ECAPA-TDNN] Epoch 08 | Train Acc: 93.07% | Val Loss: 0.1706 | Val Acc: 92.10% ( )
[ECAPA-TDNN] Epoch 09 | Train Acc: 93.40% | Val Loss: 0.1724 | Val Acc: 92.09%
[ECAPA-TDNN] Epoch 10 | Train Acc: 93.69% | Val Loss: 0.1764 | Val Acc: 92.08%
```

#### [AudioResNet-50  : Local_Light_Train.ipynb Cell 9]
```text
[ResNet-50] Epoch 01 | Train Acc: 88.56% | Val Loss: 0.2159 | Val Acc: 89.90%
[ResNet-50] Epoch 02 | Train Acc: 90.75% | Val Loss: 0.2004 | Val Acc: 90.61%
[ResNet-50] Epoch 03 | Train Acc: 91.56% | Val Loss: 0.2082 | Val Acc: 90.32%
[ResNet-50] Epoch 04 | Train Acc: 92.23% | Val Loss: 0.1924 | Val Acc: 91.03%
[ResNet-50] Epoch 05 | Train Acc: 92.96% | Val Loss: 0.1972 | Val Acc: 91.14%
[ResNet-50] Epoch 06 | Train Acc: 93.72% | Val Loss: 0.2096 | Val Acc: 91.11%
[ResNet-50] Epoch 07 | Train Acc: 94.37% | Val Loss: 0.2255 | Val Acc: 91.12%
[ResNet-50] Epoch 08 | Train Acc: 94.80% | Val Loss: 0.2737 | Val Acc: 91.14%
[ResNet-50] Epoch 09 | Train Acc: 95.02% | Val Loss: 0.3271 | Val Acc: 91.19%
[ResNet-50] Epoch 10 | Train Acc: 95.12% | Val Loss: 0.3656 | Val Acc: 91.20% ( )
```

### 4.3       
  (RTX 3060 6GB)      .

|   | VRAM  (Batch=32) | 1 Epoch   | 1 Epoch   | 10 Epoch    |    |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ReDimNet2-B2** | 1.8 GB (30%) | 1 44 | 11 15 |  19.3  | 91.89% |
| **ECAPA-TDNN** | 2.4 GB (40%) | 1 48 | 11 50 |  20.0  | 92.10% |
| **AudioResNet-50** | 4.6 GB (77%) | 1 58 | 13 10 |  21.8  | 91.20% |
| **  ** | 1.8 ~ 4.6 GB | - | - |  24.0  | - |
| **Ablation Study (4)** | 4.6 GB | - | - |  3.5  | - |
| ** GPU ** | ** 4.6 GB** | - | - | ** 88.6  ( 3.7)** | - |

---

## 5.         

### 5.1    
2  5      4 Butterworth   (200~4,000Hz)  (RMS)    `DCCSpecializedAudioDataset`  .        (Fine-tuning,  5e-05) .

### 5.2   
 ,             .

#### [ReDimNet2-B2   : Local_Light_Train.ipynb Cell 17]
```text
 [REDIMNET]   (200~4000Hz) 10   (: 91.89%)
[ FT 1ep] Val Acc: 91.25% (  -0.64%p) | Val Loss: 0.1795
[ FT 2ep] Val Acc: 91.42% (  -0.47%p) | Val Loss: 0.1793
[ FT 3ep] Val Acc: 91.48% (  -0.41%p) | Val Loss: 0.1845
[ FT 4ep] Val Acc: 91.57% (  -0.32%p) | Val Loss: 0.1848
[ FT 5ep] Val Acc: 91.58% (  -0.31%p) | Val Loss: 0.1833
[ FT 6ep] Val Acc: 91.24% (  -0.65%p) | Val Loss: 0.1894
[ FT 7ep] Val Acc: 91.29% (  -0.60%p) | Val Loss: 0.2057
[ FT 8ep] Val Acc: 91.33% (  -0.56%p) | Val Loss: 0.2124
```

#### [AudioResNet-50   : Local_Light_Train.ipynb Cell 19]
```text
 [RESNET50]   (200~4000Hz) 3   (: 91.20%)
[ FT 1ep] Val Acc: 90.54% (  -0.66%p) | Val Loss: 0.2034
[ FT 2ep] Val Acc: 90.67% (  -0.53%p) | Val Loss: 0.2136
[ FT 3ep] Val Acc: 90.70% (  -0.50%p) | Val Loss: 0.2634
```
![ 3:     /  ](figures/fig3_domain_failure.png)
*< 3>  (0~4,000Hz Nyquist)  200~4,000Hz         *


### 5.3 5       
   5           .

|  |   |     |         |
| :---: | :--- | :--- | :--- |
| **①** | **  (Duration)** | 3.0   |     85.93%  100%   . |
| **②** | **   (RMS Std)** | +29.8%    | **[ ]**   ,           .     . |
| **③**<br>**④** | ** **<br>** (ZCR)** |      | **[  ]**  ZCR     . ⑤     (>4000Hz)      . |
| **⑤** | **   (Bandwidth)** | 200~4000Hz  | **[ :       ]**<br>• **200Hz  **:    (F0, Pitch: 85~180Hz)    (Proximity Effect)         .<br>• **  **: 8kHz    (4,000Hz)   4 IIR     (Phase Distortion) .<br>** 0~4,000Hz         ,        .** |

### 5.4       
"       Delta      "    .
1. **   (Coupling)**:     ,            .
2. **(End-to-End)    **:       **0~8,000Hz           (92.10%) ** .              .

---

## 6.        (Threshold) 

### 6.1 Soft Voting     
           3  (ECAPA-TDNN, ReDimNet2-B2, AudioResNet-50)    Soft Voting  .          (ReDimNet 0.45 : ECAPA-TDNN 0.40 : AudioResNet-50 0.15) .

### 6.2 111,947      (Threshold)  
   111,947  Soft Voting    ,   F1-Score    (Decision Threshold) 0.35 0.65 0.01   .
-   0.50 **92.48% (Macro F1: 0.9244)**     .
- F1-Score     **Threshold = 0.51** ,  **   92.58%, Macro F1 0.9258** .

#### [   ]
```text
=================================================================
[    ( )]
 1. ReDimNet2-B2:   91.89% | Macro F1: 0.9185
 2. ECAPA-TDNN:     92.10% | Macro F1: 0.9206
 3. AudioResNet-50:  91.20% | Macro F1: 0.9115
=================================================================
[3 Soft Voting     ]
 -   : ReDimNet 0.45, ECAPA-TDNN 0.40, AudioResNet-50 0.15
 -    (Optimal Threshold): 0.51
 -     (Accuracy):    92.58%
 -   Macro F1-Score:            0.9258
=================================================================

[    (Classification Report: Threshold 0.51 )]
                     precision    recall  f1-score   support

 (Dispatcher, 0)     0.9482    0.8953    0.9210     53845
     (Caller, 1)     0.9074    0.9540    0.9301     58102

           accuracy                         0.9258    111947
          macro avg     0.9278    0.9247    0.9258    111947
       weighted avg     0.9270    0.9258    0.9257    111947
```
![ 4:      11.2  ](figures/fig4_ensemble_performance.png)
*< 4>     3 Soft Voting      111,947   (Confusion Matrix)*

### 6.3      (Robustness) 
   ECAPA-TDNN(92.10%) ,    **    92.58% **.
1. ** (TDNN)  (ResNet) **:          TDNN 2     ResNet     .
2. ** **:  0.50~0.52    92.48%~92.58%    ,           .

---

## 7.        

### 7.1        (RTX 3060 )

|     |   |    |   (Latency) | RTF (3 ) |    (Throughput) |    ( ) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **ReDimNet2-B2 ()** | 2.57 M | 9.8 MB | 1.73 ms | 0.0006 | 1,999.4 / | 91.89% |
| **ECAPA-TDNN ()** | 5.80 M | 22.2 MB | 5.12 ms | 0.0017 | 1,304.2 / | 92.10% |
| **AudioResNet-50 ()** | 23.50 M | 90.0 MB | 4.65 ms | 0.0016 | 632.9 / | 91.20% |
| **3 Soft Voting ** | **31.87 M** | **122.0 MB** | **11.50 ms** | **0.0038** | ** 350.0 /** | **92.58% (F1: 0.9258)** |

![ 2:        ](figures/fig2_model_efficiency.png)
*< 2> RTX 3060     vs  (Latency) vs  (Throughput)  *


### 7.2       
1. ** **: 3    **0.0115(11.5ms)**   RTF **0.0038** .    (200ms)  17         .
2. **   **:  RTX 3060 6GB      ** 350    **  , 119        1  .
3. **  **:   (Wav2Vec )   VRAM(A100 )  ,    ** VRAM 4.6GB  92.48%    **       .

### 7.3   (Checkpoints)     

> **:    **:       (Overfitting)     (Validation Loss)    (ModelCheckpoint)   . AudioResNet-50  10  , ReDimNet2-B2  ECAPA-TDNN   (10ep)   0.06%p          .


  3    (`.pt`) (Reproducibility)            ,   PyTorch `load_state_dict`   .

|   |   (Acc) |   |   |    |    |
| :--- | :---: | :--- | :---: | :---: | :---: |
| **ReDimNet2-B2** | 91.89% | `mission2_speaker/checkpoints/best_redimnet.pt` | 9.84 MB | 91 |    (100% ) |
| **ECAPA-TDNN** | 92.10% | `mission2_speaker/checkpoints/best_ecapa_tdnn.pt` | 22.23 MB | 233 |    (100% ) |
| **AudioResNet-50**| 91.20% | `mission2_speaker/checkpoints/best_resnet50.pt` | 89.97 MB | 320 |    (100% ) |
| *( )* | - | `mission2_speaker/test_results/checkpoints/` | - | - |     |

> **:     **:   (31.87M),  (122.0MB), (11.50ms) 3             .

> ****:   (`.pt`)         `.gitignore`    ,   (`inference.py`  `Local_Light_Train.ipynb`)         .




---

## 8.    

1. **  **:    1.5 vs 3.0      VRAM    3.0    .
2. **    **:               (200Hz     4000Hz   )    .               .
3. ** **: 88.6     3    **111,947    92.48% (Macro F1 0.9244), RTF 0.0038 ,  GPU 350/ **      ·    .
