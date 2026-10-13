# Satellite Image Change Detection Using Temporal Transformers

M.Tech dissertation implementation for **bi-temporal remote-sensing change detection**.

## Frozen architecture

```text
T1 image ─┐
          ├─ Shared CNN Encoder ──┐
T2 image ─┘                        │
                     Temporal Self-Attention
                                ↓
                         Cross-Attention
                                ↓
                      Multi-Scale Feature Fusion
                                ↓
                            Decoder
                                ↓
                       Binary Change Map
```

The repository includes:
- LEVIR-CD dataset loader with support for `T1/T2/label` or `A/B/label` folder naming.
- Siamese CNN baseline.
- Temporal Transformer baseline.
- Proposed Temporal Transformer + cross-attention + multi-scale decoder.
- BCE + Dice loss.
- Precision, Recall, F1, IoU, OA, Kappa, FPR/FNR, FP/FN.
- Training, validation, test evaluation and prediction scripts.
- Ablation configurations for removing temporal self-attention, cross-attention and multi-scale fusion.
- Dataset visualization and a CPU smoke test.

## 1. Environment

Windows PowerShell or Command Prompt:

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 2. Dataset

Download LEVIR-CD from its official/reliable repository separately. The dataset is intentionally **not included** in this package because it is large and externally distributed.

Place the extracted files under:

```text
datasets/LEVIR-CD/
├── train/
│   ├── T1/
│   ├── T2/
│   └── label/
├── val/
│   ├── T1/
│   ├── T2/
│   └── label/
└── test/
    ├── T1/
    ├── T2/
    └── label/
```

The loader also accepts `A/B/label` folder naming.

Filenames must match by stem. Example:

```text
train/T1/00001.png
train/T2/00001.png
train/label/00001.png
```

## 3. Validate dataset

```bash
python tools/validate_dataset.py --root ./datasets/LEVIR-CD
```

Expected:

```text
train: ... pairs
val: ... pairs
test: ... pairs
Dataset validation: PASSED
```

## 4. Visualize a sample

```bash
python visualize_dataset.py --config configs/config.yaml --split train --index 0 --output results/sample.png
```

## 5. Run smoke test before real training

This uses generated synthetic data and does not require LEVIR-CD.

```bash
python tools/smoke_test.py
```

Expected final line:

```text
SMOKE TEST PASSED
```

## 6. Train baselines and proposed model

Siamese CNN:

```bash
python train.py --config configs/cnn_baseline.yaml
```

Transformer baseline:

```bash
python train.py --config configs/transformer_baseline.yaml
```

Proposed model:

```bash
python train.py --config configs/config.yaml
```

The best checkpoint is saved as:

```text
experiments/<experiment_name>/best.pt
```

## 7. Evaluate on test set

```bash
python evaluate.py --config configs/config.yaml --checkpoint experiments/proposed_model/best.pt --split test
```

## 8. Predict on a single T1/T2 pair

```bash
python predict.py \
  --config configs/config.yaml \
  --checkpoint experiments/proposed_model/best.pt \
  --t1 path/to/t1.png \
  --t2 path/to/t2.png \
  --output predictions/sample_001.png
```

## 9. Ablation studies

Remove temporal self-attention:

```bash
python train.py --config configs/ablation_no_temporal_attention.yaml
```

Remove cross-attention:

```bash
python train.py --config configs/ablation_no_cross_attention.yaml
```

Remove multi-scale fusion:

```bash
python train.py --config configs/ablation_no_multiscale.yaml
```

## 10. Recommended experiment order

1. Dataset validation
2. Visualization
3. Smoke test
4. Siamese CNN baseline
5. Transformer baseline
6. Proposed model
7. No temporal attention ablation
8. No cross-attention ablation
9. No multi-scale ablation
10. Test-set comparison
11. Qualitative prediction visualization
12. Cross-dataset/generalization experiment

## 11. Research reporting

Do not put invented accuracy values in the dissertation. Record the actual metrics from `evaluate.py` and compare:

| Model | Precision | Recall | F1 | IoU | OA | Kappa | FP | FN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Siamese CNN | actual | actual | actual | actual | actual | actual | actual | actual |
| Transformer | actual | actual | actual | actual | actual | actual | actual | actual |
| Proposed | actual | actual | actual | actual | actual | actual | actual | actual |

## 12. Important implementation note

The project deliberately avoids claiming that the proposed model is automatically superior. The code provides the baselines, proposed architecture and ablations needed to experimentally test the research hypothesis.
