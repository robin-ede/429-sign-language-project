# DAEN 429 - Sign Language Recognition

Transfer learning project for ASL alphabet classification and dynamic word recognition.

## Files

**Phase 1: Static ASL Alphabet**
- `phase1_asl_training.ipynb` - ResNet-18 transfer learning (T-A, T-B, T-C, S-A)
- `main.tex` & `main.pdf` - Phase 1 report

**Phase 2: Dynamic WLASL100**
- `preprocess_wlasl_videos.py` - Video frame extraction
- `phase2_temporal_modeling.ipynb` - LSTM/Transformer temporal models

## Datasets

- **ASL Alphabet**: [kaggle.com/datasets/grassknoted/asl-alphabet](https://www.kaggle.com/datasets/grassknoted/asl-alphabet)
- **WLASL100**: [kaggle.com/datasets/thtrnphc/wlasl100-new](https://www.kaggle.com/datasets/thtrnphc/wlasl100-new)
