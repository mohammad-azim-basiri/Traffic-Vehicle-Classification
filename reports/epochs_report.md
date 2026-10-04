# Training Experiments Report

> Cleaned and reformatted from the original experiment log. Training values are preserved from the source; only the Markdown structure and headings have been normalized.

## Table of Contents

1. [Small CNN — Dropout = 0.0](#small-cnn-dropout-0-0)
2. [Small CNN — Dropout = 0.0 (vanet + neisan merged)](#small-cnn-dropout-0-0-vanet-neisan-merged)
3. [Small CNN — Augmentation: Horizontal Flip (p = 0.5)](#small-cnn-augmentation-horizontal-flip-p-0-5)
4. [Small CNN — Augmentation: Random Horizontal Flip + Gaussian Blur + Random Erasing](#small-cnn-augmentation-random-horizontal-flip-gaussian-blur-random-erasing)
5. [Small CNN — Dropout = 0.3](#small-cnn-dropout-0-3)
6. [Small CNN — Dropout = 0.5](#small-cnn-dropout-0-5)
7. [Small CNN — Average Pooling](#small-cnn-average-pooling)
8. [Small CNN — Weight Decay = 0](#small-cnn-weight-decay-0)
9. [Small CNN — Weight Decay = 0.0001](#small-cnn-weight-decay-0-0001)
10. [Small CNN — StepLR](#small-cnn-steplr)
11. [Small CNN — ReduceLROnPlateau](#small-cnn-reducelronplateau)
12. [Small CNN — Standard Imbalanced Loader](#small-cnn-standard-imbalanced-loader)
13. [Small CNN — Balanced Imbalanced Loader](#small-cnn-balanced-imbalanced-loader)
14. [Small CNN — BCE](#small-cnn-bce)
15. [ResNet18 — Full Training](#resnet18-full-training)
16. [ResNet18 — Fine-Tuning](#resnet18-fine-tuning)
17. [MobileNetV3-Small — Full Fine-Tuning (ResNet18 Weights Transform)](#mobilenetv3-small-full-fine-tuning-resnet18-weights-transform)
18. [MobileNetV3-Small — Full Fine-Tuning (Baseline Transform)](#mobilenetv3-small-full-fine-tuning-baseline-transform)
19. [MobileNetV3-Large — Full Fine-Tuning (ResNet-Style Transform)](#mobilenetv3-large-full-fine-tuning-resnet-style-transform)
20. [MobileNetV3-Large — Head Fine-Tuning (ResNet-Style Transform)](#mobilenetv3-large-head-fine-tuning-resnet-style-transform)
21. [MobileNetV3-Large — Last Block (3) Fine-Tuning (ResNet-Style Transform)](#mobilenetv3-large-last-block-3-fine-tuning-resnet-style-transform)
22. [MobileNetV3-Large — Full Fine-Tuning (Mean/Std = 0.5)](#mobilenetv3-large-full-fine-tuning-mean-std-0-5)
23. [MobileNetV3-Large — Head Fine-Tuning (Mean/Std = 0.5)](#mobilenetv3-large-head-fine-tuning-mean-std-0-5)
24. [MobileNetV3-Large — Last Block (3) Fine-Tuning (Mean/Std = 0.5)](#mobilenetv3-large-last-block-3-fine-tuning-mean-std-0-5)
25. [Depthwise Model — Dropout = 0.2](#depthwise-model-dropout-0-2)
26. [Depthwise Model — Dropout = 0.5](#depthwise-model-dropout-0-5)

## Experiment Summary

| # | Experiment                                                                        | Epochs | Best Val. Acc. | Best Epoch | Final Val. Acc. | Training Time | Trainable Params |
|---:|-----------------------------------------------------------------------------------|---:|---:|---:|---:|---:|-----------------:|
| 1 | Small CNN — Dropout = 0.0                                                         | 50 | 93.3% | 50 | 93.3% | 29.32 min |          585,640 |
| 2 | Small CNN — Dropout = 0.0 (vanet + neisan merged)                                 | 50 | 93.7% | 42 | 90.7% | 50.20 min |          585,640 |
| 3 | Small CNN — Augmentation: Horizontal Flip (p = 0.5)                               | 50 | 92.0% | 38 | 74.7% | 25.73 min |          585,640 |
| 4 | Small CNN — Augmentation: Random Horizontal Flip + Gaussian Blur + Random Erasing | 50 | 92.4% | 49 | 87.6% | 71.89 min |          585,640 |
| 5 | Small CNN — Dropout = 0.3                                                         | 50 | 92.5% | 44 | 83.7% | 25.75 min |          585,640 |
| 6 | Small CNN — Dropout = 0.5                                                         | 50 | 92.6% | 44 | 89.6% | 25.17 min |          585,640 |
| 7 | Small CNN — Average Pooling                                                       | 50 | 91.6% | 47 | 91.0% | 45.38 min |          585,640 |
| 8 | Small CNN — Weight Decay = 0                                                      | 50 | 93.3% | 50 | 93.3% | 62.74 min |          585,640 |
| 9 | Small CNN — Weight Decay = 0.0001                                                 | 50 | 93.4% | 50 | 93.4% | 63.09 min |          585,640 |
| 10 | Small CNN — StepLR                                                                | 50 | 90.8% | 27 | 90.4% | 62.98 min |          585,640 |
| 11 | Small CNN — ReduceLROnPlateau                                                     | 50 | 92.5% | 23 | 91.6% | 43.00 min |          585,640 |
| 12 | Small CNN — Standard Imbalanced Loader                                            | 50 | 89.6% | 40 | 84.9% | 32.29 min |          585,640 |
| 13 | Small CNN — Balanced Imbalanced Loader                                            | 50 | 88.5% | 46 | 88.5% | 32.30 min |          585,640 |
| 14 | Small CNN — BCE                                                                   | 50 | 93.4% | 36 | 84.5% | 42.79 min |          585,640 |
| 15 | ResNet18 —Head Training                                                           | 50 | 91.0% | 37 | 89.6% | 12.61 min |             4104 |
| 16 | ResNet18 — Fine-Tuning                                                            | 50 | 96.1% | 45 | 95.2% | 13.74 min |                8397832 |
| 17 | MobileNetV3-Small — Full Fine-Tuning (ResNet18 Weights Transform)                 | 50 | 96.9% | 36 | 95.1% | 12.54 min |        1,526,056 |
| 18 | MobileNetV3-Small — Full Fine-Tuning (Baseline Transform)                         | 50 | 97.6% | 26 | 95.9% | 10.09 min |        1,526,056 |
| 19 | MobileNetV3-Large — Full Fine-Tuning (ResNet-Style Transform)                     | 50 | 96.5% | 26 | 95.5% | 19.20 min |        4,212,280 |
| 20 | MobileNetV3-Large — Head Fine-Tuning (ResNet-Style Transform)                     | 50 | 91.8% | 15 | 90.2% | 10.86 min |        1,240,328 |
| 21 | MobileNetV3-Large — Last Block (3) Fine-Tuning (ResNet-Style Transform)           | 50 | 96.0% | 36 | 91.8% | 11.49 min |        2,990,568 |
| 22 | MobileNetV3-Large — Full Fine-Tuning (Mean/Std = 0.5)                             | 50 | 97.5% | 29 | 90.2% | 16.34 min |        4,212,280 |
| 23 | MobileNetV3-Large — Head Fine-Tuning (Mean/Std = 0.5)                             | 50 | 91.0% | 50 | 91.0% | 8.88 min |        1,240,328 |
| 24 | MobileNetV3-Large — Last Block (3) Fine-Tuning (Mean/Std = 0.5)                   | 50 | 97.1% | 29 | 96.4% | 9.90 min |        2,990,568 |
| 25 | Depthwise Model — Dropout = 0.2                                                   | 50 | 88.6% | 45 | 80.1% | 25.52 min |                73,033 |
| 26 | Depthwise Model — Dropout = 0.5                                                   | 50 | 88.8% | 45 | 79.0% | 25.36 min |                73,033 |

---

## Detailed Experiment Logs

## Small CNN — Dropout = 0.0

- **Best validation accuracy:** 93.3% (epoch 50)
- **Final validation accuracy:** 93.3%
- **Training time:** 1759.49 seconds (29.32 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.596 | val loss: 1.612 | train acc: 43.1% | val acc: 43.3% | current lr: [0.001]
Epoch 02/50 | train loss: 1.266 | val loss: 1.609 | train acc: 56.5% | val acc: 43.7% | current lr: [0.001]
Epoch 03/50 | train loss: 1.082 | val loss: 1.095 | train acc: 62.0% | val acc: 60.3% | current lr: [0.001]
Epoch 04/50 | train loss: 0.945 | val loss: 1.361 | train acc: 66.1% | val acc: 51.6% | current lr: [0.001]
Epoch 05/50 | train loss: 0.807 | val loss: 2.192 | train acc: 72.0% | val acc: 40.6% | current lr: [0.001]
Epoch 06/50 | train loss: 0.696 | val loss: 1.133 | train acc: 76.3% | val acc: 67.2% | current lr: [0.001]
Epoch 07/50 | train loss: 0.615 | val loss: 2.080 | train acc: 79.6% | val acc: 41.0% | current lr: [0.001]
Epoch 08/50 | train loss: 0.571 | val loss: 0.857 | train acc: 81.5% | val acc: 72.9% | current lr: [0.001]
Epoch 09/50 | train loss: 0.484 | val loss: 0.524 | train acc: 83.9% | val acc: 82.6% | current lr: [0.001]
Epoch 10/50 | train loss: 0.405 | val loss: 1.204 | train acc: 86.9% | val acc: 65.0% | current lr: [0.001]
Epoch 11/50 | train loss: 0.408 | val loss: 0.510 | train acc: 86.5% | val acc: 83.4% | current lr: [0.001]
Epoch 12/50 | train loss: 0.373 | val loss: 1.127 | train acc: 87.3% | val acc: 65.4% | current lr: [0.001]
Epoch 13/50 | train loss: 0.337 | val loss: 0.695 | train acc: 89.2% | val acc: 76.2% | current lr: [0.001]
Epoch 14/50 | train loss: 0.305 | val loss: 0.583 | train acc: 90.5% | val acc: 80.7% | current lr: [0.001]
Epoch 15/50 | train loss: 0.266 | val loss: 0.528 | train acc: 91.6% | val acc: 82.6% | current lr: [0.001]
Epoch 16/50 | train loss: 0.275 | val loss: 1.009 | train acc: 91.7% | val acc: 69.8% | current lr: [0.001]
Epoch 17/50 | train loss: 0.210 | val loss: 1.284 | train acc: 93.3% | val acc: 62.3% | current lr: [0.001]
Epoch 18/50 | train loss: 0.226 | val loss: 0.699 | train acc: 92.6% | val acc: 77.4% | current lr: [0.001]
Epoch 19/50 | train loss: 0.202 | val loss: 0.729 | train acc: 93.6% | val acc: 75.1% | current lr: [0.001]
Epoch 20/50 | train loss: 0.186 | val loss: 1.437 | train acc: 94.1% | val acc: 66.8% | current lr: [0.001]
Epoch 21/50 | train loss: 0.164 | val loss: 1.661 | train acc: 95.1% | val acc: 55.3% | current lr: [0.001]
Epoch 22/50 | train loss: 0.134 | val loss: 0.414 | train acc: 95.8% | val acc: 88.9% | current lr: [0.001]
Epoch 23/50 | train loss: 0.117 | val loss: 0.524 | train acc: 96.7% | val acc: 83.7% | current lr: [0.001]
Epoch 24/50 | train loss: 0.116 | val loss: 0.601 | train acc: 96.9% | val acc: 83.8% | current lr: [0.001]
Epoch 25/50 | train loss: 0.129 | val loss: 1.212 | train acc: 96.3% | val acc: 68.4% | current lr: [0.001]
Epoch 26/50 | train loss: 0.122 | val loss: 1.717 | train acc: 96.3% | val acc: 72.7% | current lr: [0.001]
Epoch 27/50 | train loss: 0.096 | val loss: 0.561 | train acc: 97.6% | val acc: 84.6% | current lr: [0.001]
Epoch 28/50 | train loss: 0.091 | val loss: 0.432 | train acc: 97.3% | val acc: 87.2% | current lr: [0.001]
Epoch 29/50 | train loss: 0.068 | val loss: 0.762 | train acc: 98.3% | val acc: 77.5% | current lr: [0.001]
Epoch 30/50 | train loss: 0.081 | val loss: 0.330 | train acc: 97.9% | val acc: 91.3% | current lr: [0.001]
Epoch 31/50 | train loss: 0.060 | val loss: 0.958 | train acc: 98.2% | val acc: 76.1% | current lr: [0.001]
Epoch 32/50 | train loss: 0.073 | val loss: 2.164 | train acc: 97.9% | val acc: 55.9% | current lr: [0.001]
Epoch 33/50 | train loss: 0.063 | val loss: 0.508 | train acc: 98.3% | val acc: 84.8% | current lr: [0.001]
Epoch 34/50 | train loss: 0.051 | val loss: 1.274 | train acc: 98.6% | val acc: 72.7% | current lr: [0.001]
Epoch 35/50 | train loss: 0.062 | val loss: 0.379 | train acc: 98.3% | val acc: 88.6% | current lr: [0.001]
Epoch 36/50 | train loss: 0.036 | val loss: 0.286 | train acc: 99.3% | val acc: 91.2% | current lr: [0.001]
Epoch 37/50 | train loss: 0.027 | val loss: 0.435 | train acc: 99.5% | val acc: 87.6% | current lr: [0.001]
Epoch 38/50 | train loss: 0.040 | val loss: 0.477 | train acc: 99.1% | val acc: 85.7% | current lr: [0.001]
Epoch 39/50 | train loss: 0.032 | val loss: 0.343 | train acc: 99.3% | val acc: 90.9% | current lr: [0.001]
Epoch 40/50 | train loss: 0.045 | val loss: 0.492 | train acc: 98.6% | val acc: 86.0% | current lr: [0.001]
Epoch 41/50 | train loss: 0.050 | val loss: 0.854 | train acc: 98.6% | val acc: 83.3% | current lr: [0.001]
Epoch 42/50 | train loss: 0.043 | val loss: 0.384 | train acc: 99.1% | val acc: 89.3% | current lr: [0.001]
Epoch 43/50 | train loss: 0.061 | val loss: 1.414 | train acc: 98.2% | val acc: 70.6% | current lr: [0.001]
Epoch 44/50 | train loss: 0.077 | val loss: 0.701 | train acc: 98.0% | val acc: 81.4% | current lr: [0.001]
Epoch 45/50 | train loss: 0.034 | val loss: 0.818 | train acc: 99.2% | val acc: 79.5% | current lr: [0.001]
Epoch 46/50 | train loss: 0.074 | val loss: 0.443 | train acc: 97.6% | val acc: 88.6% | current lr: [0.001]
Epoch 47/50 | train loss: 0.031 | val loss: 0.408 | train acc: 99.4% | val acc: 87.7% | current lr: [0.001]
Epoch 48/50 | train loss: 0.008 | val loss: 0.227 | train acc: 100.0% | val acc: 93.0% | current lr: [0.001]
Epoch 49/50 | train loss: 0.004 | val loss: 0.261 | train acc: 100.0% | val acc: 92.9% | current lr: [0.001]
Epoch 50/50 | train loss: 0.007 | val loss: 0.277 | train acc: 99.9% | val acc: 93.3% | current lr: [0.001]=> Gap= 6.6%
Training time: 1759.49 seconds
```

</details>

## Small CNN — Dropout = 0.0 (vanet + neisan merged)

- **Best validation accuracy:** 93.7% (epoch 42)
- **Final validation accuracy:** 90.7%
- **Training time:** 3012.04 seconds (50.20 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.530 | val loss: 1.817 | train acc: 46.2% | val acc: 34.4% | current lr: [0.001]
Epoch 02/50 | train loss: 1.243 | val loss: 1.294 | train acc: 56.9% | val acc: 52.0% | current lr: [0.001]
Epoch 03/50 | train loss: 1.009 | val loss: 1.134 | train acc: 64.6% | val acc: 62.0% | current lr: [0.001]
Epoch 04/50 | train loss: 0.912 | val loss: 1.417 | train acc: 69.3% | val acc: 48.3% | current lr: [0.001]
Epoch 05/50 | train loss: 0.773 | val loss: 1.017 | train acc: 73.8% | val acc: 63.0% | current lr: [0.001]
Epoch 06/50 | train loss: 0.659 | val loss: 0.762 | train acc: 77.6% | val acc: 74.4% | current lr: [0.001]
Epoch 07/50 | train loss: 0.596 | val loss: 0.807 | train acc: 79.8% | val acc: 74.4% | current lr: [0.001]
Epoch 08/50 | train loss: 0.519 | val loss: 0.549 | train acc: 82.6% | val acc: 82.3% | current lr: [0.001]
Epoch 09/50 | train loss: 0.454 | val loss: 0.844 | train acc: 85.5% | val acc: 69.2% | current lr: [0.001]
Epoch 10/50 | train loss: 0.423 | val loss: 0.911 | train acc: 86.5% | val acc: 72.6% | current lr: [0.001]
Epoch 11/50 | train loss: 0.373 | val loss: 0.915 | train acc: 87.6% | val acc: 70.7% | current lr: [0.001]
Epoch 12/50 | train loss: 0.318 | val loss: 0.754 | train acc: 89.7% | val acc: 76.2% | current lr: [0.001]
Epoch 13/50 | train loss: 0.299 | val loss: 0.844 | train acc: 90.9% | val acc: 75.9% | current lr: [0.001]
Epoch 14/50 | train loss: 0.272 | val loss: 0.842 | train acc: 91.8% | val acc: 72.6% | current lr: [0.001]
Epoch 15/50 | train loss: 0.242 | val loss: 0.569 | train acc: 92.3% | val acc: 80.5% | current lr: [0.001]
Epoch 16/50 | train loss: 0.220 | val loss: 1.290 | train acc: 93.0% | val acc: 66.6% | current lr: [0.001]
Epoch 17/50 | train loss: 0.216 | val loss: 0.791 | train acc: 92.6% | val acc: 77.0% | current lr: [0.001]
Epoch 18/50 | train loss: 0.198 | val loss: 0.721 | train acc: 93.7% | val acc: 78.3% | current lr: [0.001]
Epoch 19/50 | train loss: 0.170 | val loss: 0.347 | train acc: 94.4% | val acc: 89.2% | current lr: [0.001]
Epoch 20/50 | train loss: 0.147 | val loss: 0.341 | train acc: 95.7% | val acc: 88.7% | current lr: [0.001]
Epoch 21/50 | train loss: 0.133 | val loss: 0.667 | train acc: 96.1% | val acc: 81.2% | current lr: [0.001]
Epoch 22/50 | train loss: 0.127 | val loss: 0.432 | train acc: 96.1% | val acc: 85.8% | current lr: [0.001]
Epoch 23/50 | train loss: 0.111 | val loss: 1.081 | train acc: 96.7% | val acc: 70.2% | current lr: [0.001]
Epoch 24/50 | train loss: 0.114 | val loss: 0.565 | train acc: 96.6% | val acc: 86.4% | current lr: [0.001]
Epoch 25/50 | train loss: 0.109 | val loss: 0.501 | train acc: 96.7% | val acc: 84.9% | current lr: [0.001]
Epoch 26/50 | train loss: 0.101 | val loss: 1.301 | train acc: 97.3% | val acc: 72.8% | current lr: [0.001]
Epoch 27/50 | train loss: 0.076 | val loss: 1.044 | train acc: 98.0% | val acc: 79.3% | current lr: [0.001]
Epoch 28/50 | train loss: 0.068 | val loss: 0.711 | train acc: 98.0% | val acc: 80.3% | current lr: [0.001]
Epoch 29/50 | train loss: 0.072 | val loss: 0.385 | train acc: 97.9% | val acc: 87.8% | current lr: [0.001]
Epoch 30/50 | train loss: 0.059 | val loss: 0.532 | train acc: 98.4% | val acc: 87.0% | current lr: [0.001]
Epoch 31/50 | train loss: 0.109 | val loss: 0.539 | train acc: 96.7% | val acc: 85.6% | current lr: [0.001]
Epoch 32/50 | train loss: 0.079 | val loss: 0.417 | train acc: 97.7% | val acc: 88.7% | current lr: [0.001]
Epoch 33/50 | train loss: 0.080 | val loss: 0.423 | train acc: 97.6% | val acc: 90.4% | current lr: [0.001]
Epoch 34/50 | train loss: 0.046 | val loss: 0.673 | train acc: 98.8% | val acc: 84.8% | current lr: [0.001]
Epoch 35/50 | train loss: 0.051 | val loss: 0.304 | train acc: 98.5% | val acc: 91.7% | current lr: [0.001]
Epoch 36/50 | train loss: 0.044 | val loss: 0.379 | train acc: 98.8% | val acc: 88.2% | current lr: [0.001]
Epoch 37/50 | train loss: 0.042 | val loss: 0.523 | train acc: 98.8% | val acc: 87.0% | current lr: [0.001]
Epoch 38/50 | train loss: 0.043 | val loss: 0.424 | train acc: 98.8% | val acc: 88.1% | current lr: [0.001]
Epoch 39/50 | train loss: 0.055 | val loss: 0.358 | train acc: 98.6% | val acc: 91.7% | current lr: [0.001]
Epoch 40/50 | train loss: 0.031 | val loss: 0.366 | train acc: 99.2% | val acc: 89.5% | current lr: [0.001]
Epoch 41/50 | train loss: 0.039 | val loss: 1.197 | train acc: 99.1% | val acc: 72.8% | current lr: [0.001]
✅Epoch 42/50 | train loss: 0.040 | val loss: 0.249 | train acc: 99.0% | val acc: 93.7% | current lr: [0.001]=>Gap =5.3%
Epoch 43/50 | train loss: 0.013 | val loss: 0.419 | train acc: 99.8% | val acc: 90.9% | current lr: [0.001]
Epoch 44/50 | train loss: 0.038 | val loss: 0.667 | train acc: 98.9% | val acc: 82.8% | current lr: [0.001]
Epoch 45/50 | train loss: 0.072 | val loss: 0.726 | train acc: 97.5% | val acc: 81.9% | current lr: [0.001]
Epoch 46/50 | train loss: 0.046 | val loss: 0.462 | train acc: 98.7% | val acc: 87.4% | current lr: [0.001]
Epoch 47/50 | train loss: 0.052 | val loss: 0.740 | train acc: 98.5% | val acc: 81.0% | current lr: [0.001]
Epoch 48/50 | train loss: 0.028 | val loss: 0.299 | train acc: 99.4% | val acc: 90.4% | current lr: [0.001]
Epoch 49/50 | train loss: 0.031 | val loss: 0.493 | train acc: 99.1% | val acc: 88.1% | current lr: [0.001]
Epoch 50/50 | train loss: 0.019 | val loss: 0.354 | train acc: 99.6% | val acc: 90.7% | current lr: [0.001]
Training time: 3012.04 seconds
```

</details>

## Small CNN — Augmentation: Horizontal Flip (p = 0.5)

- **Best validation accuracy:** 92.0% (epoch 38)
- **Final validation accuracy:** 74.7%
- **Training time:** 1543.55 seconds (25.73 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.614 | val loss: 1.504 | train acc: 42.7% | val acc: 41.6% | current lr: [0.001]
Epoch 02/50 | train loss: 1.303 | val loss: 1.326 | train acc: 54.1% | val acc: 50.1% | current lr: [0.001]
Epoch 03/50 | train loss: 1.134 | val loss: 1.929 | train acc: 60.1% | val acc: 41.0% | current lr: [0.001]
Epoch 04/50 | train loss: 0.989 | val loss: 1.174 | train acc: 66.0% | val acc: 55.6% | current lr: [0.001]
Epoch 05/50 | train loss: 0.896 | val loss: 1.222 | train acc: 68.7% | val acc: 56.1% | current lr: [0.001]
Epoch 06/50 | train loss: 0.787 | val loss: 1.310 | train acc: 72.3% | val acc: 51.7% | current lr: [0.001]
Epoch 07/50 | train loss: 0.676 | val loss: 0.728 | train acc: 77.2% | val acc: 75.7% | current lr: [0.001]
Epoch 08/50 | train loss: 0.590 | val loss: 0.728 | train acc: 80.7% | val acc: 76.6% | current lr: [0.001]
Epoch 09/50 | train loss: 0.574 | val loss: 1.024 | train acc: 81.9% | val acc: 61.8% | current lr: [0.001]
Epoch 10/50 | train loss: 0.486 | val loss: 0.722 | train acc: 84.0% | val acc: 77.0% | current lr: [0.001]
Epoch 11/50 | train loss: 0.471 | val loss: 1.242 | train acc: 83.5% | val acc: 64.7% | current lr: [0.001]
Epoch 12/50 | train loss: 0.407 | val loss: 0.942 | train acc: 87.1% | val acc: 67.2% | current lr: [0.001]
Epoch 13/50 | train loss: 0.347 | val loss: 0.747 | train acc: 89.4% | val acc: 76.6% | current lr: [0.001]
Epoch 14/50 | train loss: 0.366 | val loss: 0.688 | train acc: 88.6% | val acc: 80.3% | current lr: [0.001]
Epoch 15/50 | train loss: 0.331 | val loss: 0.549 | train acc: 89.1% | val acc: 81.0% | current lr: [0.001]
Epoch 16/50 | train loss: 0.299 | val loss: 1.480 | train acc: 90.3% | val acc: 62.4% | current lr: [0.001]
Epoch 17/50 | train loss: 0.291 | val loss: 0.642 | train acc: 90.7% | val acc: 78.2% | current lr: [0.001]
Epoch 18/50 | train loss: 0.237 | val loss: 0.654 | train acc: 92.2% | val acc: 79.0% | current lr: [0.001]
Epoch 19/50 | train loss: 0.214 | val loss: 0.947 | train acc: 93.2% | val acc: 71.0% | current lr: [0.001]
Epoch 20/50 | train loss: 0.219 | val loss: 0.807 | train acc: 93.0% | val acc: 76.1% | current lr: [0.001]
Epoch 21/50 | train loss: 0.180 | val loss: 0.697 | train acc: 94.5% | val acc: 76.3% | current lr: [0.001]
Epoch 22/50 | train loss: 0.181 | val loss: 0.911 | train acc: 94.5% | val acc: 73.8% | current lr: [0.001]
Epoch 23/50 | train loss: 0.165 | val loss: 0.584 | train acc: 95.0% | val acc: 82.0% | current lr: [0.001]
Epoch 24/50 | train loss: 0.162 | val loss: 0.584 | train acc: 95.0% | val acc: 81.6% | current lr: [0.001]
Epoch 25/50 | train loss: 0.142 | val loss: 0.906 | train acc: 95.5% | val acc: 75.3% | current lr: [0.001]
Epoch 26/50 | train loss: 0.111 | val loss: 0.414 | train acc: 96.5% | val acc: 86.5% | current lr: [0.001]
Epoch 27/50 | train loss: 0.105 | val loss: 0.502 | train acc: 97.4% | val acc: 85.6% | current lr: [0.001]
Epoch 28/50 | train loss: 0.108 | val loss: 0.589 | train acc: 97.3% | val acc: 81.4% | current lr: [0.001]
Epoch 29/50 | train loss: 0.096 | val loss: 0.480 | train acc: 97.2% | val acc: 83.3% | current lr: [0.001]
Epoch 30/50 | train loss: 0.123 | val loss: 0.779 | train acc: 96.4% | val acc: 78.1% | current lr: [0.001]
Epoch 31/50 | train loss: 0.129 | val loss: 1.055 | train acc: 95.9% | val acc: 74.7% | current lr: [0.001]
Epoch 32/50 | train loss: 0.121 | val loss: 1.233 | train acc: 96.2% | val acc: 66.6% | current lr: [0.001]
Epoch 33/50 | train loss: 0.086 | val loss: 0.495 | train acc: 97.5% | val acc: 85.8% | current lr: [0.001]
Epoch 34/50 | train loss: 0.048 | val loss: 0.637 | train acc: 99.0% | val acc: 82.4% | current lr: [0.001]
Epoch 35/50 | train loss: 0.052 | val loss: 0.633 | train acc: 98.7% | val acc: 84.9% | current lr: [0.001]
Epoch 36/50 | train loss: 0.057 | val loss: 0.434 | train acc: 98.7% | val acc: 87.2% | current lr: [0.001]
Epoch 37/50 | train loss: 0.090 | val loss: 0.379 | train acc: 97.6% | val acc: 87.4% | current lr: [0.001]
Epoch 38/50 | train loss: 0.057 | val loss: 0.293 | train acc: 98.6% | val acc: 92.0% | current lr: [0.001]=> Gap=6.6%
Epoch 39/50 | train loss: 0.040 | val loss: 0.524 | train acc: 99.1% | val acc: 85.6% | current lr: [0.001]
Epoch 40/50 | train loss: 0.036 | val loss: 0.583 | train acc: 99.2% | val acc: 84.1% | current lr: [0.001]
Epoch 41/50 | train loss: 0.041 | val loss: 0.533 | train acc: 99.0% | val acc: 84.8% | current lr: [0.001]
Epoch 42/50 | train loss: 0.044 | val loss: 0.635 | train acc: 98.9% | val acc: 84.0% | current lr: [0.001]
Epoch 43/50 | train loss: 0.078 | val loss: 0.821 | train acc: 97.7% | val acc: 80.3% | current lr: [0.001]
Epoch 44/50 | train loss: 0.062 | val loss: 0.576 | train acc: 98.3% | val acc: 83.0% | current lr: [0.001]
Epoch 45/50 | train loss: 0.033 | val loss: 0.587 | train acc: 99.3% | val acc: 84.8% | current lr: [0.001]
Epoch 46/50 | train loss: 0.018 | val loss: 0.321 | train acc: 99.8% | val acc: 90.8% | current lr: [0.001]
Epoch 47/50 | train loss: 0.041 | val loss: 0.609 | train acc: 99.0% | val acc: 86.2% | current lr: [0.001]
Epoch 48/50 | train loss: 0.039 | val loss: 0.519 | train acc: 99.2% | val acc: 87.0% | current lr: [0.001]
Epoch 49/50 | train loss: 0.019 | val loss: 0.392 | train acc: 99.8% | val acc: 90.5% | current lr: [0.001]
Epoch 50/50 | train loss: 0.043 | val loss: 1.266 | train acc: 98.7% | val acc: 74.7% | current lr: [0.001]
Training time: 1543.55 seconds
```

</details>

## Small CNN — Augmentation: Random Horizontal Flip + Gaussian Blur + Random Erasing

- **Best validation accuracy:** 92.4% (epoch 49)
- **Final validation accuracy:** 87.6%
- **Training time:** 4313.12 seconds (71.89 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.664 | val loss: 1.470 | train acc: 39.3% | val acc: 45.3% | current lr: [0.001]
Epoch 02/50 | train loss: 1.373 | val loss: 1.273 | train acc: 51.0% | val acc: 53.5% | current lr: [0.001]
Epoch 03/50 | train loss: 1.234 | val loss: 1.324 | train acc: 55.9% | val acc: 48.7% | current lr: [0.001]
Epoch 04/50 | train loss: 1.124 | val loss: 1.237 | train acc: 60.4% | val acc: 54.8% | current lr: [0.001]
Epoch 05/50 | train loss: 1.037 | val loss: 1.170 | train acc: 63.6% | val acc: 58.6% | current lr: [0.001]
Epoch 06/50 | train loss: 0.908 | val loss: 0.969 | train acc: 68.5% | val acc: 66.3% | current lr: [0.001]
Epoch 07/50 | train loss: 0.846 | val loss: 1.001 | train acc: 70.7% | val acc: 65.9% | current lr: [0.001]
Epoch 08/50 | train loss: 0.810 | val loss: 0.947 | train acc: 72.6% | val acc: 68.6% | current lr: [0.001]
Epoch 09/50 | train loss: 0.687 | val loss: 0.645 | train acc: 76.7% | val acc: 77.4% | current lr: [0.001]
Epoch 10/50 | train loss: 0.620 | val loss: 0.984 | train acc: 78.8% | val acc: 68.3% | current lr: [0.001]
Taking a break...(Cooling GPU)
Epoch 11/50 | train loss: 0.577 | val loss: 0.832 | train acc: 79.9% | val acc: 72.1% | current lr: [0.001]
Epoch 12/50 | train loss: 0.546 | val loss: 0.686 | train acc: 81.7% | val acc: 76.6% | current lr: [0.001]
Epoch 13/50 | train loss: 0.479 | val loss: 0.651 | train acc: 84.0% | val acc: 79.5% | current lr: [0.001]
Epoch 14/50 | train loss: 0.462 | val loss: 0.733 | train acc: 84.4% | val acc: 77.9% | current lr: [0.001]
Epoch 15/50 | train loss: 0.426 | val loss: 1.105 | train acc: 86.5% | val acc: 70.6% | current lr: [0.001]
Epoch 16/50 | train loss: 0.415 | val loss: 0.981 | train acc: 86.6% | val acc: 70.1% | current lr: [0.001]
Epoch 17/50 | train loss: 0.404 | val loss: 0.439 | train acc: 86.7% | val acc: 85.6% | current lr: [0.001]
Epoch 18/50 | train loss: 0.348 | val loss: 0.602 | train acc: 88.8% | val acc: 80.2% | current lr: [0.001]
Epoch 19/50 | train loss: 0.341 | val loss: 0.761 | train acc: 88.6% | val acc: 76.3% | current lr: [0.001]
Epoch 20/50 | train loss: 0.293 | val loss: 0.940 | train acc: 91.2% | val acc: 72.1% | current lr: [0.001]
Taking a break...(Cooling GPU)
Epoch 21/50 | train loss: 0.272 | val loss: 0.964 | train acc: 91.2% | val acc: 72.2% | current lr: [0.001]
Epoch 22/50 | train loss: 0.249 | val loss: 0.644 | train acc: 91.8% | val acc: 80.6% | current lr: [0.001]
Epoch 23/50 | train loss: 0.239 | val loss: 0.493 | train acc: 92.6% | val acc: 84.0% | current lr: [0.001]
Epoch 24/50 | train loss: 0.232 | val loss: 0.744 | train acc: 92.5% | val acc: 78.1% | current lr: [0.001]
Epoch 25/50 | train loss: 0.228 | val loss: 0.733 | train acc: 91.8% | val acc: 77.8% | current lr: [0.001]
Epoch 26/50 | train loss: 0.207 | val loss: 0.485 | train acc: 93.3% | val acc: 85.4% | current lr: [0.001]
Epoch 27/50 | train loss: 0.164 | val loss: 0.542 | train acc: 94.7% | val acc: 83.2% | current lr: [0.001]
Epoch 28/50 | train loss: 0.160 | val loss: 1.077 | train acc: 95.0% | val acc: 71.5% | current lr: [0.001]
Epoch 29/50 | train loss: 0.168 | val loss: 0.535 | train acc: 94.6% | val acc: 84.5% | current lr: [0.001]
Epoch 30/50 | train loss: 0.159 | val loss: 0.677 | train acc: 95.0% | val acc: 79.1% | current lr: [0.001]
Taking a break...(Cooling GPU)
Epoch 31/50 | train loss: 0.165 | val loss: 0.403 | train acc: 94.8% | val acc: 88.9% | current lr: [0.001]
Epoch 32/50 | train loss: 0.185 | val loss: 0.392 | train acc: 94.5% | val acc: 88.8% | current lr: [0.001]
Epoch 33/50 | train loss: 0.137 | val loss: 0.542 | train acc: 95.6% | val acc: 84.2% | current lr: [0.001]
Epoch 34/50 | train loss: 0.136 | val loss: 0.428 | train acc: 95.9% | val acc: 85.4% | current lr: [0.001]
Epoch 35/50 | train loss: 0.132 | val loss: 0.769 | train acc: 95.5% | val acc: 80.9% | current lr: [0.001]
Epoch 36/50 | train loss: 0.091 | val loss: 0.399 | train acc: 97.4% | val acc: 88.1% | current lr: [0.001]
Epoch 37/50 | train loss: 0.092 | val loss: 0.470 | train acc: 97.6% | val acc: 86.9% | current lr: [0.001]
Epoch 38/50 | train loss: 0.111 | val loss: 1.135 | train acc: 96.6% | val acc: 76.3% | current lr: [0.001]
Epoch 39/50 | train loss: 0.110 | val loss: 0.491 | train acc: 96.8% | val acc: 85.2% | current lr: [0.001]
Epoch 40/50 | train loss: 0.096 | val loss: 1.060 | train acc: 97.0% | val acc: 73.8% | current lr: [0.001]
Taking a break...(Cooling GPU)
Epoch 41/50 | train loss: 0.084 | val loss: 1.040 | train acc: 97.2% | val acc: 74.6% | current lr: [0.001]
Epoch 42/50 | train loss: 0.076 | val loss: 0.529 | train acc: 97.7% | val acc: 84.0% | current lr: [0.001]
Epoch 43/50 | train loss: 0.066 | val loss: 0.346 | train acc: 98.1% | val acc: 88.6% | current lr: [0.001]
Epoch 44/50 | train loss: 0.073 | val loss: 0.575 | train acc: 97.9% | val acc: 83.6% | current lr: [0.001]
Epoch 45/50 | train loss: 0.078 | val loss: 0.411 | train acc: 97.4% | val acc: 86.6% | current lr: [0.001]
Epoch 46/50 | train loss: 0.072 | val loss: 0.493 | train acc: 98.1% | val acc: 86.8% | current lr: [0.001]
Epoch 47/50 | train loss: 0.088 | val loss: 0.548 | train acc: 97.4% | val acc: 84.4% | current lr: [0.001]
Epoch 48/50 | train loss: 0.049 | val loss: 0.437 | train acc: 98.7% | val acc: 87.6% | current lr: [0.001]
Epoch 49/50 | train loss: 0.047 | val loss: 0.297 | train acc: 98.7% | val acc: 92.4% | current lr: [0.001] => Gap=6.3%
Epoch 50/50 | train loss: 0.046 | val loss: 0.419 | train acc: 98.7% | val acc: 87.6% | current lr: [0.001]
Taking a break...(Cooling GPU)
Training time: 4313.12 seconds
```

</details>

## Small CNN — Dropout = 0.3

- **Best validation accuracy:** 92.5% (epoch 44)
- **Final validation accuracy:** 83.7%
- **Training time:** 1545.22 seconds (25.75 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.634 | val loss: 1.416 | train acc: 40.5% | val acc: 48.3% | current lr: [0.001]
Epoch 02/50 | train loss: 1.325 | val loss: 1.563 | train acc: 54.4% | val acc: 48.5% | current lr: [0.001]
Epoch 03/50 | train loss: 1.162 | val loss: 2.089 | train acc: 59.3% | val acc: 35.3% | current lr: [0.001]
Epoch 04/50 | train loss: 1.038 | val loss: 1.733 | train acc: 62.9% | val acc: 48.0% | current lr: [0.001]
Epoch 05/50 | train loss: 0.907 | val loss: 1.477 | train acc: 67.8% | val acc: 49.3% | current lr: [0.001]
Epoch 06/50 | train loss: 0.798 | val loss: 1.485 | train acc: 73.2% | val acc: 54.7% | current lr: [0.001]
Epoch 07/50 | train loss: 0.712 | val loss: 1.040 | train acc: 75.8% | val acc: 61.6% | current lr: [0.001]
Epoch 08/50 | train loss: 0.635 | val loss: 0.858 | train acc: 79.3% | val acc: 71.7% | current lr: [0.001]
Epoch 09/50 | train loss: 0.562 | val loss: 0.821 | train acc: 81.0% | val acc: 69.3% | current lr: [0.001]
Epoch 10/50 | train loss: 0.462 | val loss: 1.178 | train acc: 85.2% | val acc: 63.6% | current lr: [0.001]
Epoch 11/50 | train loss: 0.487 | val loss: 0.721 | train acc: 84.4% | val acc: 75.4% | current lr: [0.001]
Epoch 12/50 | train loss: 0.435 | val loss: 1.057 | train acc: 85.3% | val acc: 67.6% | current lr: [0.001]
Epoch 13/50 | train loss: 0.387 | val loss: 0.882 | train acc: 87.5% | val acc: 70.2% | current lr: [0.001]
Epoch 14/50 | train loss: 0.349 | val loss: 0.682 | train acc: 88.3% | val acc: 77.3% | current lr: [0.001]
Epoch 15/50 | train loss: 0.318 | val loss: 0.683 | train acc: 89.8% | val acc: 76.5% | current lr: [0.001]
Epoch 16/50 | train loss: 0.325 | val loss: 1.366 | train acc: 88.9% | val acc: 60.7% | current lr: [0.001]
Epoch 17/50 | train loss: 0.275 | val loss: 0.619 | train acc: 91.5% | val acc: 79.5% | current lr: [0.001]
Epoch 18/50 | train loss: 0.255 | val loss: 1.483 | train acc: 91.7% | val acc: 59.8% | current lr: [0.001]
Epoch 19/50 | train loss: 0.263 | val loss: 0.960 | train acc: 91.7% | val acc: 73.1% | current lr: [0.001]
Epoch 20/50 | train loss: 0.242 | val loss: 1.012 | train acc: 91.9% | val acc: 73.0% | current lr: [0.001]
Epoch 21/50 | train loss: 0.219 | val loss: 0.689 | train acc: 93.1% | val acc: 76.7% | current lr: [0.001]
Epoch 22/50 | train loss: 0.190 | val loss: 0.402 | train acc: 94.0% | val acc: 86.6% | current lr: [0.001]
Epoch 23/50 | train loss: 0.171 | val loss: 1.173 | train acc: 94.6% | val acc: 68.2% | current lr: [0.001]
Epoch 24/50 | train loss: 0.171 | val loss: 0.491 | train acc: 94.1% | val acc: 82.4% | current lr: [0.001]
Epoch 25/50 | train loss: 0.174 | val loss: 1.116 | train acc: 94.4% | val acc: 71.1% | current lr: [0.001]
Epoch 26/50 | train loss: 0.163 | val loss: 2.237 | train acc: 94.9% | val acc: 63.6% | current lr: [0.001]
Epoch 27/50 | train loss: 0.156 | val loss: 0.888 | train acc: 95.0% | val acc: 78.7% | current lr: [0.001]
Epoch 28/50 | train loss: 0.124 | val loss: 0.860 | train acc: 96.0% | val acc: 77.8% | current lr: [0.001]
Epoch 29/50 | train loss: 0.124 | val loss: 0.544 | train acc: 96.0% | val acc: 81.1% | current lr: [0.001]
Epoch 30/50 | train loss: 0.117 | val loss: 0.381 | train acc: 96.4% | val acc: 88.9% | current lr: [0.001]
Epoch 31/50 | train loss: 0.101 | val loss: 1.142 | train acc: 96.7% | val acc: 70.2% | current lr: [0.001]
Epoch 32/50 | train loss: 0.117 | val loss: 0.522 | train acc: 96.1% | val acc: 83.4% | current lr: [0.001]
Epoch 33/50 | train loss: 0.098 | val loss: 0.837 | train acc: 97.2% | val acc: 73.1% | current lr: [0.001]
Epoch 34/50 | train loss: 0.079 | val loss: 0.633 | train acc: 97.6% | val acc: 83.2% | current lr: [0.001]
Epoch 35/50 | train loss: 0.110 | val loss: 0.983 | train acc: 96.6% | val acc: 73.7% | current lr: [0.001]
Epoch 36/50 | train loss: 0.072 | val loss: 0.316 | train acc: 98.2% | val acc: 92.0% | current lr: [0.001]
Epoch 37/50 | train loss: 0.074 | val loss: 0.808 | train acc: 97.9% | val acc: 75.5% | current lr: [0.001]
Epoch 38/50 | train loss: 0.105 | val loss: 0.446 | train acc: 96.9% | val acc: 87.2% | current lr: [0.001]
Epoch 39/50 | train loss: 0.065 | val loss: 0.372 | train acc: 98.1% | val acc: 89.3% | current lr: [0.001]
Epoch 40/50 | train loss: 0.056 | val loss: 0.433 | train acc: 98.4% | val acc: 85.8% | current lr: [0.001]
Epoch 41/50 | train loss: 0.057 | val loss: 0.577 | train acc: 98.2% | val acc: 83.3% | current lr: [0.001]
Epoch 42/50 | train loss: 0.049 | val loss: 0.399 | train acc: 98.8% | val acc: 87.2% | current lr: [0.001]
Epoch 43/50 | train loss: 0.044 | val loss: 0.361 | train acc: 98.9% | val acc: 90.5% | current lr: [0.001]
Epoch 44/50 | train loss: 0.039 | val loss: 0.253 | train acc: 98.9% | val acc: 92.5% | current lr: [0.001]=>Gap=6.4%
Epoch 45/50 | train loss: 0.034 | val loss: 0.574 | train acc: 99.2% | val acc: 84.5% | current lr: [0.001]
Epoch 46/50 | train loss: 0.055 | val loss: 0.637 | train acc: 98.4% | val acc: 84.5% | current lr: [0.001]
Epoch 47/50 | train loss: 0.073 | val loss: 0.455 | train acc: 97.8% | val acc: 87.0% | current lr: [0.001]
Epoch 48/50 | train loss: 0.041 | val loss: 0.453 | train acc: 98.8% | val acc: 85.0% | current lr: [0.001]
Epoch 49/50 | train loss: 0.038 | val loss: 2.518 | train acc: 98.9% | val acc: 63.9% | current lr: [0.001]
Epoch 50/50 | train loss: 0.050 | val loss: 0.653 | train acc: 98.7% | val acc: 83.7% | current lr: [0.001]
Training time: 1545.22 seconds
```

</details>

## Small CNN — Dropout = 0.5

- **Best validation accuracy:** 92.6% (epoch 44)
- **Final validation accuracy:** 89.6%
- **Training time:** 1510.49 seconds (25.17 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.659 | val loss: 1.623 | train acc: 39.3% | val acc: 42.4% | current lr: [0.001]
Epoch 02/50 | train loss: 1.345 | val loss: 1.348 | train acc: 52.9% | val acc: 52.9% | current lr: [0.001]
Epoch 03/50 | train loss: 1.175 | val loss: 1.520 | train acc: 59.9% | val acc: 43.9% | current lr: [0.001]
Epoch 04/50 | train loss: 1.071 | val loss: 2.334 | train acc: 61.9% | val acc: 32.2% | current lr: [0.001]
Epoch 05/50 | train loss: 0.943 | val loss: 1.634 | train acc: 66.8% | val acc: 41.4% | current lr: [0.001]
Epoch 06/50 | train loss: 0.844 | val loss: 1.396 | train acc: 70.8% | val acc: 54.5% | current lr: [0.001]
Epoch 07/50 | train loss: 0.767 | val loss: 1.598 | train acc: 73.8% | val acc: 47.6% | current lr: [0.001]
Epoch 08/50 | train loss: 0.692 | val loss: 1.347 | train acc: 76.8% | val acc: 55.7% | current lr: [0.001]
Epoch 09/50 | train loss: 0.609 | val loss: 0.670 | train acc: 79.8% | val acc: 74.9% | current lr: [0.001]
Epoch 10/50 | train loss: 0.533 | val loss: 0.761 | train acc: 82.1% | val acc: 75.1% | current lr: [0.001]
Epoch 11/50 | train loss: 0.544 | val loss: 0.551 | train acc: 81.5% | val acc: 80.6% | current lr: [0.001]
Epoch 12/50 | train loss: 0.479 | val loss: 0.638 | train acc: 84.2% | val acc: 79.1% | current lr: [0.001]
Epoch 13/50 | train loss: 0.432 | val loss: 0.552 | train acc: 85.8% | val acc: 79.5% | current lr: [0.001]
Epoch 14/50 | train loss: 0.399 | val loss: 0.566 | train acc: 86.5% | val acc: 80.6% | current lr: [0.001]
Epoch 15/50 | train loss: 0.374 | val loss: 0.499 | train acc: 87.4% | val acc: 83.3% | current lr: [0.001]
Epoch 16/50 | train loss: 0.368 | val loss: 0.858 | train acc: 87.4% | val acc: 70.3% | current lr: [0.001]
Epoch 17/50 | train loss: 0.343 | val loss: 0.676 | train acc: 88.9% | val acc: 76.3% | current lr: [0.001]
Epoch 18/50 | train loss: 0.301 | val loss: 0.587 | train acc: 90.2% | val acc: 79.4% | current lr: [0.001]
Epoch 19/50 | train loss: 0.295 | val loss: 0.642 | train acc: 90.6% | val acc: 77.5% | current lr: [0.001]
Epoch 20/50 | train loss: 0.296 | val loss: 1.213 | train acc: 90.1% | val acc: 76.2% | current lr: [0.001]
Epoch 21/50 | train loss: 0.265 | val loss: 1.161 | train acc: 91.0% | val acc: 66.2% | current lr: [0.001]
Epoch 22/50 | train loss: 0.236 | val loss: 0.700 | train acc: 92.3% | val acc: 79.1% | current lr: [0.001]
Epoch 23/50 | train loss: 0.203 | val loss: 0.969 | train acc: 93.6% | val acc: 70.6% | current lr: [0.001]
Epoch 24/50 | train loss: 0.226 | val loss: 0.498 | train acc: 92.5% | val acc: 83.8% | current lr: [0.001]
Epoch 25/50 | train loss: 0.206 | val loss: 1.309 | train acc: 93.3% | val acc: 71.8% | current lr: [0.001]
Epoch 26/50 | train loss: 0.243 | val loss: 0.437 | train acc: 92.1% | val acc: 85.4% | current lr: [0.001]
Epoch 27/50 | train loss: 0.186 | val loss: 1.093 | train acc: 94.2% | val acc: 66.8% | current lr: [0.001]
Epoch 28/50 | train loss: 0.183 | val loss: 0.466 | train acc: 94.2% | val acc: 84.4% | current lr: [0.001]
Epoch 29/50 | train loss: 0.171 | val loss: 0.503 | train acc: 94.3% | val acc: 83.8% | current lr: [0.001]
Epoch 30/50 | train loss: 0.134 | val loss: 0.289 | train acc: 96.0% | val acc: 90.2% | current lr: [0.001]
Epoch 31/50 | train loss: 0.150 | val loss: 0.677 | train acc: 94.9% | val acc: 76.6% | current lr: [0.001]
Epoch 32/50 | train loss: 0.166 | val loss: 0.716 | train acc: 94.2% | val acc: 80.1% | current lr: [0.001]
Epoch 33/50 | train loss: 0.161 | val loss: 0.415 | train acc: 94.8% | val acc: 86.0% | current lr: [0.001]
Epoch 34/50 | train loss: 0.128 | val loss: 0.369 | train acc: 95.8% | val acc: 89.0% | current lr: [0.001]
Epoch 35/50 | train loss: 0.108 | val loss: 0.398 | train acc: 96.9% | val acc: 88.4% | current lr: [0.001]
Epoch 36/50 | train loss: 0.127 | val loss: 0.636 | train acc: 95.9% | val acc: 84.0% | current lr: [0.001]
Epoch 37/50 | train loss: 0.094 | val loss: 0.537 | train acc: 97.3% | val acc: 82.8% | current lr: [0.001]
Epoch 38/50 | train loss: 0.114 | val loss: 0.416 | train acc: 96.2% | val acc: 88.2% | current lr: [0.001]
Epoch 39/50 | train loss: 0.080 | val loss: 0.477 | train acc: 97.4% | val acc: 86.2% | current lr: [0.001]
Epoch 40/50 | train loss: 0.117 | val loss: 0.587 | train acc: 96.1% | val acc: 82.1% | current lr: [0.001]
Epoch 41/50 | train loss: 0.087 | val loss: 0.741 | train acc: 97.3% | val acc: 77.4% | current lr: [0.001]
Epoch 42/50 | train loss: 0.095 | val loss: 0.369 | train acc: 97.3% | val acc: 88.1% | current lr: [0.001]
Epoch 43/50 | train loss: 0.081 | val loss: 0.402 | train acc: 97.5% | val acc: 88.8% | current lr: [0.001]
Epoch 44/50 | train loss: 0.067 | val loss: 0.233 | train acc: 98.1% | val acc: 92.6% | current lr: [0.001]=>Gap=5.5%
Epoch 45/50 | train loss: 0.064 | val loss: 0.839 | train acc: 98.2% | val acc: 80.6% | current lr: [0.001]
Epoch 46/50 | train loss: 0.090 | val loss: 0.731 | train acc: 97.2% | val acc: 81.3% | current lr: [0.001]
Epoch 47/50 | train loss: 0.079 | val loss: 0.389 | train acc: 97.8% | val acc: 89.8% | current lr: [0.001]
Epoch 48/50 | train loss: 0.068 | val loss: 0.346 | train acc: 97.7% | val acc: 88.6% | current lr: [0.001]
Epoch 49/50 | train loss: 0.050 | val loss: 1.274 | train acc: 98.6% | val acc: 74.6% | current lr: [0.001]
Epoch 50/50 | train loss: 0.070 | val loss: 0.350 | train acc: 97.8% | val acc: 89.6% | current lr: [0.001]
Training time: 1510.49 seconds
```

</details>

## Small CNN — Average Pooling

- **Best validation accuracy:** 91.6% (epoch 47)
- **Final validation accuracy:** 91.0%
- **Training time:** 2723.03 seconds (45.38 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.703 | val loss: 1.529 | train acc: 37.5% | val acc: 43.7% | current lr: [0.001]
Epoch 02/50 | train loss: 1.453 | val loss: 1.393 | train acc: 48.7% | val acc: 48.4% | current lr: [0.001]
Epoch 03/50 | train loss: 1.253 | val loss: 1.303 | train acc: 55.6% | val acc: 54.3% | current lr: [0.001]
Epoch 04/50 | train loss: 1.137 | val loss: 1.125 | train acc: 59.3% | val acc: 58.3% | current lr: [0.001]
Epoch 05/50 | train loss: 1.061 | val loss: 2.040 | train acc: 63.1% | val acc: 35.6% | current lr: [0.001]
Epoch 06/50 | train loss: 0.965 | val loss: 1.058 | train acc: 66.7% | val acc: 62.0% | current lr: [0.001]
Epoch 07/50 | train loss: 0.864 | val loss: 0.868 | train acc: 70.1% | val acc: 72.6% | current lr: [0.001]
Epoch 08/50 | train loss: 0.794 | val loss: 0.862 | train acc: 73.2% | val acc: 68.2% | current lr: [0.001]
Epoch 09/50 | train loss: 0.736 | val loss: 0.941 | train acc: 74.9% | val acc: 67.6% | current lr: [0.001]
Epoch 10/50 | train loss: 0.641 | val loss: 0.830 | train acc: 78.8% | val acc: 70.7% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 11/50 | train loss: 0.607 | val loss: 0.771 | train acc: 80.2% | val acc: 72.9% | current lr: [0.001]
Epoch 12/50 | train loss: 0.546 | val loss: 0.585 | train acc: 81.5% | val acc: 81.8% | current lr: [0.001]
Epoch 13/50 | train loss: 0.508 | val loss: 0.821 | train acc: 83.8% | val acc: 73.1% | current lr: [0.001]
Epoch 14/50 | train loss: 0.467 | val loss: 0.484 | train acc: 84.8% | val acc: 85.4% | current lr: [0.001]
Epoch 15/50 | train loss: 0.425 | val loss: 0.612 | train acc: 85.7% | val acc: 80.1% | current lr: [0.001]
Epoch 16/50 | train loss: 0.594 | val loss: 0.702 | train acc: 80.2% | val acc: 76.3% | current lr: [0.001]
Epoch 17/50 | train loss: 0.397 | val loss: 0.515 | train acc: 86.8% | val acc: 83.4% | current lr: [0.001]
Epoch 18/50 | train loss: 0.338 | val loss: 0.592 | train acc: 88.6% | val acc: 81.1% | current lr: [0.001]
Epoch 19/50 | train loss: 0.339 | val loss: 0.539 | train acc: 88.5% | val acc: 82.5% | current lr: [0.001]
Epoch 20/50 | train loss: 0.330 | val loss: 0.751 | train acc: 88.6% | val acc: 76.7% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 21/50 | train loss: 0.312 | val loss: 0.514 | train acc: 90.0% | val acc: 84.0% | current lr: [0.001]
Epoch 22/50 | train loss: 0.266 | val loss: 0.515 | train acc: 90.9% | val acc: 82.2% | current lr: [0.001]
Epoch 23/50 | train loss: 0.240 | val loss: 0.485 | train acc: 92.2% | val acc: 84.6% | current lr: [0.001]
Epoch 24/50 | train loss: 0.236 | val loss: 0.425 | train acc: 92.5% | val acc: 86.0% | current lr: [0.001]
Epoch 25/50 | train loss: 0.210 | val loss: 0.590 | train acc: 93.7% | val acc: 79.7% | current lr: [0.001]
Epoch 26/50 | train loss: 0.243 | val loss: 0.507 | train acc: 92.1% | val acc: 84.2% | current lr: [0.001]
Epoch 27/50 | train loss: 0.206 | val loss: 0.421 | train acc: 93.1% | val acc: 87.3% | current lr: [0.001]
Epoch 28/50 | train loss: 0.197 | val loss: 0.788 | train acc: 93.6% | val acc: 77.3% | current lr: [0.001]
Epoch 29/50 | train loss: 0.174 | val loss: 0.530 | train acc: 95.0% | val acc: 84.0% | current lr: [0.001]
Epoch 30/50 | train loss: 0.173 | val loss: 0.438 | train acc: 94.4% | val acc: 87.3% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 31/50 | train loss: 0.137 | val loss: 0.560 | train acc: 95.9% | val acc: 84.0% | current lr: [0.001]
Epoch 32/50 | train loss: 0.170 | val loss: 0.726 | train acc: 94.3% | val acc: 80.6% | current lr: [0.001]
Epoch 33/50 | train loss: 0.141 | val loss: 0.390 | train acc: 95.7% | val acc: 86.6% | current lr: [0.001]
Epoch 34/50 | train loss: 0.122 | val loss: 0.397 | train acc: 96.4% | val acc: 87.6% | current lr: [0.001]
Epoch 35/50 | train loss: 0.135 | val loss: 0.489 | train acc: 95.8% | val acc: 87.3% | current lr: [0.001]
Epoch 36/50 | train loss: 0.128 | val loss: 0.439 | train acc: 95.7% | val acc: 87.2% | current lr: [0.001]
Epoch 37/50 | train loss: 0.108 | val loss: 0.583 | train acc: 96.8% | val acc: 84.0% | current lr: [0.001]
Epoch 38/50 | train loss: 0.125 | val loss: 0.848 | train acc: 96.1% | val acc: 77.1% | current lr: [0.001]
Epoch 39/50 | train loss: 0.107 | val loss: 0.874 | train acc: 96.6% | val acc: 77.5% | current lr: [0.001]
Epoch 40/50 | train loss: 0.091 | val loss: 0.445 | train acc: 97.3% | val acc: 86.9% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 41/50 | train loss: 0.100 | val loss: 0.502 | train acc: 97.2% | val acc: 84.8% | current lr: [0.001]
Epoch 42/50 | train loss: 0.089 | val loss: 0.452 | train acc: 97.3% | val acc: 86.6% | current lr: [0.001]
Epoch 43/50 | train loss: 0.082 | val loss: 0.496 | train acc: 97.7% | val acc: 84.0% | current lr: [0.001]
Epoch 44/50 | train loss: 0.075 | val loss: 0.391 | train acc: 97.7% | val acc: 88.4% | current lr: [0.001]
Epoch 45/50 | train loss: 0.060 | val loss: 0.511 | train acc: 98.4% | val acc: 85.2% | current lr: [0.001]
Epoch 46/50 | train loss: 0.055 | val loss: 0.647 | train acc: 98.6% | val acc: 82.8% | current lr: [0.001]
Epoch 47/50 | train loss: 0.074 | val loss: 0.330 | train acc: 97.7% | val acc: 91.6% | current lr: [0.001]=>Gap=6.1%
Epoch 48/50 | train loss: 0.060 | val loss: 0.543 | train acc: 98.4% | val acc: 86.1% | current lr: [0.001]
Epoch 49/50 | train loss: 0.084 | val loss: 0.390 | train acc: 96.9% | val acc: 90.6% | current lr: [0.001]
Epoch 50/50 | train loss: 0.059 | val loss: 0.346 | train acc: 98.4% | val acc: 91.0% | current lr: [0.001]
Training time: 2723.03 seconds
```

</details>

## Small CNN — Weight Decay = 0

- **Best validation accuracy:** 93.3% (epoch 50)
- **Final validation accuracy:** 93.3%
- **Training time:** 3764.15 seconds (62.74 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.596 | val loss: 1.612 | train acc: 43.1% | val acc: 43.3% | current lr: [0.001]
Epoch 02/50 | train loss: 1.266 | val loss: 1.609 | train acc: 56.5% | val acc: 43.7% | current lr: [0.001]
Epoch 03/50 | train loss: 1.082 | val loss: 1.095 | train acc: 62.0% | val acc: 60.3% | current lr: [0.001]
Epoch 04/50 | train loss: 0.945 | val loss: 1.361 | train acc: 66.1% | val acc: 51.6% | current lr: [0.001]
Epoch 05/50 | train loss: 0.807 | val loss: 2.192 | train acc: 72.0% | val acc: 40.6% | current lr: [0.001]
Epoch 06/50 | train loss: 0.696 | val loss: 1.133 | train acc: 76.3% | val acc: 67.2% | current lr: [0.001]
Epoch 07/50 | train loss: 0.615 | val loss: 2.080 | train acc: 79.6% | val acc: 41.0% | current lr: [0.001]
Epoch 08/50 | train loss: 0.571 | val loss: 0.857 | train acc: 81.5% | val acc: 72.9% | current lr: [0.001]
Epoch 09/50 | train loss: 0.484 | val loss: 0.524 | train acc: 83.9% | val acc: 82.6% | current lr: [0.001]
Epoch 10/50 | train loss: 0.405 | val loss: 1.204 | train acc: 86.9% | val acc: 65.0% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 11/50 | train loss: 0.408 | val loss: 0.510 | train acc: 86.5% | val acc: 83.4% | current lr: [0.001]
Epoch 12/50 | train loss: 0.373 | val loss: 1.127 | train acc: 87.3% | val acc: 65.4% | current lr: [0.001]
Epoch 13/50 | train loss: 0.337 | val loss: 0.695 | train acc: 89.2% | val acc: 76.2% | current lr: [0.001]
Epoch 14/50 | train loss: 0.305 | val loss: 0.583 | train acc: 90.5% | val acc: 80.7% | current lr: [0.001]
Epoch 15/50 | train loss: 0.266 | val loss: 0.528 | train acc: 91.6% | val acc: 82.6% | current lr: [0.001]
Epoch 16/50 | train loss: 0.275 | val loss: 1.009 | train acc: 91.7% | val acc: 69.8% | current lr: [0.001]
Epoch 17/50 | train loss: 0.210 | val loss: 1.284 | train acc: 93.3% | val acc: 62.3% | current lr: [0.001]
Epoch 18/50 | train loss: 0.226 | val loss: 0.699 | train acc: 92.6% | val acc: 77.4% | current lr: [0.001]
Epoch 19/50 | train loss: 0.202 | val loss: 0.729 | train acc: 93.6% | val acc: 75.1% | current lr: [0.001]
Epoch 20/50 | train loss: 0.186 | val loss: 1.437 | train acc: 94.1% | val acc: 66.8% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 21/50 | train loss: 0.164 | val loss: 1.661 | train acc: 95.1% | val acc: 55.3% | current lr: [0.001]
Epoch 22/50 | train loss: 0.134 | val loss: 0.414 | train acc: 95.8% | val acc: 88.9% | current lr: [0.001]
Epoch 23/50 | train loss: 0.117 | val loss: 0.524 | train acc: 96.7% | val acc: 83.7% | current lr: [0.001]
Epoch 24/50 | train loss: 0.116 | val loss: 0.601 | train acc: 96.9% | val acc: 83.8% | current lr: [0.001]
Epoch 25/50 | train loss: 0.129 | val loss: 1.212 | train acc: 96.3% | val acc: 68.4% | current lr: [0.001]
Epoch 26/50 | train loss: 0.122 | val loss: 1.717 | train acc: 96.3% | val acc: 72.7% | current lr: [0.001]
Epoch 27/50 | train loss: 0.096 | val loss: 0.561 | train acc: 97.6% | val acc: 84.6% | current lr: [0.001]
Epoch 28/50 | train loss: 0.091 | val loss: 0.432 | train acc: 97.3% | val acc: 87.2% | current lr: [0.001]
Epoch 29/50 | train loss: 0.068 | val loss: 0.762 | train acc: 98.3% | val acc: 77.5% | current lr: [0.001]
Epoch 30/50 | train loss: 0.081 | val loss: 0.330 | train acc: 97.9% | val acc: 91.3% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 31/50 | train loss: 0.060 | val loss: 0.958 | train acc: 98.2% | val acc: 76.1% | current lr: [0.001]
Epoch 32/50 | train loss: 0.073 | val loss: 2.164 | train acc: 97.9% | val acc: 55.9% | current lr: [0.001]
Epoch 33/50 | train loss: 0.063 | val loss: 0.508 | train acc: 98.3% | val acc: 84.8% | current lr: [0.001]
Epoch 34/50 | train loss: 0.051 | val loss: 1.274 | train acc: 98.6% | val acc: 72.7% | current lr: [0.001]
Epoch 35/50 | train loss: 0.062 | val loss: 0.379 | train acc: 98.3% | val acc: 88.6% | current lr: [0.001]
Epoch 36/50 | train loss: 0.036 | val loss: 0.286 | train acc: 99.3% | val acc: 91.2% | current lr: [0.001]
Epoch 37/50 | train loss: 0.027 | val loss: 0.435 | train acc: 99.5% | val acc: 87.6% | current lr: [0.001]
Epoch 38/50 | train loss: 0.040 | val loss: 0.477 | train acc: 99.1% | val acc: 85.7% | current lr: [0.001]
Epoch 39/50 | train loss: 0.032 | val loss: 0.343 | train acc: 99.3% | val acc: 90.9% | current lr: [0.001]
Epoch 40/50 | train loss: 0.045 | val loss: 0.492 | train acc: 98.6% | val acc: 86.0% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 41/50 | train loss: 0.050 | val loss: 0.854 | train acc: 98.6% | val acc: 83.3% | current lr: [0.001]
Epoch 42/50 | train loss: 0.043 | val loss: 0.384 | train acc: 99.1% | val acc: 89.3% | current lr: [0.001]
Epoch 43/50 | train loss: 0.061 | val loss: 1.414 | train acc: 98.2% | val acc: 70.6% | current lr: [0.001]
Epoch 44/50 | train loss: 0.077 | val loss: 0.701 | train acc: 98.0% | val acc: 81.4% | current lr: [0.001]
Epoch 45/50 | train loss: 0.034 | val loss: 0.818 | train acc: 99.2% | val acc: 79.5% | current lr: [0.001]
Epoch 46/50 | train loss: 0.074 | val loss: 0.443 | train acc: 97.6% | val acc: 88.6% | current lr: [0.001]
Epoch 47/50 | train loss: 0.031 | val loss: 0.408 | train acc: 99.4% | val acc: 87.7% | current lr: [0.001]
Epoch 48/50 | train loss: 0.008 | val loss: 0.227 | train acc: 100.0% | val acc: 93.0% | current lr: [0.001]
Epoch 49/50 | train loss: 0.004 | val loss: 0.261 | train acc: 100.0% | val acc: 92.9% | current lr: [0.001]
Epoch 50/50 | train loss: 0.007 | val loss: 0.277 | train acc: 99.9% | val acc: 93.3% | current lr: [0.001]=>Gap=6.6%
Training time: 3764.15 seconds
```

</details>

## Small CNN — Weight Decay = 0.0001

- **Best validation accuracy:** 93.4% (epoch 50)
- **Final validation accuracy:** 93.4%
- **Training time:** 3785.13 seconds (63.09 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.596 | val loss: 1.560 | train acc: 43.0% | val acc: 45.5% | current lr: [0.001]
Epoch 02/50 | train loss: 1.269 | val loss: 1.999 | train acc: 56.6% | val acc: 32.9% | current lr: [0.001]
Epoch 03/50 | train loss: 1.077 | val loss: 1.198 | train acc: 62.3% | val acc: 56.4% | current lr: [0.001]
Epoch 04/50 | train loss: 0.957 | val loss: 2.061 | train acc: 66.5% | val acc: 37.7% | current lr: [0.001]
Epoch 05/50 | train loss: 0.806 | val loss: 1.903 | train acc: 71.8% | val acc: 40.5% | current lr: [0.001]
Epoch 06/50 | train loss: 0.718 | val loss: 1.052 | train acc: 75.6% | val acc: 66.4% | current lr: [0.001]
Epoch 07/50 | train loss: 0.609 | val loss: 1.165 | train acc: 79.8% | val acc: 59.5% | current lr: [0.001]
Epoch 08/50 | train loss: 0.565 | val loss: 1.236 | train acc: 81.6% | val acc: 63.6% | current lr: [0.001]
Epoch 09/50 | train loss: 0.483 | val loss: 0.696 | train acc: 84.2% | val acc: 77.3% | current lr: [0.001]
Epoch 10/50 | train loss: 0.403 | val loss: 0.949 | train acc: 87.4% | val acc: 69.9% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 11/50 | train loss: 0.417 | val loss: 0.482 | train acc: 86.5% | val acc: 83.8% | current lr: [0.001]
Epoch 12/50 | train loss: 0.358 | val loss: 0.756 | train acc: 88.3% | val acc: 76.9% | current lr: [0.001]
Epoch 13/50 | train loss: 0.328 | val loss: 0.896 | train acc: 89.5% | val acc: 69.1% | current lr: [0.001]
Epoch 14/50 | train loss: 0.280 | val loss: 0.378 | train acc: 91.0% | val acc: 88.8% | current lr: [0.001]
Epoch 15/50 | train loss: 0.271 | val loss: 0.814 | train acc: 91.4% | val acc: 72.3% | current lr: [0.001]
Epoch 16/50 | train loss: 0.281 | val loss: 0.834 | train acc: 91.2% | val acc: 73.1% | current lr: [0.001]
Epoch 17/50 | train loss: 0.213 | val loss: 0.925 | train acc: 93.4% | val acc: 69.7% | current lr: [0.001]
Epoch 18/50 | train loss: 0.189 | val loss: 0.646 | train acc: 94.0% | val acc: 79.5% | current lr: [0.001]
Epoch 19/50 | train loss: 0.200 | val loss: 0.787 | train acc: 93.4% | val acc: 77.1% | current lr: [0.001]
Epoch 20/50 | train loss: 0.172 | val loss: 1.169 | train acc: 94.7% | val acc: 71.0% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 21/50 | train loss: 0.158 | val loss: 0.683 | train acc: 95.4% | val acc: 76.9% | current lr: [0.001]
Epoch 22/50 | train loss: 0.130 | val loss: 0.564 | train acc: 96.2% | val acc: 83.4% | current lr: [0.001]
Epoch 23/50 | train loss: 0.109 | val loss: 1.055 | train acc: 96.9% | val acc: 74.3% | current lr: [0.001]
Epoch 24/50 | train loss: 0.117 | val loss: 0.570 | train acc: 96.7% | val acc: 81.6% | current lr: [0.001]
Epoch 25/50 | train loss: 0.099 | val loss: 0.873 | train acc: 97.2% | val acc: 76.6% | current lr: [0.001]
Epoch 26/50 | train loss: 0.109 | val loss: 0.671 | train acc: 96.8% | val acc: 81.6% | current lr: [0.001]
Epoch 27/50 | train loss: 0.106 | val loss: 0.733 | train acc: 97.1% | val acc: 79.3% | current lr: [0.001]
Epoch 28/50 | train loss: 0.081 | val loss: 1.036 | train acc: 97.8% | val acc: 72.9% | current lr: [0.001]
Epoch 29/50 | train loss: 0.068 | val loss: 0.684 | train acc: 98.0% | val acc: 80.5% | current lr: [0.001]
Epoch 30/50 | train loss: 0.088 | val loss: 0.325 | train acc: 97.7% | val acc: 88.9% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 31/50 | train loss: 0.045 | val loss: 0.973 | train acc: 98.9% | val acc: 74.5% | current lr: [0.001]
Epoch 32/50 | train loss: 0.036 | val loss: 0.377 | train acc: 99.5% | val acc: 89.6% | current lr: [0.001]
Epoch 33/50 | train loss: 0.028 | val loss: 0.258 | train acc: 99.5% | val acc: 90.9% | current lr: [0.001]
Epoch 34/50 | train loss: 0.039 | val loss: 0.773 | train acc: 99.1% | val acc: 78.3% | current lr: [0.001]
Epoch 35/50 | train loss: 0.086 | val loss: 2.811 | train acc: 97.2% | val acc: 58.7% | current lr: [0.001]
Epoch 36/50 | train loss: 0.057 | val loss: 0.465 | train acc: 98.3% | val acc: 87.3% | current lr: [0.001]
Epoch 37/50 | train loss: 0.052 | val loss: 0.476 | train acc: 98.4% | val acc: 86.1% | current lr: [0.001]
Epoch 38/50 | train loss: 0.057 | val loss: 0.479 | train acc: 98.3% | val acc: 86.4% | current lr: [0.001]
Epoch 39/50 | train loss: 0.039 | val loss: 0.519 | train acc: 99.2% | val acc: 86.2% | current lr: [0.001]
Epoch 40/50 | train loss: 0.053 | val loss: 1.173 | train acc: 98.7% | val acc: 71.4% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 41/50 | train loss: 0.029 | val loss: 1.933 | train acc: 99.4% | val acc: 62.7% | current lr: [0.001]
Epoch 42/50 | train loss: 0.025 | val loss: 0.311 | train acc: 99.6% | val acc: 90.4% | current lr: [0.001]
Epoch 43/50 | train loss: 0.104 | val loss: 4.276 | train acc: 97.0% | val acc: 42.8% | current lr: [0.001]
Epoch 44/50 | train loss: 0.091 | val loss: 1.061 | train acc: 97.2% | val acc: 73.5% | current lr: [0.001]
Epoch 45/50 | train loss: 0.037 | val loss: 0.604 | train acc: 99.0% | val acc: 82.9% | current lr: [0.001]
Epoch 46/50 | train loss: 0.025 | val loss: 0.521 | train acc: 99.5% | val acc: 87.3% | current lr: [0.001]
Epoch 47/50 | train loss: 0.018 | val loss: 0.407 | train acc: 99.6% | val acc: 88.6% | current lr: [0.001]
Epoch 48/50 | train loss: 0.012 | val loss: 0.236 | train acc: 99.9% | val acc: 92.5% | current lr: [0.001]
Epoch 49/50 | train loss: 0.009 | val loss: 0.245 | train acc: 99.9% | val acc: 93.2% | current lr: [0.001]
Epoch 50/50 | train loss: 0.003 | val loss: 0.243 | train acc: 100.0% | val acc: 93.4% | current lr: [0.001]=>Gap=6.6%
Training time: 3785.13 seconds
```

</details>

## Small CNN — StepLR

- **Best validation accuracy:** 90.8% (epoch 27)
- **Final validation accuracy:** 90.4%
- **Training time:** 3778.79 seconds (62.98 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.596 | val loss: 1.612 | train acc: 43.1% | val acc: 43.3% | current lr: [0.001]
Epoch 02/50 | train loss: 1.266 | val loss: 1.609 | train acc: 56.5% | val acc: 43.7% | current lr: [0.001]
Epoch 03/50 | train loss: 1.082 | val loss: 1.095 | train acc: 62.0% | val acc: 60.3% | current lr: [0.001]
Epoch 04/50 | train loss: 0.945 | val loss: 1.361 | train acc: 66.1% | val acc: 51.6% | current lr: [0.001]
Epoch 05/50 | train loss: 0.807 | val loss: 2.192 | train acc: 72.0% | val acc: 40.6% | current lr: [0.001]
Epoch 06/50 | train loss: 0.696 | val loss: 1.133 | train acc: 76.3% | val acc: 67.2% | current lr: [0.001]
Epoch 07/50 | train loss: 0.615 | val loss: 2.080 | train acc: 79.6% | val acc: 41.0% | current lr: [0.001]
Epoch 08/50 | train loss: 0.571 | val loss: 0.857 | train acc: 81.5% | val acc: 72.9% | current lr: [0.001]
Epoch 09/50 | train loss: 0.484 | val loss: 0.524 | train acc: 83.9% | val acc: 82.6% | current lr: [0.001]
Epoch 10/50 | train loss: 0.405 | val loss: 1.204 | train acc: 86.9% | val acc: 65.0% | current lr: [0.001]
Cooling down... 001 seconds remaining

Epoch 11/50 | train loss: 0.315 | val loss: 0.379 | train acc: 90.2% | val acc: 88.9% | current lr: [0.0001]
Epoch 12/50 | train loss: 0.274 | val loss: 0.380 | train acc: 92.4% | val acc: 88.0% | current lr: [0.0001]
Epoch 13/50 | train loss: 0.256 | val loss: 0.352 | train acc: 92.9% | val acc: 90.0% | current lr: [0.0001]
Epoch 14/50 | train loss: 0.248 | val loss: 0.367 | train acc: 92.6% | val acc: 88.9% | current lr: [0.0001]
Epoch 15/50 | train loss: 0.239 | val loss: 0.360 | train acc: 93.1% | val acc: 88.9% | current lr: [0.0001]
Epoch 16/50 | train loss: 0.233 | val loss: 0.389 | train acc: 93.6% | val acc: 87.6% | current lr: [0.0001]
Epoch 17/50 | train loss: 0.223 | val loss: 0.355 | train acc: 93.5% | val acc: 88.8% | current lr: [0.0001]
Epoch 18/50 | train loss: 0.211 | val loss: 0.358 | train acc: 94.1% | val acc: 87.3% | current lr: [0.0001]
Epoch 19/50 | train loss: 0.207 | val loss: 0.349 | train acc: 94.4% | val acc: 90.0% | current lr: [0.0001]
Epoch 20/50 | train loss: 0.201 | val loss: 0.388 | train acc: 94.6% | val acc: 87.3% | current lr: [0.0001]
Cooling down... 001 seconds remaining

Epoch 21/50 | train loss: 0.188 | val loss: 0.322 | train acc: 95.5% | val acc: 89.7% | current lr: [1e-05]
Epoch 22/50 | train loss: 0.184 | val loss: 0.322 | train acc: 95.3% | val acc: 89.8% | current lr: [1e-05]
Epoch 23/50 | train loss: 0.176 | val loss: 0.319 | train acc: 96.1% | val acc: 90.6% | current lr: [1e-05]
Epoch 24/50 | train loss: 0.180 | val loss: 0.319 | train acc: 95.9% | val acc: 90.6% | current lr: [1e-05]
Epoch 25/50 | train loss: 0.181 | val loss: 0.320 | train acc: 95.3% | val acc: 90.2% | current lr: [1e-05]
Epoch 26/50 | train loss: 0.181 | val loss: 0.313 | train acc: 95.8% | val acc: 90.4% | current lr: [1e-05]
Epoch 27/50 | train loss: 0.183 | val loss: 0.324 | train acc: 95.5% | val acc: 90.8% | current lr: [1e-05]
Epoch 28/50 | train loss: 0.178 | val loss: 0.318 | train acc: 95.5% | val acc: 89.4% | current lr: [1e-05]
Epoch 29/50 | train loss: 0.177 | val loss: 0.317 | train acc: 95.9% | val acc: 90.1% | current lr: [1e-05]
Epoch 30/50 | train loss: 0.173 | val loss: 0.316 | train acc: 95.6% | val acc: 90.4% | current lr: [1e-05]
Cooling down... 001 seconds remaining

Epoch 31/50 | train loss: 0.165 | val loss: 0.314 | train acc: 96.3% | val acc: 90.5% | current lr: [1.0000000000000002e-06]
Epoch 32/50 | train loss: 0.172 | val loss: 0.320 | train acc: 95.9% | val acc: 90.8% | current lr: [1.0000000000000002e-06]
Epoch 33/50 | train loss: 0.173 | val loss: 0.321 | train acc: 95.7% | val acc: 90.8% | current lr: [1.0000000000000002e-06]
Epoch 34/50 | train loss: 0.170 | val loss: 0.320 | train acc: 95.7% | val acc: 90.5% | current lr: [1.0000000000000002e-06]
Epoch 35/50 | train loss: 0.174 | val loss: 0.312 | train acc: 95.8% | val acc: 90.8% | current lr: [1.0000000000000002e-06]
Epoch 36/50 | train loss: 0.169 | val loss: 0.313 | train acc: 95.8% | val acc: 90.2% | current lr: [1.0000000000000002e-06]
Epoch 37/50 | train loss: 0.173 | val loss: 0.311 | train acc: 95.5% | val acc: 90.5% | current lr: [1.0000000000000002e-06]
Epoch 38/50 | train loss: 0.173 | val loss: 0.321 | train acc: 95.6% | val acc: 90.1% | current lr: [1.0000000000000002e-06]
Epoch 39/50 | train loss: 0.178 | val loss: 0.313 | train acc: 95.3% | val acc: 90.5% | current lr: [1.0000000000000002e-06]
✅Epoch 40/50 | train loss: 0.172 | val loss: 0.313 | train acc: 95.7% | val acc: 90.8% | current lr: [1.0000000000000002e-06]=>GAp = 5.9%
Cooling down... 001 seconds remaining

Epoch 41/50 | train loss: 0.169 | val loss: 0.314 | train acc: 96.0% | val acc: 90.5% | current lr: [1.0000000000000002e-07]
Epoch 42/50 | train loss: 0.173 | val loss: 0.319 | train acc: 95.7% | val acc: 90.1% | current lr: [1.0000000000000002e-07]
Epoch 43/50 | train loss: 0.175 | val loss: 0.317 | train acc: 96.0% | val acc: 90.4% | current lr: [1.0000000000000002e-07]
Epoch 44/50 | train loss: 0.169 | val loss: 0.318 | train acc: 95.9% | val acc: 90.6% | current lr: [1.0000000000000002e-07]
Epoch 45/50 | train loss: 0.172 | val loss: 0.311 | train acc: 95.8% | val acc: 90.4% | current lr: [1.0000000000000002e-07]
Epoch 46/50 | train loss: 0.168 | val loss: 0.313 | train acc: 95.9% | val acc: 90.4% | current lr: [1.0000000000000002e-07]
Epoch 47/50 | train loss: 0.169 | val loss: 0.314 | train acc: 96.0% | val acc: 90.1% | current lr: [1.0000000000000002e-07]
Epoch 48/50 | train loss: 0.171 | val loss: 0.315 | train acc: 95.8% | val acc: 90.1% | current lr: [1.0000000000000002e-07]
Epoch 49/50 | train loss: 0.169 | val loss: 0.312 | train acc: 95.9% | val acc: 90.6% | current lr: [1.0000000000000002e-07]
Epoch 50/50 | train loss: 0.169 | val loss: 0.322 | train acc: 95.9% | val acc: 90.4% | current lr: [1.0000000000000002e-07]
Training time: 3778.79 seconds
```

</details>

## Small CNN — ReduceLROnPlateau

- **Best validation accuracy:** 92.5% (epoch 23)
- **Final validation accuracy:** 91.6%
- **Training time:** 2580.05 seconds (43.00 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.596 | val loss: 1.612 | train acc: 43.1% | val acc: 43.3% | current lr: [0.001]
Epoch 02/50 | train loss: 1.266 | val loss: 1.609 | train acc: 56.5% | val acc: 43.7% | current lr: [0.001]
Epoch 03/50 | train loss: 1.082 | val loss: 1.095 | train acc: 62.0% | val acc: 60.3% | current lr: [0.001]
Epoch 04/50 | train loss: 0.945 | val loss: 1.361 | train acc: 66.1% | val acc: 51.6% | current lr: [0.001]
Epoch 05/50 | train loss: 0.807 | val loss: 2.192 | train acc: 72.0% | val acc: 40.6% | current lr: [0.001]
Epoch 06/50 | train loss: 0.696 | val loss: 1.133 | train acc: 76.3% | val acc: 67.2% | current lr: [0.001]
Epoch 07/50 | train loss: 0.615 | val loss: 2.080 | train acc: 79.6% | val acc: 41.0% | current lr: [0.001]
Epoch 08/50 | train loss: 0.571 | val loss: 0.857 | train acc: 81.5% | val acc: 72.9% | current lr: [0.001]
Epoch 09/50 | train loss: 0.484 | val loss: 0.524 | train acc: 83.9% | val acc: 82.6% | current lr: [0.001]
Epoch 10/50 | train loss: 0.405 | val loss: 1.204 | train acc: 86.9% | val acc: 65.0% | current lr: [0.001]
Epoch 11/50 | train loss: 0.408 | val loss: 0.510 | train acc: 86.5% | val acc: 83.4% | current lr: [0.001]
Epoch 12/50 | train loss: 0.373 | val loss: 1.127 | train acc: 87.3% | val acc: 65.4% | current lr: [0.001]
Epoch 13/50 | train loss: 0.337 | val loss: 0.695 | train acc: 89.2% | val acc: 76.2% | current lr: [0.001]
Epoch 14/50 | train loss: 0.305 | val loss: 0.583 | train acc: 90.5% | val acc: 80.7% | current lr: [0.001]
Epoch 15/50 | train loss: 0.266 | val loss: 0.528 | train acc: 91.6% | val acc: 82.6% | current lr: [0.001]
Epoch 16/50 | train loss: 0.275 | val loss: 1.009 | train acc: 91.7% | val acc: 69.8% | current lr: [0.001]
Epoch 17/50 | train loss: 0.210 | val loss: 1.284 | train acc: 93.3% | val acc: 62.3% | current lr: [0.001]
Epoch 18/50 | train loss: 0.155 | val loss: 0.271 | train acc: 95.8% | val acc: 90.9% | current lr: [0.0001]
Epoch 19/50 | train loss: 0.127 | val loss: 0.264 | train acc: 96.8% | val acc: 92.0% | current lr: [0.0001]
Epoch 20/50 | train loss: 0.115 | val loss: 0.282 | train acc: 96.8% | val acc: 90.1% | current lr: [0.0001]
Epoch 21/50 | train loss: 0.106 | val loss: 0.264 | train acc: 97.6% | val acc: 91.3% | current lr: [0.0001]
Epoch 22/50 | train loss: 0.102 | val loss: 0.263 | train acc: 97.7% | val acc: 91.8% | current lr: [0.0001]
Epoch 23/50 | train loss: 0.093 | val loss: 0.250 | train acc: 98.2% | val acc: 92.5% | current lr: [0.0001]=>Gap=5.7%
Epoch 24/50 | train loss: 0.092 | val loss: 0.271 | train acc: 98.0% | val acc: 91.4% | current lr: [0.0001]
Epoch 25/50 | train loss: 0.091 | val loss: 0.252 | train acc: 98.1% | val acc: 91.2% | current lr: [0.0001]
Epoch 26/50 | train loss: 0.088 | val loss: 0.267 | train acc: 98.2% | val acc: 90.4% | current lr: [0.0001]
Epoch 27/50 | train loss: 0.085 | val loss: 0.286 | train acc: 98.3% | val acc: 90.8% | current lr: [0.0001]
Epoch 28/50 | train loss: 0.080 | val loss: 0.294 | train acc: 98.5% | val acc: 90.9% | current lr: [0.0001]
Epoch 29/50 | train loss: 0.076 | val loss: 0.259 | train acc: 98.8% | val acc: 90.9% | current lr: [0.0001]
Epoch 30/50 | train loss: 0.068 | val loss: 0.244 | train acc: 98.9% | val acc: 92.1% | current lr: [1e-05]
Epoch 31/50 | train loss: 0.063 | val loss: 0.238 | train acc: 99.0% | val acc: 92.0% | current lr: [1e-05]
Epoch 32/50 | train loss: 0.066 | val loss: 0.249 | train acc: 98.9% | val acc: 91.4% | current lr: [1e-05]
Epoch 33/50 | train loss: 0.065 | val loss: 0.246 | train acc: 99.2% | val acc: 91.6% | current lr: [1e-05]
Epoch 34/50 | train loss: 0.063 | val loss: 0.245 | train acc: 99.2% | val acc: 91.6% | current lr: [1e-05]
Epoch 35/50 | train loss: 0.067 | val loss: 0.240 | train acc: 99.0% | val acc: 92.0% | current lr: [1e-05]
Epoch 36/50 | train loss: 0.061 | val loss: 0.243 | train acc: 99.1% | val acc: 92.1% | current lr: [1e-05]
Epoch 37/50 | train loss: 0.065 | val loss: 0.237 | train acc: 99.0% | val acc: 92.0% | current lr: [1e-05]
Epoch 38/50 | train loss: 0.064 | val loss: 0.248 | train acc: 99.2% | val acc: 91.4% | current lr: [1e-05]
Epoch 39/50 | train loss: 0.066 | val loss: 0.240 | train acc: 99.0% | val acc: 91.8% | current lr: [1e-05]
Epoch 40/50 | train loss: 0.063 | val loss: 0.241 | train acc: 99.1% | val acc: 92.5% | current lr: [1e-05]
Epoch 41/50 | train loss: 0.061 | val loss: 0.243 | train acc: 99.2% | val acc: 91.4% | current lr: [1e-05]
Epoch 42/50 | train loss: 0.060 | val loss: 0.250 | train acc: 99.4% | val acc: 91.4% | current lr: [1e-05]
Epoch 43/50 | train loss: 0.063 | val loss: 0.248 | train acc: 99.1% | val acc: 92.0% | current lr: [1e-05]
Epoch 44/50 | train loss: 0.059 | val loss: 0.249 | train acc: 99.3% | val acc: 91.6% | current lr: [1.0000000000000002e-06]
Epoch 45/50 | train loss: 0.060 | val loss: 0.239 | train acc: 99.1% | val acc: 92.2% | current lr: [1.0000000000000002e-06]
Epoch 46/50 | train loss: 0.058 | val loss: 0.244 | train acc: 99.4% | val acc: 91.8% | current lr: [1.0000000000000002e-06]
Epoch 47/50 | train loss: 0.058 | val loss: 0.242 | train acc: 99.3% | val acc: 91.8% | current lr: [1.0000000000000002e-06]
Epoch 48/50 | train loss: 0.060 | val loss: 0.243 | train acc: 99.1% | val acc: 91.7% | current lr: [1.0000000000000002e-06]
Epoch 49/50 | train loss: 0.058 | val loss: 0.241 | train acc: 99.3% | val acc: 91.8% | current lr: [1.0000000000000002e-06]
Epoch 50/50 | train loss: 0.057 | val loss: 0.250 | train acc: 99.5% | val acc: 91.6% | current lr: [1.0000000000000002e-06]
Training time: 2580.05 seconds
```

</details>

## Small CNN — Standard Imbalanced Loader

- **Best validation accuracy:** 89.6% (epoch 40)
- **Final validation accuracy:** 84.9%
- **Training time:** 1937.13 seconds (32.29 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.552 | val loss: 1.785 | train acc: 46.2% | val acc: 40.2% | current lr: [0.001]
Epoch 02/50 | train loss: 1.286 | val loss: 1.566 | train acc: 56.3% | val acc: 43.0% | current lr: [0.001]
Epoch 03/50 | train loss: 1.138 | val loss: 1.638 | train acc: 59.4% | val acc: 37.4% | current lr: [0.001]
Epoch 04/50 | train loss: 1.014 | val loss: 1.759 | train acc: 64.1% | val acc: 41.4% | current lr: [0.001]
Epoch 05/50 | train loss: 0.945 | val loss: 1.971 | train acc: 66.1% | val acc: 44.1% | current lr: [0.001]
Epoch 06/50 | train loss: 0.862 | val loss: 1.224 | train acc: 69.2% | val acc: 59.4% | current lr: [0.001]
Epoch 07/50 | train loss: 0.766 | val loss: 1.442 | train acc: 73.0% | val acc: 51.9% | current lr: [0.001]
Epoch 08/50 | train loss: 0.687 | val loss: 1.419 | train acc: 75.9% | val acc: 51.1% | current lr: [0.001]
Epoch 09/50 | train loss: 0.632 | val loss: 0.936 | train acc: 79.0% | val acc: 67.1% | current lr: [0.001]
Epoch 10/50 | train loss: 0.547 | val loss: 1.444 | train acc: 82.4% | val acc: 49.3% | current lr: [0.001]
Epoch 11/50 | train loss: 0.484 | val loss: 1.563 | train acc: 85.0% | val acc: 55.7% | current lr: [0.001]
Epoch 12/50 | train loss: 0.457 | val loss: 1.501 | train acc: 85.0% | val acc: 58.7% | current lr: [0.001]
Epoch 13/50 | train loss: 0.444 | val loss: 0.667 | train acc: 85.6% | val acc: 78.1% | current lr: [0.001]
Epoch 14/50 | train loss: 0.349 | val loss: 1.257 | train acc: 89.7% | val acc: 61.9% | current lr: [0.001]
Epoch 15/50 | train loss: 0.334 | val loss: 0.771 | train acc: 89.6% | val acc: 75.5% | current lr: [0.001]
Epoch 16/50 | train loss: 0.328 | val loss: 0.552 | train acc: 89.7% | val acc: 81.8% | current lr: [0.001]
Epoch 17/50 | train loss: 0.303 | val loss: 2.028 | train acc: 91.0% | val acc: 56.3% | current lr: [0.001]
Epoch 18/50 | train loss: 0.281 | val loss: 1.453 | train acc: 91.5% | val acc: 66.4% | current lr: [0.001]
Epoch 19/50 | train loss: 0.268 | val loss: 1.008 | train acc: 91.8% | val acc: 71.4% | current lr: [0.001]
Epoch 20/50 | train loss: 0.215 | val loss: 0.657 | train acc: 93.3% | val acc: 79.4% | current lr: [0.001]
Epoch 21/50 | train loss: 0.205 | val loss: 0.625 | train acc: 94.0% | val acc: 79.4% | current lr: [0.001]
Epoch 22/50 | train loss: 0.178 | val loss: 0.988 | train acc: 94.9% | val acc: 77.3% | current lr: [0.001]
Epoch 23/50 | train loss: 0.158 | val loss: 0.701 | train acc: 95.3% | val acc: 79.8% | current lr: [0.001]
Epoch 24/50 | train loss: 0.150 | val loss: 0.872 | train acc: 95.8% | val acc: 75.4% | current lr: [0.001]
Epoch 25/50 | train loss: 0.124 | val loss: 0.376 | train acc: 96.7% | val acc: 87.4% | current lr: [0.001]
Epoch 26/50 | train loss: 0.100 | val loss: 0.880 | train acc: 97.6% | val acc: 74.7% | current lr: [0.001]
Epoch 27/50 | train loss: 0.100 | val loss: 0.843 | train acc: 97.6% | val acc: 74.1% | current lr: [0.001]
Epoch 28/50 | train loss: 0.105 | val loss: 0.418 | train acc: 96.9% | val acc: 87.2% | current lr: [0.001]
Epoch 29/50 | train loss: 0.106 | val loss: 1.118 | train acc: 96.7% | val acc: 70.1% | current lr: [0.001]
Epoch 30/50 | train loss: 0.080 | val loss: 0.625 | train acc: 98.1% | val acc: 81.7% | current lr: [0.001]
Epoch 31/50 | train loss: 0.083 | val loss: 2.046 | train acc: 97.6% | val acc: 54.4% | current lr: [0.001]
Epoch 32/50 | train loss: 0.100 | val loss: 1.028 | train acc: 97.1% | val acc: 72.9% | current lr: [0.001]
Epoch 33/50 | train loss: 0.072 | val loss: 1.242 | train acc: 98.2% | val acc: 75.5% | current lr: [0.001]
Epoch 34/50 | train loss: 0.075 | val loss: 0.573 | train acc: 98.1% | val acc: 85.3% | current lr: [0.001]
Epoch 35/50 | train loss: 0.084 | val loss: 0.982 | train acc: 97.9% | val acc: 77.3% | current lr: [0.001]
Epoch 36/50 | train loss: 0.036 | val loss: 0.648 | train acc: 99.4% | val acc: 82.2% | current lr: [0.001]
Epoch 37/50 | train loss: 0.040 | val loss: 0.438 | train acc: 99.0% | val acc: 87.3% | current lr: [0.001]
Epoch 38/50 | train loss: 0.038 | val loss: 0.692 | train acc: 99.4% | val acc: 82.6% | current lr: [0.001]
Epoch 39/50 | train loss: 0.049 | val loss: 0.903 | train acc: 98.7% | val acc: 79.3% | current lr: [0.001]
✅Epoch 40/50 | train loss: 0.031 | val loss: 0.356 | train acc: 99.4% | val acc: 89.6% | current lr: [0.001] => Gap=9.8%
Epoch 41/50 | train loss: 0.031 | val loss: 0.482 | train acc: 99.4% | val acc: 85.8% | current lr: [0.001]
Epoch 42/50 | train loss: 0.030 | val loss: 0.519 | train acc: 99.4% | val acc: 86.5% | current lr: [0.001]
Epoch 43/50 | train loss: 0.023 | val loss: 0.465 | train acc: 99.6% | val acc: 87.2% | current lr: [0.001]
Epoch 44/50 | train loss: 0.031 | val loss: 0.579 | train acc: 99.2% | val acc: 83.8% | current lr: [0.001]
Epoch 45/50 | train loss: 0.051 | val loss: 1.530 | train acc: 98.2% | val acc: 67.0% | current lr: [0.001]
Epoch 46/50 | train loss: 0.104 | val loss: 1.392 | train acc: 97.2% | val acc: 72.3% | current lr: [0.001]
Epoch 47/50 | train loss: 0.107 | val loss: 1.436 | train acc: 96.7% | val acc: 67.8% | current lr: [0.001]
Epoch 48/50 | train loss: 0.052 | val loss: 0.654 | train acc: 98.9% | val acc: 81.4% | current lr: [0.001]
Epoch 49/50 | train loss: 0.023 | val loss: 0.424 | train acc: 99.7% | val acc: 89.3% | current lr: [0.001]
Epoch 50/50 | train loss: 0.017 | val loss: 0.557 | train acc: 99.6% | val acc: 84.9% | current lr: [0.001]
Training time: 1937.13 seconds
```

</details>

## Small CNN — Balanced Imbalanced Loader

- **Best validation accuracy:** 88.5% (epoch 46)
- **Final validation accuracy:** 88.5%
- **Training time:** 1938.22 seconds (32.30 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.644 | val loss: 1.613 | train acc: 39.3% | val acc: 39.8% | current lr: [0.001]
Epoch 02/50 | train loss: 1.321 | val loss: 1.887 | train acc: 51.5% | val acc: 38.5% | current lr: [0.001]
Epoch 03/50 | train loss: 1.101 | val loss: 1.752 | train acc: 58.9% | val acc: 45.2% | current lr: [0.001]
Epoch 04/50 | train loss: 0.930 | val loss: 2.307 | train acc: 66.5% | val acc: 39.8% | current lr: [0.001]
Epoch 05/50 | train loss: 0.797 | val loss: 2.120 | train acc: 71.1% | val acc: 42.5% | current lr: [0.001]
Epoch 06/50 | train loss: 0.689 | val loss: 1.765 | train acc: 76.0% | val acc: 46.4% | current lr: [0.001]
Epoch 07/50 | train loss: 0.572 | val loss: 1.605 | train acc: 80.6% | val acc: 54.9% | current lr: [0.001]
Epoch 08/50 | train loss: 0.477 | val loss: 1.772 | train acc: 85.0% | val acc: 53.9% | current lr: [0.001]
Epoch 09/50 | train loss: 0.375 | val loss: 2.215 | train acc: 88.7% | val acc: 42.6% | current lr: [0.001]
Epoch 10/50 | train loss: 0.305 | val loss: 1.717 | train acc: 90.4% | val acc: 55.9% | current lr: [0.001]
Epoch 11/50 | train loss: 0.236 | val loss: 0.986 | train acc: 93.2% | val acc: 68.9% | current lr: [0.001]
Epoch 12/50 | train loss: 0.212 | val loss: 1.370 | train acc: 93.4% | val acc: 63.9% | current lr: [0.001]
Epoch 13/50 | train loss: 0.191 | val loss: 0.958 | train acc: 94.3% | val acc: 74.3% | current lr: [0.001]
Epoch 14/50 | train loss: 0.152 | val loss: 0.910 | train acc: 95.6% | val acc: 71.8% | current lr: [0.001]
Epoch 15/50 | train loss: 0.164 | val loss: 1.092 | train acc: 94.9% | val acc: 70.3% | current lr: [0.001]
Epoch 16/50 | train loss: 0.115 | val loss: 0.807 | train acc: 97.2% | val acc: 76.3% | current lr: [0.001]
Epoch 17/50 | train loss: 0.123 | val loss: 0.861 | train acc: 96.7% | val acc: 75.4% | current lr: [0.001]
Epoch 18/50 | train loss: 0.114 | val loss: 1.747 | train acc: 97.2% | val acc: 58.4% | current lr: [0.001]
Epoch 19/50 | train loss: 0.099 | val loss: 0.841 | train acc: 97.4% | val acc: 77.4% | current lr: [0.001]
Epoch 20/50 | train loss: 0.049 | val loss: 0.688 | train acc: 99.2% | val acc: 81.8% | current lr: [0.001]
Epoch 21/50 | train loss: 0.034 | val loss: 0.886 | train acc: 99.6% | val acc: 74.5% | current lr: [0.001]
Epoch 22/50 | train loss: 0.030 | val loss: 0.782 | train acc: 99.5% | val acc: 79.1% | current lr: [0.001]
Epoch 23/50 | train loss: 0.020 | val loss: 0.617 | train acc: 99.8% | val acc: 82.1% | current lr: [0.001]
Epoch 24/50 | train loss: 0.022 | val loss: 1.174 | train acc: 99.6% | val acc: 71.4% | current lr: [0.001]
Epoch 25/50 | train loss: 0.018 | val loss: 0.592 | train acc: 99.9% | val acc: 83.2% | current lr: [0.001]
Epoch 26/50 | train loss: 0.007 | val loss: 0.439 | train acc: 100.0% | val acc: 87.2% | current lr: [0.001]
Epoch 27/50 | train loss: 0.004 | val loss: 0.440 | train acc: 100.0% | val acc: 87.4% | current lr: [0.001]
Epoch 28/50 | train loss: 0.003 | val loss: 0.437 | train acc: 100.0% | val acc: 87.6% | current lr: [0.001]
Epoch 29/50 | train loss: 0.003 | val loss: 0.449 | train acc: 100.0% | val acc: 87.7% | current lr: [0.001]
Epoch 30/50 | train loss: 0.002 | val loss: 0.452 | train acc: 100.0% | val acc: 87.6% | current lr: [0.001]
Epoch 31/50 | train loss: 0.002 | val loss: 0.452 | train acc: 100.0% | val acc: 87.7% | current lr: [0.001]
Epoch 32/50 | train loss: 0.001 | val loss: 0.450 | train acc: 100.0% | val acc: 88.0% | current lr: [0.001]
Epoch 33/50 | train loss: 0.001 | val loss: 0.450 | train acc: 100.0% | val acc: 88.0% | current lr: [0.001]
Epoch 34/50 | train loss: 0.001 | val loss: 0.450 | train acc: 100.0% | val acc: 88.1% | current lr: [0.001]
Epoch 35/50 | train loss: 0.001 | val loss: 0.451 | train acc: 100.0% | val acc: 88.1% | current lr: [0.001]
Epoch 36/50 | train loss: 0.001 | val loss: 0.451 | train acc: 100.0% | val acc: 88.1% | current lr: [0.001]
Epoch 37/50 | train loss: 0.001 | val loss: 0.451 | train acc: 100.0% | val acc: 88.4% | current lr: [0.001]
Epoch 38/50 | train loss: 0.001 | val loss: 0.452 | train acc: 100.0% | val acc: 88.4% | current lr: [0.001]
Epoch 39/50 | train loss: 0.001 | val loss: 0.452 | train acc: 100.0% | val acc: 88.2% | current lr: [0.001]
Epoch 40/50 | train loss: 0.001 | val loss: 0.453 | train acc: 100.0% | val acc: 88.1% | current lr: [0.001]
Epoch 41/50 | train loss: 0.001 | val loss: 0.454 | train acc: 100.0% | val acc: 88.1% | current lr: [0.001]
Epoch 42/50 | train loss: 0.001 | val loss: 0.454 | train acc: 100.0% | val acc: 88.2% | current lr: [0.001]
Epoch 43/50 | train loss: 0.001 | val loss: 0.455 | train acc: 100.0% | val acc: 88.2% | current lr: [0.001]
Epoch 44/50 | train loss: 0.001 | val loss: 0.455 | train acc: 100.0% | val acc: 88.4% | current lr: [0.001]
Epoch 45/50 | train loss: 0.001 | val loss: 0.456 | train acc: 100.0% | val acc: 88.4% | current lr: [0.001]
Epoch 46/50 | train loss: 0.001 | val loss: 0.456 | train acc: 100.0% | val acc: 88.5% | current lr: [0.001]=>Gap=11.5%
Epoch 47/50 | train loss: 0.001 | val loss: 0.457 | train acc: 100.0% | val acc: 88.4% | current lr: [0.001]
Epoch 48/50 | train loss: 0.000 | val loss: 0.457 | train acc: 100.0% | val acc: 88.2% | current lr: [0.001]
Epoch 49/50 | train loss: 0.000 | val loss: 0.458 | train acc: 100.0% | val acc: 88.4% | current lr: [0.001]
Epoch 50/50 | train loss: 0.000 | val loss: 0.458 | train acc: 100.0% | val acc: 88.5% | current lr: [0.001]
Training time: 1938.22 seconds
```

</details>

## Small CNN — BCE

- **Best validation accuracy:** 93.4% (epoch 36)
- **Final validation accuracy:** 84.5%
- **Training time:** 2567.16 seconds (42.79 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 0.347 | val loss: 0.305 | train acc: 41.6% | val acc: 42.8% | current lr: 0.001
Epoch 02/50 | train loss: 0.268 | val loss: 0.326 | train acc: 53.6% | val acc: 35.7% | current lr: 0.001
Epoch 03/50 | train loss: 0.241 | val loss: 0.274 | train acc: 60.0% | val acc: 50.1% | current lr: 0.001
Epoch 04/50 | train loss: 0.220 | val loss: 0.254 | train acc: 64.3% | val acc: 59.5% | current lr: 0.001
Epoch 05/50 | train loss: 0.198 | val loss: 0.194 | train acc: 70.0% | val acc: 70.2% | current lr: 0.001
Epoch 06/50 | train loss: 0.180 | val loss: 0.296 | train acc: 73.1% | val acc: 48.8% | current lr: 0.001
Epoch 07/50 | train loss: 0.157 | val loss: 0.270 | train acc: 77.7% | val acc: 59.0% | current lr: 0.001
Epoch 08/50 | train loss: 0.144 | val loss: 0.179 | train acc: 80.1% | val acc: 71.7% | current lr: 0.001
Epoch 09/50 | train loss: 0.128 | val loss: 0.169 | train acc: 82.2% | val acc: 73.3% | current lr: 0.001
Epoch 10/50 | train loss: 0.114 | val loss: 0.281 | train acc: 85.1% | val acc: 60.8% | current lr: 0.001
Epoch 11/50 | train loss: 0.108 | val loss: 0.156 | train acc: 85.7% | val acc: 75.4% | current lr: 0.001
Epoch 12/50 | train loss: 0.095 | val loss: 0.185 | train acc: 88.0% | val acc: 76.2% | current lr: 0.001
Epoch 13/50 | train loss: 0.091 | val loss: 0.134 | train acc: 88.7% | val acc: 79.5% | current lr: 0.001
Epoch 14/50 | train loss: 0.084 | val loss: 0.188 | train acc: 90.2% | val acc: 74.3% | current lr: 0.001
Epoch 15/50 | train loss: 0.077 | val loss: 0.323 | train acc: 91.6% | val acc: 65.0% | current lr: 0.001
Epoch 16/50 | train loss: 0.070 | val loss: 0.128 | train acc: 93.0% | val acc: 83.7% | current lr: 0.001
Epoch 17/50 | train loss: 0.062 | val loss: 0.285 | train acc: 93.6% | val acc: 66.3% | current lr: 0.001
Epoch 18/50 | train loss: 0.058 | val loss: 0.174 | train acc: 93.7% | val acc: 77.4% | current lr: 0.001
Epoch 19/50 | train loss: 0.055 | val loss: 0.149 | train acc: 94.7% | val acc: 79.3% | current lr: 0.001
Epoch 20/50 | train loss: 0.052 | val loss: 0.195 | train acc: 94.9% | val acc: 77.8% | current lr: 0.001
Epoch 21/50 | train loss: 0.050 | val loss: 0.331 | train acc: 95.1% | val acc: 59.2% | current lr: 0.001
Epoch 22/50 | train loss: 0.043 | val loss: 0.089 | train acc: 96.2% | val acc: 89.7% | current lr: 0.001
Epoch 23/50 | train loss: 0.036 | val loss: 0.149 | train acc: 96.8% | val acc: 80.9% | current lr: 0.001
Epoch 24/50 | train loss: 0.034 | val loss: 0.131 | train acc: 97.1% | val acc: 83.3% | current lr: 0.001
Epoch 25/50 | train loss: 0.032 | val loss: 0.113 | train acc: 97.6% | val acc: 86.0% | current lr: 0.001
Epoch 26/50 | train loss: 0.028 | val loss: 0.111 | train acc: 98.2% | val acc: 86.2% | current lr: 0.001
Epoch 27/50 | train loss: 0.029 | val loss: 0.170 | train acc: 97.6% | val acc: 83.6% | current lr: 0.001
Epoch 28/50 | train loss: 0.023 | val loss: 0.122 | train acc: 98.6% | val acc: 85.7% | current lr: 0.001
Epoch 29/50 | train loss: 0.022 | val loss: 0.141 | train acc: 98.6% | val acc: 84.8% | current lr: 0.001
Epoch 30/50 | train loss: 0.020 | val loss: 0.167 | train acc: 99.2% | val acc: 82.4% | current lr: 0.001
Epoch 31/50 | train loss: 0.015 | val loss: 0.106 | train acc: 99.3% | val acc: 88.5% | current lr: 0.001
Epoch 32/50 | train loss: 0.017 | val loss: 0.402 | train acc: 99.1% | val acc: 67.8% | current lr: 0.001
Epoch 33/50 | train loss: 0.015 | val loss: 0.098 | train acc: 99.5% | val acc: 87.4% | current lr: 0.001
Epoch 34/50 | train loss: 0.015 | val loss: 0.089 | train acc: 99.3% | val acc: 89.8% | current lr: 0.001
Epoch 35/50 | train loss: 0.014 | val loss: 0.091 | train acc: 99.5% | val acc: 90.2% | current lr: 0.001
Epoch 36/50 | train loss: 0.010 | val loss: 0.069 | train acc: 99.7% | val acc: 93.4% | current lr: 0.001=>Gap=6.3%
Epoch 37/50 | train loss: 0.011 | val loss: 0.134 | train acc: 99.6% | val acc: 84.1% | current lr: 0.001
Epoch 38/50 | train loss: 0.016 | val loss: 0.178 | train acc: 99.3% | val acc: 80.2% | current lr: 0.001
Epoch 39/50 | train loss: 0.015 | val loss: 0.165 | train acc: 99.2% | val acc: 84.8% | current lr: 0.001
Epoch 40/50 | train loss: 0.015 | val loss: 0.113 | train acc: 99.2% | val acc: 86.4% | current lr: 0.001
Epoch 41/50 | train loss: 0.010 | val loss: 0.077 | train acc: 99.7% | val acc: 91.0% | current lr: 0.001
Epoch 42/50 | train loss: 0.007 | val loss: 0.073 | train acc: 99.8% | val acc: 91.4% | current lr: 0.001
Epoch 43/50 | train loss: 0.009 | val loss: 0.158 | train acc: 99.6% | val acc: 82.0% | current lr: 0.001
Epoch 44/50 | train loss: 0.008 | val loss: 0.147 | train acc: 99.7% | val acc: 85.6% | current lr: 0.001
Epoch 45/50 | train loss: 0.006 | val loss: 0.072 | train acc: 99.8% | val acc: 91.8% | current lr: 0.001
Epoch 46/50 | train loss: 0.004 | val loss: 0.069 | train acc: 99.9% | val acc: 93.2% | current lr: 0.001
Epoch 47/50 | train loss: 0.004 | val loss: 0.081 | train acc: 100.0% | val acc: 92.1% | current lr: 0.001
Epoch 48/50 | train loss: 0.002 | val loss: 0.073 | train acc: 100.0% | val acc: 92.5% | current lr: 0.001
Epoch 49/50 | train loss: 0.003 | val loss: 0.241 | train acc: 99.9% | val acc: 80.1% | current lr: 0.001
Epoch 50/50 | train loss: 0.014 | val loss: 0.199 | train acc: 99.3% | val acc: 84.5% | current lr: 0.001
Training time: 2567.16 seconds
```

</details>

## ResNet18 — Full Training

- **Best validation accuracy:** 91.0% (epoch 37)
- **Final validation accuracy:** 89.6%
- **Training time:** 756.63 seconds (12.61 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.379 | val loss: 0.888 | train acc: 55.9% | val acc: 74.2% | current lr: [0.001]
Epoch 02/50 | train loss: 0.764 | val loss: 0.677 | train acc: 79.4% | val acc: 79.5% | current lr: [0.001]
Epoch 03/50 | train loss: 0.596 | val loss: 0.551 | train acc: 84.2% | val acc: 85.0% | current lr: [0.001]
Epoch 04/50 | train loss: 0.501 | val loss: 0.485 | train acc: 86.2% | val acc: 86.9% | current lr: [0.001]
Epoch 05/50 | train loss: 0.447 | val loss: 0.456 | train acc: 87.9% | val acc: 86.8% | current lr: [0.001]
Epoch 06/50 | train loss: 0.413 | val loss: 0.427 | train acc: 88.4% | val acc: 87.6% | current lr: [0.001]
Epoch 07/50 | train loss: 0.378 | val loss: 0.414 | train acc: 89.0% | val acc: 87.7% | current lr: [0.001]
Epoch 08/50 | train loss: 0.350 | val loss: 0.391 | train acc: 90.5% | val acc: 88.6% | current lr: [0.001]
Epoch 09/50 | train loss: 0.338 | val loss: 0.382 | train acc: 89.8% | val acc: 88.5% | current lr: [0.001]
Epoch 10/50 | train loss: 0.326 | val loss: 0.373 | train acc: 90.4% | val acc: 88.6% | current lr: [0.001]
Epoch 11/50 | train loss: 0.303 | val loss: 0.359 | train acc: 91.7% | val acc: 89.3% | current lr: [0.001]
Epoch 12/50 | train loss: 0.297 | val loss: 0.362 | train acc: 91.0% | val acc: 89.0% | current lr: [0.001]
Epoch 13/50 | train loss: 0.296 | val loss: 0.353 | train acc: 91.4% | val acc: 89.4% | current lr: [0.001]
Epoch 14/50 | train loss: 0.278 | val loss: 0.351 | train acc: 92.0% | val acc: 89.3% | current lr: [0.001]
Epoch 15/50 | train loss: 0.273 | val loss: 0.348 | train acc: 91.9% | val acc: 88.9% | current lr: [0.001]
Epoch 16/50 | train loss: 0.259 | val loss: 0.348 | train acc: 92.2% | val acc: 89.2% | current lr: [0.001]
Epoch 17/50 | train loss: 0.245 | val loss: 0.333 | train acc: 92.9% | val acc: 90.5% | current lr: [0.001]
Epoch 18/50 | train loss: 0.250 | val loss: 0.332 | train acc: 92.5% | val acc: 90.1% | current lr: [0.001]
Epoch 19/50 | train loss: 0.234 | val loss: 0.330 | train acc: 93.3% | val acc: 90.1% | current lr: [0.001]
Epoch 20/50 | train loss: 0.227 | val loss: 0.327 | train acc: 93.2% | val acc: 90.0% | current lr: [0.001]
Epoch 21/50 | train loss: 0.218 | val loss: 0.333 | train acc: 93.5% | val acc: 90.1% | current lr: [0.001]
Epoch 22/50 | train loss: 0.225 | val loss: 0.342 | train acc: 93.0% | val acc: 89.3% | current lr: [0.001]
Epoch 23/50 | train loss: 0.216 | val loss: 0.339 | train acc: 92.9% | val acc: 90.1% | current lr: [0.001]
Epoch 24/50 | train loss: 0.212 | val loss: 0.325 | train acc: 93.8% | val acc: 90.9% | current lr: [0.001]
Epoch 25/50 | train loss: 0.209 | val loss: 0.321 | train acc: 94.0% | val acc: 90.8% | current lr: [0.001]
Epoch 26/50 | train loss: 0.202 | val loss: 0.338 | train acc: 94.2% | val acc: 89.0% | current lr: [0.001]
Epoch 27/50 | train loss: 0.203 | val loss: 0.323 | train acc: 93.8% | val acc: 90.5% | current lr: [0.001]
Epoch 28/50 | train loss: 0.198 | val loss: 0.336 | train acc: 94.0% | val acc: 89.3% | current lr: [0.001]
Epoch 29/50 | train loss: 0.190 | val loss: 0.335 | train acc: 94.4% | val acc: 89.3% | current lr: [0.001]
Epoch 30/50 | train loss: 0.186 | val loss: 0.327 | train acc: 94.1% | val acc: 90.2% | current lr: [0.001]
Epoch 31/50 | train loss: 0.180 | val loss: 0.342 | train acc: 94.9% | val acc: 88.9% | current lr: [0.001]
Epoch 32/50 | train loss: 0.183 | val loss: 0.334 | train acc: 94.3% | val acc: 89.8% | current lr: [0.001]
Epoch 33/50 | train loss: 0.170 | val loss: 0.329 | train acc: 94.7% | val acc: 89.7% | current lr: [0.001]
Epoch 34/50 | train loss: 0.183 | val loss: 0.323 | train acc: 94.4% | val acc: 90.4% | current lr: [0.001]
Epoch 35/50 | train loss: 0.173 | val loss: 0.331 | train acc: 94.9% | val acc: 90.1% | current lr: [0.001]
Epoch 36/50 | train loss: 0.164 | val loss: 0.326 | train acc: 95.3% | val acc: 90.8% | current lr: [0.001]
✅Epoch 37/50 | train loss: 0.169 | val loss: 0.316 | train acc: 95.0% | val acc: 91.0% | current lr: [0.001]=> Gap=4%
Epoch 38/50 | train loss: 0.169 | val loss: 0.323 | train acc: 95.1% | val acc: 90.8% | current lr: [0.001]
Epoch 39/50 | train loss: 0.165 | val loss: 0.317 | train acc: 94.8% | val acc: 91.0% | current lr: [0.001]
Epoch 40/50 | train loss: 0.166 | val loss: 0.320 | train acc: 94.9% | val acc: 90.2% | current lr: [0.001]
Epoch 41/50 | train loss: 0.155 | val loss: 0.334 | train acc: 95.2% | val acc: 89.8% | current lr: [0.001]
Epoch 42/50 | train loss: 0.149 | val loss: 0.339 | train acc: 95.8% | val acc: 89.3% | current lr: [0.001]
Epoch 43/50 | train loss: 0.150 | val loss: 0.330 | train acc: 95.8% | val acc: 89.7% | current lr: [0.001]
Epoch 44/50 | train loss: 0.150 | val loss: 0.325 | train acc: 95.6% | val acc: 90.2% | current lr: [0.001]
Epoch 45/50 | train loss: 0.152 | val loss: 0.357 | train acc: 95.0% | val acc: 88.4% | current lr: [0.001]
Epoch 46/50 | train loss: 0.133 | val loss: 0.327 | train acc: 96.4% | val acc: 89.7% | current lr: [0.001]
Epoch 47/50 | train loss: 0.137 | val loss: 0.324 | train acc: 95.7% | val acc: 90.5% | current lr: [0.001]
Epoch 48/50 | train loss: 0.141 | val loss: 0.331 | train acc: 96.0% | val acc: 90.1% | current lr: [0.001]
Epoch 49/50 | train loss: 0.127 | val loss: 0.339 | train acc: 96.9% | val acc: 89.4% | current lr: [0.001]
Epoch 50/50 | train loss: 0.131 | val loss: 0.341 | train acc: 96.0% | val acc: 89.6% | current lr: [0.001]
Training time: 756.63 seconds
```

</details>

## ResNet18 — Fine-Tuning

- **Best validation accuracy:** 96.1% (epoch 45)
- **Final validation accuracy:** 95.2%
- **Training time:** 824.28 seconds (13.74 minutes)

<details>
<summary>Full training log</summary>

```text
Trainable params: 8397832
Total params: 11180616
Epoch 01/50 | train loss: 0.209 | val loss: 0.283 | train acc: 92.9% | val acc: 92.2% | current lr: [0.0001, 0.001]
Epoch 02/50 | train loss: 0.043 | val loss: 0.237 | train acc: 98.6% | val acc: 93.7% | current lr: [0.0001, 0.001]
Epoch 03/50 | train loss: 0.016 | val loss: 0.238 | train acc: 99.5% | val acc: 94.7% | current lr: [0.0001, 0.001]
Epoch 04/50 | train loss: 0.006 | val loss: 0.235 | train acc: 99.8% | val acc: 94.8% | current lr: [0.0001, 0.001]
Epoch 05/50 | train loss: 0.003 | val loss: 0.214 | train acc: 100.0% | val acc: 94.7% | current lr: [0.0001, 0.001]
Epoch 06/50 | train loss: 0.016 | val loss: 0.226 | train acc: 99.7% | val acc: 94.5% | current lr: [0.0001, 0.001]
Epoch 07/50 | train loss: 0.002 | val loss: 0.215 | train acc: 100.0% | val acc: 95.1% | current lr: [0.0001, 0.001]
Epoch 08/50 | train loss: 0.001 | val loss: 0.223 | train acc: 100.0% | val acc: 94.7% | current lr: [0.0001, 0.001]
Epoch 09/50 | train loss: 0.001 | val loss: 0.222 | train acc: 100.0% | val acc: 94.7% | current lr: [0.0001, 0.001]
Epoch 10/50 | train loss: 0.001 | val loss: 0.212 | train acc: 100.0% | val acc: 95.1% | current lr: [0.0001, 0.001]
Epoch 11/50 | train loss: 0.000 | val loss: 0.224 | train acc: 100.0% | val acc: 94.9% | current lr: [0.0001, 0.001]
Epoch 12/50 | train loss: 0.000 | val loss: 0.219 | train acc: 100.0% | val acc: 95.3% | current lr: [0.0001, 0.001]
Epoch 13/50 | train loss: 0.005 | val loss: 0.290 | train acc: 99.8% | val acc: 94.9% | current lr: [0.0001, 0.001]
Epoch 14/50 | train loss: 0.037 | val loss: 0.508 | train acc: 98.9% | val acc: 89.6% | current lr: [0.0001, 0.001]
Epoch 15/50 | train loss: 0.062 | val loss: 0.468 | train acc: 98.2% | val acc: 93.0% | current lr: [0.0001, 0.001]
Epoch 16/50 | train loss: 0.064 | val loss: 0.381 | train acc: 98.3% | val acc: 93.7% | current lr: [0.0001, 0.001]
Epoch 17/50 | train loss: 0.058 | val loss: 0.500 | train acc: 98.6% | val acc: 93.7% | current lr: [0.0001, 0.001]
Epoch 18/50 | train loss: 0.029 | val loss: 0.331 | train acc: 99.3% | val acc: 94.1% | current lr: [0.0001, 0.001]
Epoch 19/50 | train loss: 0.011 | val loss: 0.388 | train acc: 99.6% | val acc: 95.1% | current lr: [0.0001, 0.001]
Epoch 20/50 | train loss: 0.010 | val loss: 0.348 | train acc: 99.7% | val acc: 94.7% | current lr: [0.0001, 0.001]
Epoch 21/50 | train loss: 0.002 | val loss: 0.302 | train acc: 99.9% | val acc: 95.3% | current lr: [0.0001, 0.001]
Epoch 22/50 | train loss: 0.001 | val loss: 0.336 | train acc: 100.0% | val acc: 94.8% | current lr: [0.0001, 0.001]
Epoch 23/50 | train loss: 0.003 | val loss: 0.308 | train acc: 99.9% | val acc: 95.3% | current lr: [0.0001, 0.001]
Epoch 24/50 | train loss: 0.001 | val loss: 0.299 | train acc: 100.0% | val acc: 95.3% | current lr: [0.0001, 0.001]
Epoch 25/50 | train loss: 0.000 | val loss: 0.316 | train acc: 100.0% | val acc: 95.3% | current lr: [0.0001, 0.001]
Epoch 26/50 | train loss: 0.000 | val loss: 0.276 | train acc: 100.0% | val acc: 95.6% | current lr: [0.0001, 0.001]
Epoch 27/50 | train loss: 0.000 | val loss: 0.278 | train acc: 100.0% | val acc: 95.7% | current lr: [0.0001, 0.001]
Epoch 28/50 | train loss: 0.000 | val loss: 0.276 | train acc: 100.0% | val acc: 95.9% | current lr: [0.0001, 0.001]
Epoch 29/50 | train loss: 0.000 | val loss: 0.280 | train acc: 100.0% | val acc: 95.9% | current lr: [0.0001, 0.001]
Epoch 30/50 | train loss: 0.000 | val loss: 0.294 | train acc: 100.0% | val acc: 95.7% | current lr: [0.0001, 0.001]
Epoch 31/50 | train loss: 0.000 | val loss: 0.278 | train acc: 100.0% | val acc: 95.6% | current lr: [0.0001, 0.001]
Epoch 32/50 | train loss: 0.000 | val loss: 0.295 | train acc: 100.0% | val acc: 95.7% | current lr: [0.0001, 0.001]
Epoch 33/50 | train loss: 0.002 | val loss: 0.311 | train acc: 100.0% | val acc: 95.6% | current lr: [0.0001, 0.001]
Epoch 34/50 | train loss: 0.011 | val loss: 0.413 | train acc: 99.7% | val acc: 94.4% | current lr: [0.0001, 0.001]
Epoch 35/50 | train loss: 0.024 | val loss: 0.424 | train acc: 99.4% | val acc: 95.2% | current lr: [0.0001, 0.001]
Epoch 36/50 | train loss: 0.020 | val loss: 0.430 | train acc: 99.4% | val acc: 94.3% | current lr: [0.0001, 0.001]
Epoch 37/50 | train loss: 0.027 | val loss: 0.483 | train acc: 99.2% | val acc: 93.3% | current lr: [0.0001, 0.001]
Epoch 38/50 | train loss: 0.027 | val loss: 0.446 | train acc: 99.3% | val acc: 94.7% | current lr: [0.0001, 0.001]
Epoch 39/50 | train loss: 0.007 | val loss: 0.387 | train acc: 99.7% | val acc: 94.9% | current lr: [0.0001, 0.001]
Epoch 40/50 | train loss: 0.006 | val loss: 0.461 | train acc: 99.8% | val acc: 93.7% | current lr: [0.0001, 0.001]
Epoch 41/50 | train loss: 0.003 | val loss: 0.424 | train acc: 99.8% | val acc: 95.5% | current lr: [0.0001, 0.001]
Epoch 42/50 | train loss: 0.003 | val loss: 0.454 | train acc: 99.9% | val acc: 94.8% | current lr: [0.0001, 0.001]
Epoch 43/50 | train loss: 0.002 | val loss: 0.367 | train acc: 99.9% | val acc: 95.3% | current lr: [0.0001, 0.001]
Epoch 44/50 | train loss: 0.001 | val loss: 0.362 | train acc: 100.0% | val acc: 95.7% | current lr: [0.0001, 0.001]
✅Epoch 45/50 | train loss: 0.001 | val loss: 0.351 | train acc: 100.0% | val acc: 96.1% | current lr: [0.0001, 0.001]=>Gap=3.9%
Epoch 46/50 | train loss: 0.000 | val loss: 0.345 | train acc: 100.0% | val acc: 96.0% | current lr: [0.0001, 0.001]
Epoch 47/50 | train loss: 0.000 | val loss: 0.355 | train acc: 100.0% | val acc: 96.0% | current lr: [0.0001, 0.001]
Epoch 48/50 | train loss: 0.000 | val loss: 0.340 | train acc: 100.0% | val acc: 95.9% | current lr: [0.0001, 0.001]
Epoch 49/50 | train loss: 0.000 | val loss: 0.361 | train acc: 100.0% | val acc: 95.9% | current lr: [0.0001, 0.001]
Epoch 50/50 | train loss: 0.002 | val loss: 0.395 | train acc: 99.9% | val acc: 95.2% | current lr: [0.0001, 0.001]
Training time: 824.28 seconds
```

</details>

## MobileNetV3-Small — Full Fine-Tuning (ResNet18 Weights Transform)

- **Best validation accuracy:** 96.9% (epoch 36)
- **Final validation accuracy:** 95.1%
- **Training time:** 752.58 seconds (12.54 minutes)
- **Total parameters:** 1,526,056
- **Trainable parameters:** 1,526,056

<details>
<summary>Full training log</summary>

```text
cuda
Total parameters: 1,526,056
Trainable parameters: 1,526,056
Epoch 01/50 | train loss: 0.508 | val loss: 0.358 | train acc: 82.9% | val acc: 89.7% | current lr: 0.001
Epoch 02/50 | train loss: 0.151 | val loss: 0.317 | train acc: 95.0% | val acc: 91.0% | current lr: 0.001
Epoch 03/50 | train loss: 0.114 | val loss: 0.435 | train acc: 96.0% | val acc: 90.2% | current lr: 0.001
Epoch 04/50 | train loss: 0.061 | val loss: 0.363 | train acc: 97.9% | val acc: 90.0% | current lr: 0.001
Epoch 05/50 | train loss: 0.062 | val loss: 0.321 | train acc: 98.0% | val acc: 92.0% | current lr: 0.001
Epoch 06/50 | train loss: 0.072 | val loss: 0.340 | train acc: 97.3% | val acc: 90.4% | current lr: 0.001
Epoch 07/50 | train loss: 0.077 | val loss: 0.464 | train acc: 97.1% | val acc: 91.2% | current lr: 0.001
Epoch 08/50 | train loss: 0.069 | val loss: 0.351 | train acc: 98.0% | val acc: 92.1% | current lr: 0.001
Epoch 09/50 | train loss: 0.039 | val loss: 0.350 | train acc: 98.6% | val acc: 92.4% | current lr: 0.001
Epoch 10/50 | train loss: 0.030 | val loss: 0.355 | train acc: 98.9% | val acc: 92.9% | current lr: 0.001
Epoch 11/50 | train loss: 0.026 | val loss: 0.435 | train acc: 99.2% | val acc: 91.0% | current lr: 0.001
Epoch 12/50 | train loss: 0.043 | val loss: 0.430 | train acc: 98.6% | val acc: 91.7% | current lr: 0.001
Epoch 13/50 | train loss: 0.027 | val loss: 0.448 | train acc: 99.0% | val acc: 91.4% | current lr: 0.001
Epoch 14/50 | train loss: 0.038 | val loss: 0.548 | train acc: 98.9% | val acc: 87.7% | current lr: 0.001
Epoch 15/50 | train loss: 0.019 | val loss: 0.302 | train acc: 99.4% | val acc: 92.5% | current lr: 0.001
Epoch 16/50 | train loss: 0.033 | val loss: 0.258 | train acc: 99.3% | val acc: 94.7% | current lr: 0.001
Epoch 17/50 | train loss: 0.032 | val loss: 0.258 | train acc: 99.1% | val acc: 92.6% | current lr: 0.001
Epoch 18/50 | train loss: 0.033 | val loss: 0.526 | train acc: 99.1% | val acc: 90.4% | current lr: 0.001
Epoch 19/50 | train loss: 0.066 | val loss: 0.420 | train acc: 98.1% | val acc: 89.2% | current lr: 0.001
Epoch 20/50 | train loss: 0.048 | val loss: 0.344 | train acc: 98.7% | val acc: 92.5% | current lr: 0.001
Epoch 21/50 | train loss: 0.032 | val loss: 0.209 | train acc: 99.1% | val acc: 94.8% | current lr: 0.001
Epoch 22/50 | train loss: 0.018 | val loss: 0.228 | train acc: 99.6% | val acc: 94.4% | current lr: 0.001
Epoch 23/50 | train loss: 0.002 | val loss: 0.187 | train acc: 100.0% | val acc: 96.3% | current lr: 0.001
Epoch 24/50 | train loss: 0.001 | val loss: 0.198 | train acc: 100.0% | val acc: 96.0% | current lr: 0.001
Epoch 25/50 | train loss: 0.002 | val loss: 0.231 | train acc: 100.0% | val acc: 96.3% | current lr: 0.001
Epoch 26/50 | train loss: 0.000 | val loss: 0.225 | train acc: 100.0% | val acc: 96.4% | current lr: 0.001
Epoch 27/50 | train loss: 0.002 | val loss: 0.235 | train acc: 99.9% | val acc: 96.0% | current lr: 0.001
Epoch 28/50 | train loss: 0.015 | val loss: 0.251 | train acc: 99.6% | val acc: 95.1% | current lr: 0.001
Epoch 29/50 | train loss: 0.022 | val loss: 0.263 | train acc: 99.2% | val acc: 93.9% | current lr: 0.001
Epoch 30/50 | train loss: 0.038 | val loss: 0.309 | train acc: 98.8% | val acc: 93.9% | current lr: 0.001
Epoch 31/50 | train loss: 0.067 | val loss: 0.397 | train acc: 98.1% | val acc: 91.4% | current lr: 0.001
Epoch 32/50 | train loss: 0.032 | val loss: 0.237 | train acc: 99.1% | val acc: 95.1% | current lr: 0.001
Epoch 33/50 | train loss: 0.015 | val loss: 0.180 | train acc: 99.7% | val acc: 96.0% | current lr: 0.001
Epoch 34/50 | train loss: 0.013 | val loss: 0.214 | train acc: 99.8% | val acc: 95.1% | current lr: 0.001
Epoch 35/50 | train loss: 0.006 | val loss: 0.250 | train acc: 99.9% | val acc: 95.5% | current lr: 0.001
✅Epoch 36/50 | train loss: 0.002 | val loss: 0.221 | train acc: 100.0% | val acc: 96.9% | current lr: 0.001=>Gap=3.1%
Epoch 37/50 | train loss: 0.001 | val loss: 0.224 | train acc: 100.0% | val acc: 96.7% | current lr: 0.001
Epoch 38/50 | train loss: 0.000 | val loss: 0.242 | train acc: 100.0% | val acc: 96.3% | current lr: 0.001
Epoch 39/50 | train loss: 0.030 | val loss: 0.464 | train acc: 99.2% | val acc: 92.6% | current lr: 0.001
Epoch 40/50 | train loss: 0.059 | val loss: 1.025 | train acc: 98.3% | val acc: 85.2% | current lr: 0.001
Epoch 41/50 | train loss: 0.058 | val loss: 0.411 | train acc: 98.2% | val acc: 91.3% | current lr: 0.001
Epoch 42/50 | train loss: 0.032 | val loss: 0.254 | train acc: 99.3% | val acc: 94.5% | current lr: 0.001
Epoch 43/50 | train loss: 0.009 | val loss: 0.284 | train acc: 99.7% | val acc: 94.5% | current lr: 0.001
Epoch 44/50 | train loss: 0.007 | val loss: 0.242 | train acc: 99.7% | val acc: 95.3% | current lr: 0.001
Epoch 45/50 | train loss: 0.017 | val loss: 0.321 | train acc: 99.6% | val acc: 93.6% | current lr: 0.001
Epoch 46/50 | train loss: 0.024 | val loss: 0.526 | train acc: 99.3% | val acc: 89.0% | current lr: 0.001
Epoch 47/50 | train loss: 0.030 | val loss: 0.367 | train acc: 98.9% | val acc: 92.6% | current lr: 0.001
Epoch 48/50 | train loss: 0.044 | val loss: 0.776 | train acc: 98.8% | val acc: 87.6% | current lr: 0.001
Epoch 49/50 | train loss: 0.066 | val loss: 0.391 | train acc: 98.2% | val acc: 92.5% | current lr: 0.001
Epoch 50/50 | train loss: 0.018 | val loss: 0.287 | train acc: 99.3% | val acc: 95.1% | current lr: 0.001
Training time: 752.58 seconds
```

</details>

## MobileNetV3-Small — Full Fine-Tuning (Baseline Transform)

- **Best validation accuracy:** 97.6% (epoch 26)
- **Final validation accuracy:** 95.9%
- **Training time:** 605.42 seconds (10.09 minutes)
- **Total parameters:** 1,526,056
- **Trainable parameters:** 1,526,056

<details>
<summary>Full training log</summary>

```text
cuda
Total parameters: 1,526,056
Trainable parameters: 1,526,056
Epoch 01/50 | train loss: 0.480 | val loss: 1.222 | train acc: 84.7% | val acc: 70.2% | current lr: 0.001
Epoch 02/50 | train loss: 0.119 | val loss: 0.570 | train acc: 96.2% | val acc: 84.9% | current lr: 0.001
Epoch 03/50 | train loss: 0.097 | val loss: 0.232 | train acc: 96.9% | val acc: 92.5% | current lr: 0.001
Epoch 04/50 | train loss: 0.048 | val loss: 1.276 | train acc: 98.5% | val acc: 74.9% | current lr: 0.001
Epoch 05/50 | train loss: 0.069 | val loss: 0.715 | train acc: 98.0% | val acc: 83.8% | current lr: 0.001
Epoch 06/50 | train loss: 0.074 | val loss: 0.607 | train acc: 97.8% | val acc: 89.4% | current lr: 0.001
Epoch 07/50 | train loss: 0.045 | val loss: 0.323 | train acc: 98.4% | val acc: 92.2% | current lr: 0.001
Epoch 08/50 | train loss: 0.044 | val loss: 0.197 | train acc: 98.7% | val acc: 95.7% | current lr: 0.001
Epoch 09/50 | train loss: 0.047 | val loss: 0.471 | train acc: 98.5% | val acc: 90.6% | current lr: 0.001
Epoch 10/50 | train loss: 0.026 | val loss: 0.211 | train acc: 99.0% | val acc: 94.8% | current lr: 0.001
Epoch 11/50 | train loss: 0.027 | val loss: 0.272 | train acc: 99.1% | val acc: 94.3% | current lr: 0.001
Epoch 12/50 | train loss: 0.013 | val loss: 0.190 | train acc: 99.5% | val acc: 96.1% | current lr: 0.001
Epoch 13/50 | train loss: 0.029 | val loss: 0.264 | train acc: 99.4% | val acc: 93.9% | current lr: 0.001
Epoch 14/50 | train loss: 0.011 | val loss: 0.233 | train acc: 99.7% | val acc: 95.7% | current lr: 0.001
Epoch 15/50 | train loss: 0.007 | val loss: 0.195 | train acc: 99.8% | val acc: 95.1% | current lr: 0.001
Epoch 16/50 | train loss: 0.025 | val loss: 0.326 | train acc: 99.2% | val acc: 93.4% | current lr: 0.001
Epoch 17/50 | train loss: 0.059 | val loss: 0.311 | train acc: 98.2% | val acc: 93.6% | current lr: 0.001
Epoch 18/50 | train loss: 0.021 | val loss: 0.297 | train acc: 99.4% | val acc: 94.3% | current lr: 0.001
Epoch 19/50 | train loss: 0.068 | val loss: 0.801 | train acc: 98.4% | val acc: 85.7% | current lr: 0.001
Epoch 20/50 | train loss: 0.059 | val loss: 0.509 | train acc: 98.1% | val acc: 93.3% | current lr: 0.001
Epoch 21/50 | train loss: 0.066 | val loss: 0.394 | train acc: 97.9% | val acc: 91.4% | current lr: 0.001
Epoch 22/50 | train loss: 0.017 | val loss: 0.147 | train acc: 99.5% | val acc: 97.5% | current lr: 0.001
Epoch 23/50 | train loss: 0.007 | val loss: 0.155 | train acc: 99.8% | val acc: 96.7% | current lr: 0.001
Epoch 24/50 | train loss: 0.002 | val loss: 0.146 | train acc: 100.0% | val acc: 97.5% | current lr: 0.001
Epoch 25/50 | train loss: 0.002 | val loss: 0.170 | train acc: 99.9% | val acc: 96.8% | current lr: 0.001

✅✅✅Epoch 26/50 | train loss: 0.002 | val loss: 0.151 | train acc: 100.0% | val acc: 97.6% | current lr: 0.001=>Gap = 2.4%✅✅✅

Epoch 27/50 | train loss: 0.007 | val loss: 0.165 | train acc: 99.8% | val acc: 97.1% | current lr: 0.001
Epoch 28/50 | train loss: 0.038 | val loss: 0.315 | train acc: 98.9% | val acc: 94.4% | current lr: 0.001
Epoch 29/50 | train loss: 0.018 | val loss: 0.212 | train acc: 99.6% | val acc: 97.1% | current lr: 0.001
Epoch 30/50 | train loss: 0.016 | val loss: 0.366 | train acc: 99.7% | val acc: 94.5% | current lr: 0.001
Epoch 31/50 | train loss: 0.033 | val loss: 0.318 | train acc: 99.1% | val acc: 94.0% | current lr: 0.001
Epoch 32/50 | train loss: 0.032 | val loss: 0.199 | train acc: 99.1% | val acc: 96.0% | current lr: 0.001
Epoch 33/50 | train loss: 0.024 | val loss: 0.261 | train acc: 99.3% | val acc: 94.5% | current lr: 0.001
Epoch 34/50 | train loss: 0.018 | val loss: 0.328 | train acc: 99.4% | val acc: 94.1% | current lr: 0.001
Epoch 35/50 | train loss: 0.006 | val loss: 0.365 | train acc: 99.8% | val acc: 95.2% | current lr: 0.001
Epoch 36/50 | train loss: 0.007 | val loss: 0.281 | train acc: 99.8% | val acc: 95.2% | current lr: 0.001
Epoch 37/50 | train loss: 0.021 | val loss: 0.255 | train acc: 99.5% | val acc: 94.8% | current lr: 0.001
Epoch 38/50 | train loss: 0.006 | val loss: 0.230 | train acc: 99.7% | val acc: 96.4% | current lr: 0.001
Epoch 39/50 | train loss: 0.021 | val loss: 0.616 | train acc: 99.4% | val acc: 88.9% | current lr: 0.001
Epoch 40/50 | train loss: 0.034 | val loss: 0.557 | train acc: 99.1% | val acc: 90.2% | current lr: 0.001
Epoch 41/50 | train loss: 0.038 | val loss: 0.256 | train acc: 99.1% | val acc: 94.9% | current lr: 0.001
Epoch 42/50 | train loss: 0.037 | val loss: 0.364 | train acc: 99.0% | val acc: 93.6% | current lr: 0.001
Epoch 43/50 | train loss: 0.018 | val loss: 0.281 | train acc: 99.4% | val acc: 96.0% | current lr: 0.001
Epoch 44/50 | train loss: 0.002 | val loss: 0.214 | train acc: 100.0% | val acc: 96.8% | current lr: 0.001
Epoch 45/50 | train loss: 0.001 | val loss: 0.243 | train acc: 100.0% | val acc: 96.4% | current lr: 0.001
Epoch 46/50 | train loss: 0.001 | val loss: 0.214 | train acc: 100.0% | val acc: 97.1% | current lr: 0.001
Epoch 47/50 | train loss: 0.006 | val loss: 0.453 | train acc: 99.8% | val acc: 93.9% | current lr: 0.001
Epoch 48/50 | train loss: 0.042 | val loss: 0.439 | train acc: 98.9% | val acc: 92.5% | current lr: 0.001
Epoch 49/50 | train loss: 0.013 | val loss: 0.344 | train acc: 99.7% | val acc: 96.0% | current lr: 0.001
Epoch 50/50 | train loss: 0.026 | val loss: 0.244 | train acc: 99.5% | val acc: 95.9% | current lr: 0.001
Training time: 605.42 seconds
```

</details>

## MobileNetV3-Large — Full Fine-Tuning (ResNet-Style Transform)

- **Best validation accuracy:** 96.5% (epoch 26)
- **Final validation accuracy:** 95.5%
- **Training time:** 1152.30 seconds (19.20 minutes)
- **Total parameters:** 4,212,280
- **Trainable parameters:** 4,212,280

<details>
<summary>Full training log</summary>

```text
Total parameters: 4,212,280
Trainable parameters: 4,212,280
Epoch 01/50 | train loss: 0.441 | val loss: 0.566 | train acc: 85.4% | val acc: 84.0% | current lr: 0.001
Epoch 02/50 | train loss: 0.149 | val loss: 0.207 | train acc: 95.4% | val acc: 93.6% | current lr: 0.001
Epoch 03/50 | train loss: 0.114 | val loss: 0.525 | train acc: 96.5% | val acc: 89.2% | current lr: 0.001
Epoch 04/50 | train loss: 0.077 | val loss: 0.237 | train acc: 97.6% | val acc: 94.1% | current lr: 0.001
Epoch 05/50 | train loss: 0.076 | val loss: 0.290 | train acc: 97.7% | val acc: 92.2% | current lr: 0.001
Epoch 06/50 | train loss: 0.065 | val loss: 0.306 | train acc: 97.9% | val acc: 93.0% | current lr: 0.001
Epoch 07/50 | train loss: 0.061 | val loss: 0.274 | train acc: 98.2% | val acc: 92.6% | current lr: 0.001
Epoch 08/50 | train loss: 0.052 | val loss: 0.318 | train acc: 98.3% | val acc: 90.5% | current lr: 0.001
Epoch 09/50 | train loss: 0.034 | val loss: 0.242 | train acc: 98.8% | val acc: 94.3% | current lr: 0.001
Epoch 10/50 | train loss: 0.019 | val loss: 0.209 | train acc: 99.5% | val acc: 95.6% | current lr: 0.001
Epoch 11/50 | train loss: 0.079 | val loss: 0.353 | train acc: 97.8% | val acc: 91.8% | current lr: 0.001
Epoch 12/50 | train loss: 0.057 | val loss: 0.361 | train acc: 98.2% | val acc: 92.2% | current lr: 0.001
Epoch 13/50 | train loss: 0.030 | val loss: 0.418 | train acc: 99.1% | val acc: 91.7% | current lr: 0.001
Epoch 14/50 | train loss: 0.014 | val loss: 0.192 | train acc: 99.6% | val acc: 95.9% | current lr: 0.001
Epoch 15/50 | train loss: 0.055 | val loss: 0.350 | train acc: 98.4% | val acc: 94.3% | current lr: 0.001
Epoch 16/50 | train loss: 0.064 | val loss: 0.387 | train acc: 98.2% | val acc: 91.2% | current lr: 0.001
Epoch 17/50 | train loss: 0.026 | val loss: 0.214 | train acc: 99.1% | val acc: 94.3% | current lr: 0.001
Epoch 18/50 | train loss: 0.014 | val loss: 0.359 | train acc: 99.5% | val acc: 93.6% | current lr: 0.001
Epoch 19/50 | train loss: 0.010 | val loss: 0.241 | train acc: 99.7% | val acc: 93.9% | current lr: 0.001
Epoch 20/50 | train loss: 0.039 | val loss: 0.780 | train acc: 98.9% | val acc: 81.6% | current lr: 0.001
Epoch 21/50 | train loss: 0.062 | val loss: 0.538 | train acc: 97.8% | val acc: 89.2% | current lr: 0.001
Epoch 22/50 | train loss: 0.116 | val loss: 0.574 | train acc: 96.8% | val acc: 86.0% | current lr: 0.001
Epoch 23/50 | train loss: 0.042 | val loss: 0.343 | train acc: 99.0% | val acc: 92.9% | current lr: 0.001
Epoch 24/50 | train loss: 0.017 | val loss: 0.285 | train acc: 99.6% | val acc: 94.3% | current lr: 0.001
Epoch 25/50 | train loss: 0.016 | val loss: 0.308 | train acc: 99.6% | val acc: 94.0% | current lr: 0.001
✅Epoch 26/50 | train loss: 0.025 | val loss: 0.165 | train acc: 99.3% | val acc: 96.5% | current lr: 0.001=> Gap=2.8%
Epoch 27/50 | train loss: 0.004 | val loss: 0.196 | train acc: 99.9% | val acc: 95.6% | current lr: 0.001
Epoch 28/50 | train loss: 0.010 | val loss: 0.398 | train acc: 99.7% | val acc: 92.1% | current lr: 0.001
Epoch 29/50 | train loss: 0.022 | val loss: 0.303 | train acc: 99.2% | val acc: 94.1% | current lr: 0.001
Epoch 30/50 | train loss: 0.031 | val loss: 0.640 | train acc: 99.0% | val acc: 87.8% | current lr: 0.001
Epoch 31/50 | train loss: 0.051 | val loss: 0.330 | train acc: 98.8% | val acc: 92.9% | current lr: 0.001
Epoch 32/50 | train loss: 0.031 | val loss: 0.293 | train acc: 99.0% | val acc: 94.9% | current lr: 0.001
Epoch 33/50 | train loss: 0.013 | val loss: 0.290 | train acc: 99.6% | val acc: 95.7% | current lr: 0.001
Epoch 34/50 | train loss: 0.003 | val loss: 0.314 | train acc: 99.9% | val acc: 95.9% | current lr: 0.001
Epoch 35/50 | train loss: 0.001 | val loss: 0.239 | train acc: 100.0% | val acc: 96.4% | current lr: 0.001
Epoch 36/50 | train loss: 0.005 | val loss: 0.250 | train acc: 99.9% | val acc: 95.5% | current lr: 0.001
Epoch 37/50 | train loss: 0.002 | val loss: 0.230 | train acc: 99.9% | val acc: 96.3% | current lr: 0.001
Epoch 38/50 | train loss: 0.005 | val loss: 0.213 | train acc: 99.9% | val acc: 94.9% | current lr: 0.001
Epoch 39/50 | train loss: 0.001 | val loss: 0.214 | train acc: 100.0% | val acc: 96.4% | current lr: 0.001
Epoch 40/50 | train loss: 0.003 | val loss: 0.243 | train acc: 99.9% | val acc: 95.9% | current lr: 0.001
Epoch 41/50 | train loss: 0.095 | val loss: 3.263 | train acc: 98.0% | val acc: 51.3% | current lr: 0.001
Epoch 42/50 | train loss: 0.128 | val loss: 0.537 | train acc: 96.0% | val acc: 90.0% | current lr: 0.001
Epoch 43/50 | train loss: 0.030 | val loss: 0.287 | train acc: 99.3% | val acc: 93.7% | current lr: 0.001
Epoch 44/50 | train loss: 0.024 | val loss: 0.315 | train acc: 99.1% | val acc: 94.4% | current lr: 0.001
Epoch 45/50 | train loss: 0.007 | val loss: 0.269 | train acc: 99.7% | val acc: 94.7% | current lr: 0.001
Epoch 46/50 | train loss: 0.004 | val loss: 0.303 | train acc: 99.9% | val acc: 94.9% | current lr: 0.001
Epoch 47/50 | train loss: 0.002 | val loss: 0.278 | train acc: 99.9% | val acc: 95.3% | current lr: 0.001
Epoch 48/50 | train loss: 0.000 | val loss: 0.251 | train acc: 100.0% | val acc: 95.6% | current lr: 0.001
Epoch 49/50 | train loss: 0.000 | val loss: 0.241 | train acc: 100.0% | val acc: 95.9% | current lr: 0.001
Epoch 50/50 | train loss: 0.000 | val loss: 0.264 | train acc: 100.0% | val acc: 95.5% | current lr: 0.001
Training time:1152.30 seconds
```

</details>

## MobileNetV3-Large — Head Fine-Tuning (ResNet-Style Transform)

- **Best validation accuracy:** 91.8% (epoch 15)
- **Final validation accuracy:** 90.2%
- **Training time:** 651.72 seconds (10.86 minutes)
- **Total parameters:** 4,212,280
- **Trainable parameters:** 1,240,328

<details>
<summary>Full training log</summary>

```text
Total parameters: 4,212,280
Trainable parameters: 1,240,328
Epoch 01/50 | train loss: 0.752 | val loss: 0.573 | train acc: 75.9% | val acc: 80.9% | current lr: 0.001
Epoch 02/50 | train loss: 0.347 | val loss: 0.355 | train acc: 88.5% | val acc: 88.6% | current lr: 0.001
Epoch 03/50 | train loss: 0.237 | val loss: 0.396 | train acc: 92.0% | val acc: 86.0% | current lr: 0.001
Epoch 04/50 | train loss: 0.170 | val loss: 0.320 | train acc: 94.2% | val acc: 89.3% | current lr: 0.001
Epoch 05/50 | train loss: 0.136 | val loss: 0.362 | train acc: 95.4% | val acc: 88.4% | current lr: 0.001
Epoch 06/50 | train loss: 0.127 | val loss: 0.361 | train acc: 95.9% | val acc: 90.4% | current lr: 0.001
Epoch 07/50 | train loss: 0.104 | val loss: 0.380 | train acc: 96.6% | val acc: 89.2% | current lr: 0.001
Epoch 08/50 | train loss: 0.078 | val loss: 0.339 | train acc: 97.2% | val acc: 89.6% | current lr: 0.001
Epoch 09/50 | train loss: 0.057 | val loss: 0.415 | train acc: 98.3% | val acc: 89.6% | current lr: 0.001
Epoch 10/50 | train loss: 0.063 | val loss: 0.373 | train acc: 97.9% | val acc: 90.0% | current lr: 0.001
Epoch 11/50 | train loss: 0.051 | val loss: 0.403 | train acc: 98.2% | val acc: 89.7% | current lr: 0.001
Epoch 12/50 | train loss: 0.040 | val loss: 0.368 | train acc: 99.0% | val acc: 89.3% | current lr: 0.001
Epoch 13/50 | train loss: 0.037 | val loss: 0.382 | train acc: 98.9% | val acc: 90.4% | current lr: 0.001
Epoch 14/50 | train loss: 0.030 | val loss: 0.396 | train acc: 99.1% | val acc: 90.4% | current lr: 0.001
✅Epoch 15/50 | train loss: 0.023 | val loss: 0.390 | train acc: 99.5% | val acc: 91.8% | current lr: 0.001=>Gap=7.7%
Epoch 16/50 | train loss: 0.032 | val loss: 0.488 | train acc: 98.9% | val acc: 89.3% | current lr: 0.001
Epoch 17/50 | train loss: 0.035 | val loss: 0.499 | train acc: 98.6% | val acc: 89.3% | current lr: 0.001
Epoch 18/50 | train loss: 0.049 | val loss: 0.448 | train acc: 98.2% | val acc: 90.5% | current lr: 0.001
Epoch 19/50 | train loss: 0.041 | val loss: 0.406 | train acc: 98.9% | val acc: 90.5% | current lr: 0.001
Epoch 20/50 | train loss: 0.017 | val loss: 0.446 | train acc: 99.5% | val acc: 90.4% | current lr: 0.001
Epoch 21/50 | train loss: 0.011 | val loss: 0.380 | train acc: 99.8% | val acc: 90.8% | current lr: 0.001
Epoch 22/50 | train loss: 0.017 | val loss: 0.451 | train acc: 99.4% | val acc: 90.6% | current lr: 0.001
Epoch 23/50 | train loss: 0.010 | val loss: 0.440 | train acc: 99.7% | val acc: 90.1% | current lr: 0.001
Epoch 24/50 | train loss: 0.017 | val loss: 0.450 | train acc: 99.6% | val acc: 90.0% | current lr: 0.001
Epoch 25/50 | train loss: 0.012 | val loss: 0.437 | train acc: 99.6% | val acc: 90.5% | current lr: 0.001
Epoch 26/50 | train loss: 0.017 | val loss: 0.461 | train acc: 99.3% | val acc: 90.8% | current lr: 0.001
Epoch 27/50 | train loss: 0.016 | val loss: 0.477 | train acc: 99.4% | val acc: 89.4% | current lr: 0.001
Epoch 28/50 | train loss: 0.057 | val loss: 0.466 | train acc: 98.3% | val acc: 88.5% | current lr: 0.001
Epoch 29/50 | train loss: 0.040 | val loss: 0.525 | train acc: 98.3% | val acc: 89.4% | current lr: 0.001
Epoch 30/50 | train loss: 0.025 | val loss: 0.472 | train acc: 99.1% | val acc: 90.1% | current lr: 0.001
Epoch 31/50 | train loss: 0.046 | val loss: 0.444 | train acc: 98.3% | val acc: 90.0% | current lr: 0.001
Epoch 32/50 | train loss: 0.014 | val loss: 0.463 | train acc: 99.4% | val acc: 90.9% | current lr: 0.001
Epoch 33/50 | train loss: 0.020 | val loss: 0.520 | train acc: 99.3% | val acc: 90.1% | current lr: 0.001
Epoch 34/50 | train loss: 0.019 | val loss: 0.462 | train acc: 99.4% | val acc: 90.2% | current lr: 0.001
Epoch 35/50 | train loss: 0.017 | val loss: 0.402 | train acc: 99.2% | val acc: 91.0% | current lr: 0.001
Epoch 36/50 | train loss: 0.011 | val loss: 0.490 | train acc: 99.5% | val acc: 89.8% | current lr: 0.001
Epoch 37/50 | train loss: 0.012 | val loss: 0.432 | train acc: 99.7% | val acc: 91.0% | current lr: 0.001
Epoch 38/50 | train loss: 0.010 | val loss: 0.444 | train acc: 99.6% | val acc: 90.1% | current lr: 0.001
Epoch 39/50 | train loss: 0.017 | val loss: 0.601 | train acc: 99.5% | val acc: 88.4% | current lr: 0.001
Epoch 40/50 | train loss: 0.091 | val loss: 0.535 | train acc: 97.2% | val acc: 90.8% | current lr: 0.001
Epoch 41/50 | train loss: 0.016 | val loss: 0.533 | train acc: 99.4% | val acc: 91.0% | current lr: 0.001
Epoch 42/50 | train loss: 0.013 | val loss: 0.601 | train acc: 99.5% | val acc: 89.2% | current lr: 0.001
Epoch 43/50 | train loss: 0.008 | val loss: 0.534 | train acc: 99.8% | val acc: 90.4% | current lr: 0.001
Epoch 44/50 | train loss: 0.040 | val loss: 0.485 | train acc: 98.7% | val acc: 90.6% | current lr: 0.001
Epoch 45/50 | train loss: 0.070 | val loss: 0.891 | train acc: 97.7% | val acc: 87.4% | current lr: 0.001
Epoch 46/50 | train loss: 0.040 | val loss: 0.567 | train acc: 98.8% | val acc: 90.0% | current lr: 0.001
Epoch 47/50 | train loss: 0.006 | val loss: 0.524 | train acc: 99.8% | val acc: 90.6% | current lr: 0.001
Epoch 48/50 | train loss: 0.005 | val loss: 0.486 | train acc: 99.8% | val acc: 90.8% | current lr: 0.001
Epoch 49/50 | train loss: 0.005 | val loss: 0.532 | train acc: 99.8% | val acc: 90.6% | current lr: 0.001
Epoch 50/50 | train loss: 0.014 | val loss: 0.529 | train acc: 99.7% | val acc: 90.2% | current lr: 0.001
Training time:651.72 seconds
```

</details>

## MobileNetV3-Large — Last Block (3) Fine-Tuning (ResNet-Style Transform)

- **Best validation accuracy:** 96.0% (epoch 36)
- **Final validation accuracy:** 91.8%
- **Training time:** 689.21 seconds (11.49 minutes)
- **Total parameters:** 4,212,280
- **Trainable parameters:** 2,990,568

<details>
<summary>Full training log</summary>

```text
Total parameters: 4,212,280
Trainable parameters: 2,990,568
Epoch 01/50 | train loss: 0.487 | val loss: 0.449 | train acc: 83.8% | val acc: 88.5% | current lr: 0.001
Epoch 02/50 | train loss: 0.113 | val loss: 0.411 | train acc: 96.8% | val acc: 90.1% | current lr: 0.001
Epoch 03/50 | train loss: 0.092 | val loss: 0.480 | train acc: 97.1% | val acc: 89.7% | current lr: 0.001
Epoch 04/50 | train loss: 0.047 | val loss: 0.246 | train acc: 98.4% | val acc: 93.4% | current lr: 0.001
Epoch 05/50 | train loss: 0.023 | val loss: 0.233 | train acc: 99.3% | val acc: 94.3% | current lr: 0.001
Epoch 06/50 | train loss: 0.050 | val loss: 0.280 | train acc: 98.7% | val acc: 92.8% | current lr: 0.001
Epoch 07/50 | train loss: 0.073 | val loss: 0.338 | train acc: 97.7% | val acc: 93.9% | current lr: 0.001
Epoch 08/50 | train loss: 0.048 | val loss: 0.446 | train acc: 98.6% | val acc: 90.1% | current lr: 0.001
Epoch 09/50 | train loss: 0.023 | val loss: 0.411 | train acc: 99.3% | val acc: 92.6% | current lr: 0.001
Epoch 10/50 | train loss: 0.014 | val loss: 0.432 | train acc: 99.4% | val acc: 91.4% | current lr: 0.001
Epoch 11/50 | train loss: 0.015 | val loss: 0.266 | train acc: 99.6% | val acc: 95.1% | current lr: 0.001
Epoch 12/50 | train loss: 0.006 | val loss: 0.268 | train acc: 99.8% | val acc: 94.4% | current lr: 0.001
Epoch 13/50 | train loss: 0.005 | val loss: 0.265 | train acc: 99.9% | val acc: 94.7% | current lr: 0.001
Epoch 14/50 | train loss: 0.010 | val loss: 0.363 | train acc: 99.6% | val acc: 93.7% | current lr: 0.001
Epoch 15/50 | train loss: 0.069 | val loss: 0.535 | train acc: 98.2% | val acc: 91.3% | current lr: 0.001
Epoch 16/50 | train loss: 0.033 | val loss: 0.468 | train acc: 99.1% | val acc: 91.8% | current lr: 0.001
Epoch 17/50 | train loss: 0.050 | val loss: 0.368 | train acc: 98.3% | val acc: 92.6% | current lr: 0.001
Epoch 18/50 | train loss: 0.014 | val loss: 0.233 | train acc: 99.5% | val acc: 94.9% | current lr: 0.001
Epoch 19/50 | train loss: 0.007 | val loss: 0.281 | train acc: 99.8% | val acc: 94.8% | current lr: 0.001
Epoch 20/50 | train loss: 0.009 | val loss: 0.300 | train acc: 99.8% | val acc: 93.7% | current lr: 0.001
Epoch 21/50 | train loss: 0.011 | val loss: 0.426 | train acc: 99.6% | val acc: 92.6% | current lr: 0.001
Epoch 22/50 | train loss: 0.021 | val loss: 0.329 | train acc: 99.4% | val acc: 94.5% | current lr: 0.001
Epoch 23/50 | train loss: 0.003 | val loss: 0.286 | train acc: 99.9% | val acc: 95.6% | current lr: 0.001
Epoch 24/50 | train loss: 0.009 | val loss: 0.272 | train acc: 99.7% | val acc: 94.7% | current lr: 0.001
Epoch 25/50 | train loss: 0.003 | val loss: 0.266 | train acc: 99.9% | val acc: 95.6% | current lr: 0.001
Epoch 26/50 | train loss: 0.006 | val loss: 0.353 | train acc: 99.8% | val acc: 95.3% | current lr: 0.001
Epoch 27/50 | train loss: 0.015 | val loss: 0.356 | train acc: 99.5% | val acc: 94.8% | current lr: 0.001
Epoch 28/50 | train loss: 0.119 | val loss: 0.536 | train acc: 97.7% | val acc: 90.9% | current lr: 0.001
Epoch 29/50 | train loss: 0.038 | val loss: 0.442 | train acc: 98.8% | val acc: 93.9% | current lr: 0.001
Epoch 30/50 | train loss: 0.026 | val loss: 0.373 | train acc: 99.3% | val acc: 94.7% | current lr: 0.001
Epoch 31/50 | train loss: 0.028 | val loss: 0.326 | train acc: 99.2% | val acc: 94.5% | current lr: 0.001
Epoch 32/50 | train loss: 0.006 | val loss: 0.291 | train acc: 99.8% | val acc: 95.3% | current lr: 0.001
Epoch 33/50 | train loss: 0.004 | val loss: 0.371 | train acc: 99.9% | val acc: 95.3% | current lr: 0.001
Epoch 34/50 | train loss: 0.007 | val loss: 0.333 | train acc: 99.8% | val acc: 95.5% | current lr: 0.001
Epoch 35/50 | train loss: 0.003 | val loss: 0.300 | train acc: 99.9% | val acc: 95.9% | current lr: 0.001
✅Epoch 36/50 | train loss: 0.000 | val loss: 0.299 | train acc: 100.0% | val acc: 96.0% | current lr: 0.001=>Gap=4%
Epoch 37/50 | train loss: 0.012 | val loss: 0.298 | train acc: 99.7% | val acc: 95.1% | current lr: 0.001
Epoch 38/50 | train loss: 0.006 | val loss: 0.385 | train acc: 99.8% | val acc: 95.2% | current lr: 0.001
Epoch 39/50 | train loss: 0.002 | val loss: 0.353 | train acc: 99.9% | val acc: 94.5% | current lr: 0.001
Epoch 40/50 | train loss: 0.004 | val loss: 0.373 | train acc: 99.9% | val acc: 95.5% | current lr: 0.001
Epoch 41/50 | train loss: 0.011 | val loss: 0.358 | train acc: 99.8% | val acc: 94.5% | current lr: 0.001
Epoch 42/50 | train loss: 0.021 | val loss: 0.378 | train acc: 99.6% | val acc: 93.7% | current lr: 0.001
Epoch 43/50 | train loss: 0.009 | val loss: 0.394 | train acc: 99.7% | val acc: 94.8% | current lr: 0.001
Epoch 44/50 | train loss: 0.011 | val loss: 0.388 | train acc: 99.7% | val acc: 95.2% | current lr: 0.001
Epoch 45/50 | train loss: 0.006 | val loss: 0.441 | train acc: 99.8% | val acc: 94.9% | current lr: 0.001
Epoch 46/50 | train loss: 0.020 | val loss: 0.467 | train acc: 99.6% | val acc: 92.4% | current lr: 0.001
Epoch 47/50 | train loss: 0.015 | val loss: 0.480 | train acc: 99.3% | val acc: 95.3% | current lr: 0.001
Epoch 48/50 | train loss: 0.033 | val loss: 0.470 | train acc: 99.4% | val acc: 93.7% | current lr: 0.001
Epoch 49/50 | train loss: 0.022 | val loss: 0.382 | train acc: 99.6% | val acc: 95.1% | current lr: 0.001
Epoch 50/50 | train loss: 0.039 | val loss: 0.621 | train acc: 99.3% | val acc: 91.8% | current lr: 0.001
Training time:689.21 seconds
```

</details>

## MobileNetV3-Large — Full Fine-Tuning (Mean/Std = 0.5)

- **Best validation accuracy:** 97.5% (epoch 29)
- **Final validation accuracy:** 90.2%
- **Training time:** 980.44 seconds (16.34 minutes)
- **Total parameters:** 4,212,280
- **Trainable parameters:** 4,212,280

<details>
<summary>Full training log</summary>

```text
Total parameters: 4,212,280
Trainable parameters: 4,212,280
Epoch 01/50 | train loss: 0.422 | val loss: 0.575 | train acc: 86.9% | val acc: 83.2% | current lr: 0.001
Epoch 02/50 | train loss: 0.142 | val loss: 0.514 | train acc: 95.4% | val acc: 87.4% | current lr: 0.001
Epoch 03/50 | train loss: 0.092 | val loss: 0.484 | train acc: 97.3% | val acc: 87.8% | current lr: 0.001
Epoch 04/50 | train loss: 0.061 | val loss: 0.364 | train acc: 98.2% | val acc: 92.2% | current lr: 0.001
Epoch 05/50 | train loss: 0.060 | val loss: 1.356 | train acc: 98.1% | val acc: 76.3% | current lr: 0.001
Epoch 06/50 | train loss: 0.070 | val loss: 0.188 | train acc: 97.6% | val acc: 95.6% | current lr: 0.001
Epoch 07/50 | train loss: 0.025 | val loss: 0.160 | train acc: 99.2% | val acc: 95.9% | current lr: 0.001
Epoch 08/50 | train loss: 0.041 | val loss: 0.499 | train acc: 98.9% | val acc: 88.4% | current lr: 0.001
Epoch 09/50 | train loss: 0.054 | val loss: 0.685 | train acc: 98.2% | val acc: 87.0% | current lr: 0.001
Epoch 10/50 | train loss: 0.045 | val loss: 0.276 | train acc: 98.5% | val acc: 94.3% | current lr: 0.001
Epoch 11/50 | train loss: 0.070 | val loss: 0.247 | train acc: 98.0% | val acc: 94.8% | current lr: 0.001
Epoch 12/50 | train loss: 0.018 | val loss: 0.492 | train acc: 99.3% | val acc: 90.4% | current lr: 0.001
Epoch 13/50 | train loss: 0.009 | val loss: 0.194 | train acc: 99.7% | val acc: 96.4% | current lr: 0.001
Epoch 14/50 | train loss: 0.024 | val loss: 0.454 | train acc: 99.2% | val acc: 92.4% | current lr: 0.001
Epoch 15/50 | train loss: 0.078 | val loss: 0.324 | train acc: 97.4% | val acc: 91.7% | current lr: 0.001
Epoch 16/50 | train loss: 0.039 | val loss: 0.294 | train acc: 98.8% | val acc: 93.0% | current lr: 0.001
Epoch 17/50 | train loss: 0.054 | val loss: 0.326 | train acc: 98.5% | val acc: 93.3% | current lr: 0.001
Epoch 18/50 | train loss: 0.018 | val loss: 0.302 | train acc: 99.5% | val acc: 94.7% | current lr: 0.001
Epoch 19/50 | train loss: 0.019 | val loss: 0.207 | train acc: 99.6% | val acc: 96.5% | current lr: 0.001
Epoch 20/50 | train loss: 0.006 | val loss: 0.210 | train acc: 99.8% | val acc: 96.1% | current lr: 0.001
Epoch 21/50 | train loss: 0.013 | val loss: 0.261 | train acc: 99.5% | val acc: 95.5% | current lr: 0.001
Epoch 22/50 | train loss: 0.013 | val loss: 0.278 | train acc: 99.6% | val acc: 96.7% | current lr: 0.001
Epoch 23/50 | train loss: 0.039 | val loss: 0.249 | train acc: 98.8% | val acc: 93.2% | current lr: 0.001
Epoch 24/50 | train loss: 0.023 | val loss: 0.408 | train acc: 99.2% | val acc: 93.0% | current lr: 0.001
Epoch 25/50 | train loss: 0.059 | val loss: 0.244 | train acc: 98.4% | val acc: 93.3% | current lr: 0.001
Epoch 26/50 | train loss: 0.021 | val loss: 0.298 | train acc: 99.2% | val acc: 96.0% | current lr: 0.001
Epoch 27/50 | train loss: 0.025 | val loss: 0.282 | train acc: 99.4% | val acc: 95.2% | current lr: 0.001
Epoch 28/50 | train loss: 0.014 | val loss: 0.186 | train acc: 99.7% | val acc: 96.8% | current lr: 0.001
✅✅✅Epoch 29/50 | train loss: 0.005 | val loss: 0.182 | train acc: 99.8% | val acc: 97.5% | current lr: 0.001=>Gap=2.3%✅✅✅
Epoch 30/50 | train loss: 0.013 | val loss: 0.399 | train acc: 99.5% | val acc: 93.9% | current lr: 0.001
Epoch 31/50 | train loss: 0.028 | val loss: 0.230 | train acc: 99.1% | val acc: 96.1% | current lr: 0.001
Epoch 32/50 | train loss: 0.038 | val loss: 0.311 | train acc: 98.7% | val acc: 94.0% | current lr: 0.001
Epoch 33/50 | train loss: 0.043 | val loss: 0.556 | train acc: 99.1% | val acc: 88.9% | current lr: 0.001
Epoch 34/50 | train loss: 0.029 | val loss: 0.263 | train acc: 99.2% | val acc: 95.6% | current lr: 0.001
Epoch 35/50 | train loss: 0.029 | val loss: 0.290 | train acc: 99.4% | val acc: 95.5% | current lr: 0.001
Epoch 36/50 | train loss: 0.011 | val loss: 0.341 | train acc: 99.6% | val acc: 96.0% | current lr: 0.001
Epoch 37/50 | train loss: 0.019 | val loss: 0.206 | train acc: 99.5% | val acc: 96.7% | current lr: 0.001
Epoch 38/50 | train loss: 0.004 | val loss: 0.265 | train acc: 99.8% | val acc: 97.1% | current lr: 0.001
Epoch 39/50 | train loss: 0.001 | val loss: 0.214 | train acc: 100.0% | val acc: 97.5% | current lr: 0.001
Epoch 40/50 | train loss: 0.005 | val loss: 0.264 | train acc: 99.9% | val acc: 96.7% | current lr: 0.001
Epoch 41/50 | train loss: 0.016 | val loss: 0.214 | train acc: 99.6% | val acc: 96.7% | current lr: 0.001
Epoch 42/50 | train loss: 0.024 | val loss: 0.500 | train acc: 99.3% | val acc: 92.4% | current lr: 0.001
Epoch 43/50 | train loss: 0.062 | val loss: 0.352 | train acc: 98.5% | val acc: 91.7% | current lr: 0.001
Epoch 44/50 | train loss: 0.021 | val loss: 0.163 | train acc: 99.3% | val acc: 96.5% | current lr: 0.001
Epoch 45/50 | train loss: 0.009 | val loss: 0.174 | train acc: 99.7% | val acc: 96.4% | current lr: 0.001
Epoch 46/50 | train loss: 0.001 | val loss: 0.157 | train acc: 100.0% | val acc: 97.3% | current lr: 0.001
Epoch 47/50 | train loss: 0.003 | val loss: 0.171 | train acc: 99.9% | val acc: 96.8% | current lr: 0.001
Epoch 48/50 | train loss: 0.018 | val loss: 0.363 | train acc: 99.5% | val acc: 92.8% | current lr: 0.001
Epoch 49/50 | train loss: 0.025 | val loss: 0.568 | train acc: 99.2% | val acc: 91.4% | current lr: 0.001
Epoch 50/50 | train loss: 0.060 | val loss: 0.704 | train acc: 98.4% | val acc: 90.2% | current lr: 0.001
Training time:980.44 seconds
```

</details>

## MobileNetV3-Large — Head Fine-Tuning (Mean/Std = 0.5)

- **Best validation accuracy:** 91.0% (epoch 50)
- **Final validation accuracy:** 91.0%
- **Training time:** 532.67 seconds (8.88 minutes)
- **Total parameters:** 4,212,280
- **Trainable parameters:** 1,240,328

<details>
<summary>Full training log</summary>

```text
Total parameters: 4,212,280
Trainable parameters: 1,240,328
Epoch 01/50 | train loss: 0.692 | val loss: 0.503 | train acc: 76.8% | val acc: 84.0% | current lr: 0.001
Epoch 02/50 | train loss: 0.305 | val loss: 0.308 | train acc: 89.7% | val acc: 89.2% | current lr: 0.001
Epoch 03/50 | train loss: 0.207 | val loss: 0.354 | train acc: 92.7% | val acc: 88.4% | current lr: 0.001
Epoch 04/50 | train loss: 0.152 | val loss: 0.301 | train acc: 95.0% | val acc: 89.2% | current lr: 0.001
Epoch 05/50 | train loss: 0.131 | val loss: 0.278 | train acc: 95.6% | val acc: 90.5% | current lr: 0.001
Epoch 06/50 | train loss: 0.105 | val loss: 0.393 | train acc: 96.5% | val acc: 87.7% | current lr: 0.001
Epoch 07/50 | train loss: 0.112 | val loss: 0.356 | train acc: 95.8% | val acc: 88.9% | current lr: 0.001
Epoch 08/50 | train loss: 0.087 | val loss: 0.313 | train acc: 97.0% | val acc: 89.8% | current lr: 0.001
Epoch 09/50 | train loss: 0.048 | val loss: 0.362 | train acc: 98.5% | val acc: 89.3% | current lr: 0.001
Epoch 10/50 | train loss: 0.042 | val loss: 0.361 | train acc: 98.8% | val acc: 90.2% | current lr: 0.001
Epoch 11/50 | train loss: 0.033 | val loss: 0.384 | train acc: 99.1% | val acc: 90.6% | current lr: 0.001
Epoch 12/50 | train loss: 0.039 | val loss: 0.393 | train acc: 98.5% | val acc: 90.0% | current lr: 0.001
Epoch 13/50 | train loss: 0.040 | val loss: 0.406 | train acc: 98.6% | val acc: 90.0% | current lr: 0.001
Epoch 14/50 | train loss: 0.032 | val loss: 0.364 | train acc: 99.2% | val acc: 90.8% | current lr: 0.001
Epoch 15/50 | train loss: 0.029 | val loss: 0.474 | train acc: 99.0% | val acc: 88.8% | current lr: 0.001
Epoch 16/50 | train loss: 0.036 | val loss: 0.476 | train acc: 98.7% | val acc: 88.5% | current lr: 0.001
Epoch 17/50 | train loss: 0.025 | val loss: 0.444 | train acc: 99.2% | val acc: 89.2% | current lr: 0.001
Epoch 18/50 | train loss: 0.039 | val loss: 0.412 | train acc: 98.7% | val acc: 89.4% | current lr: 0.001
Epoch 19/50 | train loss: 0.021 | val loss: 0.404 | train acc: 99.4% | val acc: 89.3% | current lr: 0.001
Epoch 20/50 | train loss: 0.019 | val loss: 0.377 | train acc: 99.4% | val acc: 90.2% | current lr: 0.001
Epoch 21/50 | train loss: 0.017 | val loss: 0.436 | train acc: 99.5% | val acc: 89.6% | current lr: 0.001
Epoch 22/50 | train loss: 0.033 | val loss: 0.390 | train acc: 98.6% | val acc: 89.6% | current lr: 0.001
Epoch 23/50 | train loss: 0.019 | val loss: 0.397 | train acc: 99.3% | val acc: 90.2% | current lr: 0.001
Epoch 24/50 | train loss: 0.015 | val loss: 0.398 | train acc: 99.4% | val acc: 89.7% | current lr: 0.001
Epoch 25/50 | train loss: 0.028 | val loss: 0.465 | train acc: 99.2% | val acc: 89.0% | current lr: 0.001
Epoch 26/50 | train loss: 0.017 | val loss: 0.413 | train acc: 99.5% | val acc: 90.2% | current lr: 0.001
Epoch 27/50 | train loss: 0.020 | val loss: 0.556 | train acc: 99.1% | val acc: 88.2% | current lr: 0.001
Epoch 28/50 | train loss: 0.027 | val loss: 0.483 | train acc: 98.9% | val acc: 89.0% | current lr: 0.001
Epoch 29/50 | train loss: 0.030 | val loss: 0.440 | train acc: 99.0% | val acc: 90.0% | current lr: 0.001
Epoch 30/50 | train loss: 0.012 | val loss: 0.494 | train acc: 99.7% | val acc: 90.9% | current lr: 0.001
Epoch 31/50 | train loss: 0.029 | val loss: 0.445 | train acc: 99.2% | val acc: 90.8% | current lr: 0.001
Epoch 32/50 | train loss: 0.051 | val loss: 0.505 | train acc: 98.3% | val acc: 89.2% | current lr: 0.001
Epoch 33/50 | train loss: 0.016 | val loss: 0.592 | train acc: 99.5% | val acc: 87.2% | current lr: 0.001
Epoch 34/50 | train loss: 0.016 | val loss: 0.497 | train acc: 99.5% | val acc: 90.4% | current lr: 0.001
Epoch 35/50 | train loss: 0.009 | val loss: 0.413 | train acc: 99.7% | val acc: 90.5% | current lr: 0.001
Epoch 36/50 | train loss: 0.011 | val loss: 0.497 | train acc: 99.6% | val acc: 89.2% | current lr: 0.001
Epoch 37/50 | train loss: 0.025 | val loss: 0.455 | train acc: 99.3% | val acc: 90.1% | current lr: 0.001
Epoch 38/50 | train loss: 0.015 | val loss: 0.496 | train acc: 99.6% | val acc: 88.8% | current lr: 0.001
Epoch 39/50 | train loss: 0.027 | val loss: 0.491 | train acc: 99.1% | val acc: 90.2% | current lr: 0.001
Epoch 40/50 | train loss: 0.026 | val loss: 0.704 | train acc: 99.2% | val acc: 87.8% | current lr: 0.001
Epoch 41/50 | train loss: 0.006 | val loss: 0.537 | train acc: 99.8% | val acc: 89.4% | current lr: 0.001
Epoch 42/50 | train loss: 0.004 | val loss: 0.557 | train acc: 99.8% | val acc: 90.0% | current lr: 0.001
Epoch 43/50 | train loss: 0.007 | val loss: 0.494 | train acc: 99.8% | val acc: 90.5% | current lr: 0.001
Epoch 44/50 | train loss: 0.004 | val loss: 0.551 | train acc: 99.9% | val acc: 90.2% | current lr: 0.001
Epoch 45/50 | train loss: 0.020 | val loss: 0.572 | train acc: 99.3% | val acc: 90.2% | current lr: 0.001
Epoch 46/50 | train loss: 0.034 | val loss: 0.605 | train acc: 99.0% | val acc: 90.0% | current lr: 0.001
Epoch 47/50 | train loss: 0.017 | val loss: 0.573 | train acc: 99.4% | val acc: 89.8% | current lr: 0.001
Epoch 48/50 | train loss: 0.031 | val loss: 0.573 | train acc: 98.8% | val acc: 90.0% | current lr: 0.001
Epoch 49/50 | train loss: 0.037 | val loss: 0.555 | train acc: 98.6% | val acc: 90.9% | current lr: 0.001
✅Epoch 50/50 | train loss: 0.009 | val loss: 0.549 | train acc: 99.8% | val acc: 91.0% | current lr: 0.001=>Gap=8.8%
Training time:532.67 seconds
```

</details>

## MobileNetV3-Large — Last Block (3) Fine-Tuning (Mean/Std = 0.5)

- **Best validation accuracy:** 97.1% (epoch 29)
- **Final validation accuracy:** 96.4%
- **Training time:** 593.73 seconds (9.90 minutes)
- **Total parameters:** 4,212,280
- **Trainable parameters:** 2,990,568

<details>
<summary>Full training log</summary>

```text
Total parameters: 4,212,280
Trainable parameters: 2,990,568
Epoch 01/50 | train loss: 0.460 | val loss: 0.286 | train acc: 84.9% | val acc: 91.0% | current lr: 0.001
Epoch 02/50 | train loss: 0.098 | val loss: 0.207 | train acc: 97.1% | val acc: 94.7% | current lr: 0.001
Epoch 03/50 | train loss: 0.077 | val loss: 0.299 | train acc: 97.4% | val acc: 92.9% | current lr: 0.001
Epoch 04/50 | train loss: 0.074 | val loss: 1.182 | train acc: 97.7% | val acc: 88.0% | current lr: 0.001
Epoch 05/50 | train loss: 0.043 | val loss: 0.175 | train acc: 98.6% | val acc: 96.3% | current lr: 0.001
Epoch 06/50 | train loss: 0.015 | val loss: 0.180 | train acc: 99.6% | val acc: 95.5% | current lr: 0.001
Epoch 07/50 | train loss: 0.016 | val loss: 0.232 | train acc: 99.5% | val acc: 95.6% | current lr: 0.001
Epoch 08/50 | train loss: 0.034 | val loss: 0.326 | train acc: 99.1% | val acc: 93.6% | current lr: 0.001
Epoch 09/50 | train loss: 0.046 | val loss: 0.441 | train acc: 98.8% | val acc: 93.2% | current lr: 0.001
Epoch 10/50 | train loss: 0.039 | val loss: 0.277 | train acc: 98.9% | val acc: 94.0% | current lr: 0.001
Epoch 11/50 | train loss: 0.018 | val loss: 0.313 | train acc: 99.5% | val acc: 95.1% | current lr: 0.001
Epoch 12/50 | train loss: 0.007 | val loss: 0.250 | train acc: 99.8% | val acc: 96.7% | current lr: 0.001
Epoch 13/50 | train loss: 0.011 | val loss: 0.178 | train acc: 99.6% | val acc: 96.1% | current lr: 0.001
Epoch 14/50 | train loss: 0.005 | val loss: 0.254 | train acc: 99.8% | val acc: 96.8% | current lr: 0.001
Epoch 15/50 | train loss: 0.006 | val loss: 0.232 | train acc: 99.9% | val acc: 96.4% | current lr: 0.001
Epoch 16/50 | train loss: 0.005 | val loss: 0.379 | train acc: 99.8% | val acc: 95.5% | current lr: 0.001
Epoch 17/50 | train loss: 0.033 | val loss: 0.313 | train acc: 99.3% | val acc: 94.0% | current lr: 0.001
Epoch 18/50 | train loss: 0.007 | val loss: 0.405 | train acc: 99.7% | val acc: 95.1% | current lr: 0.001
Epoch 19/50 | train loss: 0.024 | val loss: 0.419 | train acc: 99.4% | val acc: 94.1% | current lr: 0.001
Epoch 20/50 | train loss: 0.063 | val loss: 0.328 | train acc: 98.3% | val acc: 94.0% | current lr: 0.001
Epoch 21/50 | train loss: 0.036 | val loss: 0.323 | train acc: 99.0% | val acc: 95.6% | current lr: 0.001
Epoch 22/50 | train loss: 0.007 | val loss: 0.247 | train acc: 99.7% | val acc: 95.7% | current lr: 0.001
Epoch 23/50 | train loss: 0.010 | val loss: 0.355 | train acc: 99.7% | val acc: 94.8% | current lr: 0.001
Epoch 24/50 | train loss: 0.014 | val loss: 0.346 | train acc: 99.5% | val acc: 92.5% | current lr: 0.001
Epoch 25/50 | train loss: 0.010 | val loss: 0.264 | train acc: 99.6% | val acc: 95.7% | current lr: 0.001
Epoch 26/50 | train loss: 0.009 | val loss: 0.236 | train acc: 99.8% | val acc: 95.9% | current lr: 0.001
Epoch 27/50 | train loss: 0.008 | val loss: 0.319 | train acc: 99.6% | val acc: 95.3% | current lr: 0.001
Epoch 28/50 | train loss: 0.014 | val loss: 0.319 | train acc: 99.7% | val acc: 95.1% | current lr: 0.001
✅Epoch 29/50 | train loss: 0.032 | val loss: 0.249 | train acc: 99.2% | val acc: 97.1% | current lr: 0.001=> Gap=2.1%
Epoch 30/50 | train loss: 0.020 | val loss: 0.254 | train acc: 99.6% | val acc: 96.3% | current lr: 0.001
Epoch 31/50 | train loss: 0.043 | val loss: 0.636 | train acc: 99.4% | val acc: 90.9% | current lr: 0.001
Epoch 32/50 | train loss: 0.039 | val loss: 0.294 | train acc: 99.2% | val acc: 96.0% | current lr: 0.001
Epoch 33/50 | train loss: 0.005 | val loss: 0.242 | train acc: 100.0% | val acc: 96.9% | current lr: 0.001
Epoch 34/50 | train loss: 0.001 | val loss: 0.232 | train acc: 100.0% | val acc: 96.5% | current lr: 0.001
Epoch 35/50 | train loss: 0.001 | val loss: 0.285 | train acc: 100.0% | val acc: 96.5% | current lr: 0.001
Epoch 36/50 | train loss: 0.000 | val loss: 0.264 | train acc: 100.0% | val acc: 96.8% | current lr: 0.001
Epoch 37/50 | train loss: 0.000 | val loss: 0.249 | train acc: 100.0% | val acc: 96.8% | current lr: 0.001
Epoch 38/50 | train loss: 0.004 | val loss: 0.246 | train acc: 99.9% | val acc: 94.8% | current lr: 0.001
Epoch 39/50 | train loss: 0.046 | val loss: 0.260 | train acc: 98.8% | val acc: 95.2% | current lr: 0.001
Epoch 40/50 | train loss: 0.012 | val loss: 0.365 | train acc: 99.6% | val acc: 96.0% | current lr: 0.001
Epoch 41/50 | train loss: 0.020 | val loss: 0.343 | train acc: 99.5% | val acc: 95.9% | current lr: 0.001
Epoch 42/50 | train loss: 0.007 | val loss: 0.341 | train acc: 99.9% | val acc: 95.7% | current lr: 0.001
Epoch 43/50 | train loss: 0.005 | val loss: 0.346 | train acc: 99.8% | val acc: 95.3% | current lr: 0.001
Epoch 44/50 | train loss: 0.009 | val loss: 0.348 | train acc: 99.7% | val acc: 96.0% | current lr: 0.001
Epoch 45/50 | train loss: 0.020 | val loss: 0.627 | train acc: 99.8% | val acc: 90.5% | current lr: 0.001
Epoch 46/50 | train loss: 0.024 | val loss: 0.273 | train acc: 99.4% | val acc: 95.7% | current lr: 0.001
Epoch 47/50 | train loss: 0.009 | val loss: 0.192 | train acc: 99.8% | val acc: 96.8% | current lr: 0.001
Epoch 48/50 | train loss: 0.003 | val loss: 0.184 | train acc: 99.9% | val acc: 96.8% | current lr: 0.001
Epoch 49/50 | train loss: 0.001 | val loss: 0.199 | train acc: 100.0% | val acc: 97.1% | current lr: 0.001
Epoch 50/50 | train loss: 0.001 | val loss: 0.202 | train acc: 100.0% | val acc: 96.4% | current lr: 0.001
Training time:593.73 seconds
```

</details>

## Depthwise Model — Dropout = 0.2

- **Best validation accuracy:** 88.6% (epoch 45)
- **Final validation accuracy:** 80.1%
- **Training time:** 1531.40 seconds (25.52 minutes)

<details>
<summary>Full training log</summary>

```text
cuda
73,033 params
Epoch 01/50 | train loss: 1.724 | val loss: 1.417 | train acc: 38.7% | val acc: 51.6% | current lr: 0.001
Epoch 02/50 | train loss: 1.257 | val loss: 1.204 | train acc: 56.7% | val acc: 54.8% | current lr: 0.001
Epoch 03/50 | train loss: 0.999 | val loss: 0.976 | train acc: 65.9% | val acc: 66.2% | current lr: 0.001
Epoch 04/50 | train loss: 0.823 | val loss: 0.899 | train acc: 72.4% | val acc: 69.3% | current lr: 0.001
Epoch 05/50 | train loss: 0.713 | val loss: 0.775 | train acc: 76.4% | val acc: 71.9% | current lr: 0.001
Epoch 06/50 | train loss: 0.592 | val loss: 0.884 | train acc: 80.2% | val acc: 69.5% | current lr: 0.001
Epoch 07/50 | train loss: 0.534 | val loss: 0.738 | train acc: 82.1% | val acc: 72.2% | current lr: 0.001
Epoch 08/50 | train loss: 0.461 | val loss: 0.730 | train acc: 84.6% | val acc: 74.1% | current lr: 0.001
Epoch 09/50 | train loss: 0.387 | val loss: 0.594 | train acc: 87.1% | val acc: 78.7% | current lr: 0.001
Epoch 10/50 | train loss: 0.364 | val loss: 0.550 | train acc: 88.1% | val acc: 79.5% | current lr: 0.001
Epoch 11/50 | train loss: 0.327 | val loss: 0.497 | train acc: 89.0% | val acc: 82.6% | current lr: 0.001
Epoch 12/50 | train loss: 0.275 | val loss: 0.692 | train acc: 92.1% | val acc: 76.9% | current lr: 0.001
Epoch 13/50 | train loss: 0.242 | val loss: 0.628 | train acc: 92.8% | val acc: 78.7% | current lr: 0.001
Epoch 14/50 | train loss: 0.231 | val loss: 0.514 | train acc: 93.2% | val acc: 82.5% | current lr: 0.001
Epoch 15/50 | train loss: 0.200 | val loss: 0.518 | train acc: 93.8% | val acc: 81.7% | current lr: 0.001
Epoch 16/50 | train loss: 0.168 | val loss: 0.491 | train acc: 95.5% | val acc: 83.6% | current lr: 0.001
Epoch 17/50 | train loss: 0.158 | val loss: 0.443 | train acc: 95.7% | val acc: 84.4% | current lr: 0.001
Epoch 18/50 | train loss: 0.152 | val loss: 0.482 | train acc: 95.6% | val acc: 83.0% | current lr: 0.001
Epoch 19/50 | train loss: 0.158 | val loss: 0.641 | train acc: 95.5% | val acc: 80.3% | current lr: 0.001
Epoch 20/50 | train loss: 0.156 | val loss: 0.548 | train acc: 95.5% | val acc: 81.7% | current lr: 0.001
Epoch 21/50 | train loss: 0.119 | val loss: 0.602 | train acc: 97.1% | val acc: 80.1% | current lr: 0.001
Epoch 22/50 | train loss: 0.123 | val loss: 0.513 | train acc: 96.6% | val acc: 82.0% | current lr: 0.001
Epoch 23/50 | train loss: 0.107 | val loss: 0.513 | train acc: 96.9% | val acc: 84.1% | current lr: 0.001
Epoch 24/50 | train loss: 0.139 | val loss: 0.703 | train acc: 95.3% | val acc: 80.2% | current lr: 0.001
Epoch 25/50 | train loss: 0.092 | val loss: 0.647 | train acc: 97.8% | val acc: 79.7% | current lr: 0.001
Epoch 26/50 | train loss: 0.072 | val loss: 0.395 | train acc: 98.4% | val acc: 86.8% | current lr: 0.001
Epoch 27/50 | train loss: 0.068 | val loss: 0.560 | train acc: 98.3% | val acc: 84.2% | current lr: 0.001
Epoch 28/50 | train loss: 0.068 | val loss: 0.441 | train acc: 98.4% | val acc: 85.7% | current lr: 0.001
Epoch 29/50 | train loss: 0.069 | val loss: 1.371 | train acc: 98.2% | val acc: 67.2% | current lr: 0.001
Epoch 30/50 | train loss: 0.060 | val loss: 0.545 | train acc: 98.6% | val acc: 84.8% | current lr: 0.001
Epoch 31/50 | train loss: 0.066 | val loss: 0.578 | train acc: 98.3% | val acc: 82.0% | current lr: 0.001
Epoch 32/50 | train loss: 0.077 | val loss: 0.588 | train acc: 97.9% | val acc: 82.5% | current lr: 0.001
Epoch 33/50 | train loss: 0.071 | val loss: 0.501 | train acc: 98.3% | val acc: 84.9% | current lr: 0.001
Epoch 34/50 | train loss: 0.066 | val loss: 0.621 | train acc: 98.2% | val acc: 83.2% | current lr: 0.001
Epoch 35/50 | train loss: 0.069 | val loss: 0.804 | train acc: 98.3% | val acc: 76.7% | current lr: 0.001
Epoch 36/50 | train loss: 0.058 | val loss: 0.914 | train acc: 98.6% | val acc: 75.7% | current lr: 0.001
Epoch 37/50 | train loss: 0.051 | val loss: 0.632 | train acc: 98.6% | val acc: 81.0% | current lr: 0.001
Epoch 38/50 | train loss: 0.080 | val loss: 1.004 | train acc: 97.3% | val acc: 73.9% | current lr: 0.001
Epoch 39/50 | train loss: 0.085 | val loss: 0.464 | train acc: 97.5% | val acc: 86.9% | current lr: 0.001
Epoch 40/50 | train loss: 0.050 | val loss: 0.403 | train acc: 98.6% | val acc: 87.4% | current lr: 0.001
Epoch 41/50 | train loss: 0.054 | val loss: 0.443 | train acc: 98.6% | val acc: 86.9% | current lr: 0.001
Epoch 42/50 | train loss: 0.045 | val loss: 0.760 | train acc: 98.9% | val acc: 79.5% | current lr: 0.001
Epoch 43/50 | train loss: 0.082 | val loss: 0.547 | train acc: 97.3% | val acc: 84.6% | current lr: 0.001
Epoch 44/50 | train loss: 0.037 | val loss: 0.423 | train acc: 99.2% | val acc: 87.3% | current lr: 0.001
✅Epoch 45/50 | train loss: 0.026 | val loss: 0.382 | train acc: 99.5% | val acc: 88.6% | current lr: 0.001=>Gap=10.9%
Epoch 46/50 | train loss: 0.045 | val loss: 0.530 | train acc: 98.8% | val acc: 84.5% | current lr: 0.001
Epoch 47/50 | train loss: 0.060 | val loss: 0.527 | train acc: 98.3% | val acc: 85.3% | current lr: 0.001
Epoch 48/50 | train loss: 0.042 | val loss: 0.527 | train acc: 99.0% | val acc: 84.9% | current lr: 0.001
Epoch 49/50 | train loss: 0.056 | val loss: 0.747 | train acc: 98.4% | val acc: 83.0% | current lr: 0.001
Epoch 50/50 | train loss: 0.049 | val loss: 0.747 | train acc: 98.8% | val acc: 80.1% | current lr: 0.001
Training time: 1531.40 seconds
```

</details>

## Depthwise Model — Dropout = 0.5

- **Best validation accuracy:** 88.8% (epoch 45)
- **Final validation accuracy:** 79.0%
- **Training time:** 1521.46 seconds (25.36 minutes)

<details>
<summary>Full training log</summary>

```text
Epoch 01/50 | train loss: 1.793 | val loss: 1.520 | train acc: 34.9% | val acc: 50.3% | current lr: 0.001
Epoch 02/50 | train loss: 1.354 | val loss: 1.195 | train acc: 52.9% | val acc: 56.0% | current lr: 0.001
Epoch 03/50 | train loss: 1.118 | val loss: 1.036 | train acc: 61.6% | val acc: 62.8% | current lr: 0.001
Epoch 04/50 | train loss: 0.974 | val loss: 0.972 | train acc: 66.8% | val acc: 66.0% | current lr: 0.001
Epoch 05/50 | train loss: 0.836 | val loss: 0.896 | train acc: 71.3% | val acc: 67.5% | current lr: 0.001
Epoch 06/50 | train loss: 0.714 | val loss: 1.174 | train acc: 75.7% | val acc: 59.4% | current lr: 0.001
Epoch 07/50 | train loss: 0.664 | val loss: 0.729 | train acc: 76.5% | val acc: 72.2% | current lr: 0.001
Epoch 08/50 | train loss: 0.586 | val loss: 0.737 | train acc: 79.2% | val acc: 73.0% | current lr: 0.001
Epoch 09/50 | train loss: 0.520 | val loss: 0.567 | train acc: 82.3% | val acc: 80.3% | current lr: 0.001
Epoch 10/50 | train loss: 0.500 | val loss: 0.535 | train acc: 82.5% | val acc: 82.2% | current lr: 0.001
Epoch 11/50 | train loss: 0.449 | val loss: 0.515 | train acc: 84.2% | val acc: 82.8% | current lr: 0.001
Epoch 12/50 | train loss: 0.395 | val loss: 0.690 | train acc: 87.1% | val acc: 76.3% | current lr: 0.001
Epoch 13/50 | train loss: 0.373 | val loss: 0.693 | train acc: 87.8% | val acc: 75.1% | current lr: 0.001
Epoch 14/50 | train loss: 0.364 | val loss: 0.480 | train acc: 87.9% | val acc: 83.8% | current lr: 0.001
Epoch 15/50 | train loss: 0.305 | val loss: 0.445 | train acc: 90.1% | val acc: 84.0% | current lr: 0.001
Epoch 16/50 | train loss: 0.288 | val loss: 0.551 | train acc: 90.4% | val acc: 80.6% | current lr: 0.001
Epoch 17/50 | train loss: 0.275 | val loss: 0.458 | train acc: 91.2% | val acc: 84.1% | current lr: 0.001
Epoch 18/50 | train loss: 0.252 | val loss: 0.467 | train acc: 92.5% | val acc: 83.6% | current lr: 0.001
Epoch 19/50 | train loss: 0.243 | val loss: 0.596 | train acc: 92.2% | val acc: 80.7% | current lr: 0.001
Epoch 20/50 | train loss: 0.233 | val loss: 0.415 | train acc: 93.3% | val acc: 86.0% | current lr: 0.001
Epoch 21/50 | train loss: 0.224 | val loss: 0.422 | train acc: 92.7% | val acc: 85.4% | current lr: 0.001
Epoch 22/50 | train loss: 0.208 | val loss: 0.439 | train acc: 93.8% | val acc: 85.7% | current lr: 0.001
Epoch 23/50 | train loss: 0.188 | val loss: 0.465 | train acc: 93.8% | val acc: 85.7% | current lr: 0.001
Epoch 24/50 | train loss: 0.188 | val loss: 0.410 | train acc: 94.0% | val acc: 87.4% | current lr: 0.001
Epoch 25/50 | train loss: 0.167 | val loss: 0.403 | train acc: 95.3% | val acc: 86.8% | current lr: 0.001
Epoch 26/50 | train loss: 0.143 | val loss: 0.520 | train acc: 95.7% | val acc: 81.0% | current lr: 0.001
Epoch 27/50 | train loss: 0.126 | val loss: 0.397 | train acc: 96.9% | val acc: 86.5% | current lr: 0.001
Epoch 28/50 | train loss: 0.142 | val loss: 0.489 | train acc: 95.7% | val acc: 84.4% | current lr: 0.001
Epoch 29/50 | train loss: 0.135 | val loss: 0.419 | train acc: 95.7% | val acc: 86.9% | current lr: 0.001
Epoch 30/50 | train loss: 0.138 | val loss: 0.498 | train acc: 95.7% | val acc: 84.0% | current lr: 0.001
Epoch 31/50 | train loss: 0.115 | val loss: 0.387 | train acc: 97.0% | val acc: 88.0% | current lr: 0.001
Epoch 32/50 | train loss: 0.122 | val loss: 0.484 | train acc: 96.2% | val acc: 85.2% | current lr: 0.001
Epoch 33/50 | train loss: 0.125 | val loss: 0.365 | train acc: 96.0% | val acc: 88.4% | current lr: 0.001
Epoch 34/50 | train loss: 0.112 | val loss: 0.457 | train acc: 96.9% | val acc: 86.9% | current lr: 0.001
Epoch 35/50 | train loss: 0.135 | val loss: 0.429 | train acc: 95.9% | val acc: 86.1% | current lr: 0.001
Epoch 36/50 | train loss: 0.104 | val loss: 0.481 | train acc: 96.6% | val acc: 86.2% | current lr: 0.001
Epoch 37/50 | train loss: 0.089 | val loss: 0.466 | train acc: 97.5% | val acc: 85.7% | current lr: 0.001
Epoch 38/50 | train loss: 0.121 | val loss: 0.446 | train acc: 95.8% | val acc: 85.8% | current lr: 0.001
Epoch 39/50 | train loss: 0.121 | val loss: 0.386 | train acc: 96.1% | val acc: 88.2% | current lr: 0.001
Epoch 40/50 | train loss: 0.101 | val loss: 0.436 | train acc: 97.1% | val acc: 86.1% | current lr: 0.001
Epoch 41/50 | train loss: 0.091 | val loss: 0.460 | train acc: 96.9% | val acc: 87.3% | current lr: 0.001
Epoch 42/50 | train loss: 0.070 | val loss: 0.394 | train acc: 98.1% | val acc: 87.8% | current lr: 0.001
Epoch 43/50 | train loss: 0.088 | val loss: 0.467 | train acc: 97.3% | val acc: 85.3% | current lr: 0.001
Epoch 44/50 | train loss: 0.080 | val loss: 0.459 | train acc: 97.6% | val acc: 85.3% | current lr: 0.001
✅Epoch 45/50 | train loss: 0.082 | val loss: 0.369 | train acc: 97.7% | val acc: 88.8% | current lr: 0.001=>Gap=8.9%
Epoch 46/50 | train loss: 0.093 | val loss: 0.418 | train acc: 97.4% | val acc: 87.4% | current lr: 0.001
Epoch 47/50 | train loss: 0.081 | val loss: 0.442 | train acc: 97.5% | val acc: 87.3% | current lr: 0.001
Epoch 48/50 | train loss: 0.082 | val loss: 0.387 | train acc: 97.6% | val acc: 88.6% | current lr: 0.001
Epoch 49/50 | train loss: 0.075 | val loss: 0.425 | train acc: 97.7% | val acc: 87.0% | current lr: 0.001
Epoch 50/50 | train loss: 0.085 | val loss: 0.709 | train acc: 97.4% | val acc: 79.0% | current lr: 0.001
Training time: 1521.46 seconds
```

</details>

