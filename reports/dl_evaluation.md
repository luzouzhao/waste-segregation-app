# DL Evaluation Report

**Model:** EfficientNet-B0 (fine-tuned)

**Test set size:** 3479

## Overall Accuracy

**0.8649**

## Per-Class F1-Score

| Class | F1-Score |
|-------|----------|
| electronic | 0.9493 |
| glass | 0.8502 |
| hazardous | 0.8320 |
| metal | 0.7722 |
| organic | 0.9439 |
| paper | 0.8783 |
| plastic | 0.7851 |

## Confusion Matrix

| ¡ýTrue \ Pred¡ú | electronic | glass | hazardous | metal | organic | paper | plastic |
|---|---|---|---|---|---|---|---|
| **electronic** | 440 | 0 | 1 | 4 | 3 | 2 | 0 |
| **glass** | 3 | 261 | 13 | 9 | 2 | 2 | 12 |
| **hazardous** | 2 | 20 | 510 | 41 | 5 | 11 | 20 |
| **metal** | 4 | 5 | 18 | 261 | 0 | 6 | 12 |
| **organic** | 7 | 5 | 12 | 11 | 572 | 6 | 5 |
| **paper** | 10 | 9 | 25 | 16 | 4 | 574 | 40 |
| **plastic** | 11 | 12 | 38 | 28 | 8 | 28 | 391 |
