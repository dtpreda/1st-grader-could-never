## Test Loss

|| Randomly Initialized Model | Pretext Trained Model |
| --- | --- | --- |
| Frozen Encoder | 2.301620 | 2.303379 |
| Unfrozen Encoder | 1.827206 | 1.857588 |

10 epochs, CrossEntropyLoss, lr=1e-3, Adam optimizer, 512 batch size