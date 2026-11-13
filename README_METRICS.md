# Model Evaluation Results

- **Model:** MobileNetV2 Transfer Learning
- **Test Split Samples:** 855 images
- **Overall Accuracy:** 96.96%

### Confusion Matrix
| Actual vs Predicted | Predicted Human | Predicted Non-Human |
|:---|:---:|:---:|
| **Actual Human** | 415 | 18 |
| **Actual Non-Human** | 8 | 414 |

### Classification Report
```text
              precision    recall  f1-score   support

       human     0.9811    0.9584    0.9696       433
   non_human     0.9583    0.9810    0.9696       422

    accuracy                         0.9696       855
   macro avg     0.9697    0.9697    0.9696       855
weighted avg     0.9699    0.9696    0.9696       855

```