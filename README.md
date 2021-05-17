# Transfer model for Binary Classification (for cat vs dog image recognition), 
# using MobileNetV2 pre-trained on ImageNet dataset as base model.


**Python packages used for data preprocessing and creating of the model:** TensorFlow.

**Input:** dataset of images, labeled as 1/0 (cat / dog). 240 training and 60 test images.

**Data preprocessing:**
- Resizing: into 160px * 160px images – so that they fit MobileNetV2 model That is 12 288 features in total for each image.
- Batching: BATCH_SIZE = 32
**Base model:** MobileNetV2 pre-trained on ImageNet dataset
**Transfer model (catdog_model):**
Transfer model’s layers:
- Input layer
- MobileNetV2 model (without the top 1000-classes classification layer)
- Global Average Pooling layer (to summarize the info in each channel)
- Dropout layer (to avoid overfitting)
- Prediction layer (for binary classification)

Hyperparameters of the Neural Network:
- Optimization algorithm: Adam
- Learning rate: 0.01
- Number of epochs: 3

Accuracy: 96.67% on the validation dataset

**Fine-tuning:**
- 7 last layers of the base MobileNetV2 model were unfreezed and re-trained
- Learning rate: changed to 0.001
- Number of additional epochs: 2

**Result:** prediction accuracy after fine-tuning – 100% on the validation dataset.
