# 42Amman Face Recognition - Hackathon Project

This notebook implements a simple face recognition system using the FastAI library. It trains a convolutional neural network to identify whether a test image matches a known face from the training dataset.

---

## Contents

- Data verification and cleaning
- Data loading using FastAI's DataBlock API
- Model training using `cnn_learner` with ResNet34
- Model saving and exporting
- Inference on test images
- Visualization of predictions and confidence

---

## Steps Performed

### 1. Image Verification

- Used `verify_images` to detect and remove corrupted image files from the training directory.

```python
failed = verify_images(get_image_files(path))
failed.map(Path.unlink)
```

### 2. DataBlock Setup
Defined a FastAI DataBlock that:
Uses images as input
Uses parent folder name as label
Applies item and batch transforms
Resizes all images to 224x224
Created DataLoaders with batch size of 32.

### 3. Model Training
Initialized a cnn_learner with resnet34 as the backbone.

Trained the model for 2 epochs using fine_tune(2).

learn = cnn_learner(dls, resnet34, metrics=accuracy)
learn.fine_tune(2)
### 4. Model Export
Exported the trained model as 42hackathon.pkl.

learn.export('42hackathon.pkl')
### 5. Inference
Loaded the model using load_learner.

Predicted labels for images in the /test folder using learn.predict.

pred_class, pred_idx, outputs = learn.predict(img)
Printed:

Ground truth label (from file path)

Predicted label

Confidence score

Displayed test image with title showing the prediction.

Output Example
Ground Truth: test01
Prediction  : test01
Confidence  : 0.998
