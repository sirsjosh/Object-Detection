# Car Detection Project

This project involves training a classifier to detect cars in images and applying the classifier to a video stream. The pipeline includes steps such as feature extraction, training a linear SVM classifier, and processing video frames.

## Project Structure

- `train_classify.py`: Script for training the SVM classifier.
- `lesson_functions.py`: Module containing functions for feature extraction, sliding window approach, and heatmap processing.
- `video.py`: Script for processing video frames using the trained classifier.
- `svm_model.pkl`: Saved SVM classifier model.
- `scaler.pkl`: Saved feature scaler.

## Usage

### Training the Classifier

1. Run `train_classify.py` to train the SVM classifier and save the model.

```bash
python train_classify.py
```

### Processing Video Frames
Run `video.py` to process a video using the trained classifier.

```bash
python video.py
```
