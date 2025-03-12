
# YOLO Object Detection

This repository contains a YOLO (You Only Look Once) model for object detection. The model is trained using the **coco8 dataset** and performs object detection on images, classifying objects into predefined categories.

---

## Overview

The YOLO model is an object detection algorithm that predicts bounding boxes and class labels for objects within an image. This implementation uses a pre-trained YOLO model and fine-tunes it on the **coco8 dataset**, a subset of the popular COCO dataset.

### Key Features:
- **Model**: YOLOv11 (or similar YOLO architecture).
- **Dataset**: coco8 (a subset of the COCO dataset for object detection tasks).
- **Task**: Detects objects in images (e.g., cats, dogs, etc.) and predicts their bounding boxes.
- **Training**: Trained using PyTorch and fine-tuned on the coco8 dataset.
- **Inference**: Perform object detection on test images (cat_test.jpg or dog_test.jpg).

---

## Requirements

- Python 3.x
- PyTorch
- Ultralyics YOLO
- CUDA (Optional, for GPU acceleration)

---

## Installation

1. Clone the repository or download the code.
2. Install the required dependencies by running:

   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a CUDA-compatible GPU and necessary drivers installed for GPU acceleration. If you don't have a GPU, the script will automatically run on CPU.

---

## Dataset

The model is trained using the **coco8 dataset**. The coco8 dataset is a subset of the COCO dataset, which contains images with various objects (e.g., people, animals, vehicles).

---


### Notes:
- The training data is sourced from the **coco8.yaml** configuration file.
- Test images for inference (`<test_image>.jpg`) is used to evaluate the trained model.

---

## Training

The model is trained using the **coco8.yaml** dataset configuration, and the process includes:

- Loading the dataset from the coco8.yaml file.
- Training for 5 epochs.
- Saving the trained model weights after completion.

---

## Inference

After training the model, you can perform object detection on new images. This can be done using the following command:

```python
results = model("./images/<test_image>.jpg") where <test_image> is a test image
```

The model will detect objects in the image and output the bounding boxes along with the predicted labels.

---

## Licensing

### YOLO Model

This model is based on the YOLO object detection framework, which was originally proposed in the paper:

- **Paper**: [YOLO: You Only Look Once](https://arxiv.org/abs/1506.02640)

This model uses the Ultralytics YOLO implementation, which is licensed under the [MIT License](https://opensource.org/licenses/MIT).

### COCO Dataset

The COCO dataset YAML file is obtained from:

- **COCO8 Dataset**: [COCO8 YAML] https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/datasets/coco8.yaml

The citation for the COCO dataset is the following paper:

- **COCO Paper**: [COCO Paper](https://arxiv.org/abs/1405.0312)
