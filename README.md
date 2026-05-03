# project_soumya_ranjan_pandit
# Brain Tumor Classification using Deep Learning

## Overview

This project performs **binary classification** of brain MRI images into:

* Tumor
* No Tumor

It uses a pretrained **MobileNetV2** model fine-tuned for classification.

---

##  Project Structure

```
project_soumya_ranjan_pandit/
│
├── checkpoints/
│   └── final_weights.pth
│
├── data/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
├── dataset.py
├── model.py
├── train.py
├── predict.py
├── interface.py
├── config.py
```

---

## How to Run

### Train the model

```
python train.py
```

### Run prediction

```
python predict.py
```

---

## Model

* MobileNetV2 (pretrained)
* Final layer modified for binary classification

---

## Configuration

Defined in `config.py`:

* Batch size
* Epochs
* Image size
* Learning rate

---

## Dataset

### Data Source

https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset

### Details

* ~10,000 images
* Classes:

  * Glioma
  * Meningioma
  * Pituitary
  * No Tumor

### Processing

* Tumor classes → **Tumor (1)**
* No Tumor → **No Tumor (0)**

---

## Interface

The `interface.py` maps:

* Model → `TheModel`
* Trainer → `the_trainer`
* Predictor → `the_predictor`
* Dataset → `TheDataset`
* Dataloader → `the_dataloader`

Used for automated evaluation.

---

## Output

* Tumor / No Tumor
* Confidence score
