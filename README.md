# project_soumya_ranjan_pandit
Brain Tumor Classification using Deep Learning
Overview
This project performs binary classification of brain MRI images into:
    • Tumor
    • No Tumor
It uses a pretrained MobileNetV2 model fine-tuned for classification.
Project Structure
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
How to Run
1. Train the model
python train.py
2. Run prediction
python predict.py
Model
    • MobileNetV2 (pretrained)
    • Final layer modified for binary classification
Configuration
Defined in config.py:
    • Batch size
    • Epochs
    • Image size
    • Learning rate
Dataset
Data Sources
The dataset used in this project is a Brain Tumor MRI dataset available on Kaggle, which combines images from multiple sources.
Dataset link: https://www.kaggle.com/datasets/ishans24/brain-tumor-dataset 
Dataset Details:
    • Around 10,000 images
    • Classes:
        ◦ Glioma
        ◦ Meningioma
        ◦ Pituitary
        ◦ No Tumor
Processing:
    • All tumor types (Glioma, Meningioma, Pituitary) are combined into a single class:
        ◦ Tumor (1)
    • No Tumor images are labeled as:
        ◦ No Tumor (0)
    • The dataset is already labeled and suitable for training deep learning models.

Interface
The interface.py file maps:
    • Model → TheModel
    • Trainer → the_trainer
    • Predictor → the_predictor
    • Dataset → TheDataset
    • Dataloader → the_dataloader
This is used for automated evaluation.
Output
The model predicts:
    • Tumor / No Tumor
    • Confidence score
