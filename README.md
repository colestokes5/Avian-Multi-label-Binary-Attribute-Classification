# Avian Multi-label Binary Attribute Classification
This project focuses on multi-label binary attribute classification for birds using the CUB-200-2011 dataset. The objective is to identify which of 312 attributes (e.g., wing color, beak size, etc.) are present in an image of a bird. The project employs deep learning models for image classification and attribute prediction.

## Introduction
Birds possess a rich variety of visual attributes that can be used to distinguish between species or understand their behaviors. This project leverages the CUB-200-2011 dataset to train a model that predicts multiple attributes for each bird image.

### Key Features:
Multi-label classification of 312 attributes.
Customizable model architecture for experimentation.
Evaluation metrics include F1-score, precision, and recall.

## Dataset
The CUB-200-2011 (Caltech-UCSD Birds-200-2011) dataset is a widely used benchmark for fine-grained visual classification tasks.
Link: [CUB-200-2011 Dataset](https://www.vision.caltech.edu/datasets/cub_200_2011/)
Description: Contains 11,788 images of 200 bird species with bounding boxes, part annotations, and attribute labels.

## Setup
1. Download the CUB-200-2011 dataset into the main directory.
2. Run compile_cub_attributes.py (make sure you get a CSV).
3. Experiement with the model.
