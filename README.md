# Artwork Multi-Label Classification

## Project Overview
This project focuses on multi-label classification of artwork images, aiming to predict multiple attributes such as artist, genre, and style. The pipeline utilizes deep learning and machine learning techniques for efficient classification.

## Technologies Used
- **Deep Learning Framework**: TensorFlow, Keras
- **Pre-trained Model**: ResNet50 for feature extraction
- **Dimensionality Reduction**: PCA (Principal Component Analysis)
- **Anomaly Detection**: Isolation Forest
- **Data Processing**: NumPy, Pandas, TensorFlow datasets

## Dataset
The dataset consists of artwork images annotated with multiple labels (artist, genre, and style). It is preprocessed using TensorFlow datasets.

## Workflow
### 1. Feature Extraction
- ResNet50 is used as a feature extractor by removing its top classification layer.
- Image features are extracted and stored for further processing.

### 2. Dimensionality Reduction
- PCA is applied to reduce feature dimensions while preserving essential information.

### 3. Outlier Detection
- Isolation Forest is employed to identify and remove outliers from the dataset.

### 4. Multi-Label Classification
- The processed features are fed into a machine learning model for multi-label classification.
- Various models are evaluated for accuracy and efficiency.
### 5 . Evaluation Metrics:
- Accuracy - Measures how often your model correctly classifies a painting into the correct genre or artist. However, accuracy may not be the best metric if the dataset is imbalanced.

## Clone and Setup

 **Clone the repository**:
```bash
git clone https://github.com/hillhack/HumanaAI-task.git
```
```bash
cd HumanaAI-task
```
## Installation
To set up the environment, install the required dependencies:
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

## Future Improvements
- Implement a more advanced deep learning model (e.g., Vision Transformer)
- Experiment with different feature extraction techniques
- Optimize training with data augmentation and transfer learning


