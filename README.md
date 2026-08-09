# Aircraft Image Classification

AeroVision is a computer vision application that identifies the type of aircraft present in an uploaded image using a deep learning image classification model.

The project covers the complete machine learning workflow, from dataset preparation and model training to model evaluation, inference, and deployment through a FastAPI backend.

## Overview

Aircraft recognition is an image classification problem where a model learns visual patterns from labeled aircraft images and uses those patterns to classify previously unseen images.

Given an input image:

```text
Aircraft Image
      |
      v
Image Preprocessing
      |
      v
Deep Learning Model
      |
      v
Aircraft Classification
      |
      v
Top Predictions + Confidence Scores
```

Example:

```json
{
  "aircraft": "Boeing 737",
  "confidence": 0.942,
  "predictions": [
    {
      "aircraft": "Boeing 737",
      "confidence": 0.942
    },
    {
      "aircraft": "Airbus A320",
      "confidence": 0.031
    },
    {
      "aircraft": "Boeing 777",
      "confidence": 0.014
    }
  ]
}
```

## Features

* Aircraft image classification using PyTorch
* Transfer learning using a pretrained convolutional neural network
* Image preprocessing and normalization
* Data augmentation during training
* Training and validation pipeline
* Model evaluation using classification accuracy
* Saved model weights for inference
* FastAPI inference server
* Image upload through a REST API
* Top-3 aircraft predictions with confidence scores
* React and TypeScript frontend
* Drag-and-drop image upload
* Image preview before classification
* Aviation-themed user interface
* Local inference without external AI APIs

## Aircraft Classes

The initial model is trained to classify a small set of common commercial aircraft types.

The exact classes depend on the dataset used for training. The initial target classes include aircraft such as:

* Boeing 737
* Boeing 747
* Boeing 777
* Airbus A320
* Airbus A330
* Airbus A350
* Embraer E-Jet
* Bombardier CRJ

The number of classes can be expanded as more labeled training data becomes available.

## Technology Stack

### Machine Learning

* Python
* PyTorch
* Torchvision
* NumPy
* Pillow
* Scikit-learn

### Backend

* FastAPI
* Uvicorn
* Python

### Frontend

* React
* TypeScript
* Tailwind CSS

### Model

The project uses transfer learning with a pretrained convolutional neural network such as ResNet18 or MobileNet.

Instead of training a large image model entirely from scratch, the model starts with visual features learned from a large image dataset and is fine-tuned for aircraft classification.

This approach provides a good balance between:

* Training time
* Model size
* Classification performance
* Learning value

## Machine Learning Pipeline

### 1. Dataset Preparation

The dataset contains labeled aircraft images organized by class.

A typical dataset structure is:

```text
data/
├── train/
│   ├── boeing_737/
│   ├── boeing_747/
│   ├── boeing_777/
│   ├── airbus_a320/
│   └── ...
│
├── val/
│   ├── boeing_737/
│   ├── boeing_747/
│   └── ...
│
└── test/
    ├── boeing_737/
    ├── boeing_747/
    └── ...
```

Each directory represents one aircraft class.

### 2. Image Preprocessing

Images can have different dimensions, orientations, and resolutions.

Before being passed to the model, images are:

1. Resized to the model's expected input dimensions.
2. Converted into tensors.
3. Normalized using the preprocessing expected by the pretrained model.

For training, additional transformations can be applied to improve generalization.

Examples include:

* Random horizontal flipping
* Random cropping
* Small rotations
* Color variations
* Random resizing

These transformations allow the model to learn aircraft features without memorizing the exact training images.

### 3. Model Training

The classifier consists of a pretrained convolutional neural network followed by a classification layer.

Conceptually:

```text
Input Image
     |
     v
Convolutional Feature Extractor
     |
     v
Learned Visual Features
     |
     v
Classification Layer
     |
     v
Aircraft Classes
```

The final classification layer is modified to match the number of aircraft classes in the dataset.

During training, the model repeatedly:

```text
Image
  |
  v
Prediction
  |
  v
Calculate Loss
  |
  v
Backpropagation
  |
  v
Update Weights
```

The process is repeated over multiple epochs until the model learns useful visual features for distinguishing the aircraft classes.

## Transfer Learning

AeroVision uses transfer learning because training a deep convolutional neural network from scratch requires a large dataset and significant computational resources.

A pretrained model already contains useful low-level visual representations such as:

* Edges
* Shapes
* Textures
* Patterns
* Object structures

These learned features can then be adapted to aircraft classification.

The final classification layer is replaced with a new layer corresponding to the aircraft classes.

This allows the project to achieve useful results with a comparatively smaller dataset and training setup.

## Model Evaluation

The model is evaluated using a separate validation and test dataset.

The primary metric is classification accuracy:

```text
Accuracy = Correct Predictions / Total Predictions
```

The project can also evaluate:

* Precision
* Recall
* F1-score
* Confusion matrix

A confusion matrix is particularly useful for aircraft classification because some aircraft can have very similar visual characteristics.

For example:

```text
             Predicted
             A320  B737  B777
Actual A320   92     7     1
Actual B737    6    91     3
Actual B777    1     4    95
```

This helps identify which aircraft classes the model struggles to distinguish.

## Backend API

The trained model is loaded by the FastAPI application and used for inference.

### Health Check

```http
GET /health
```

Example response:

```json
{
  "status": "healthy"
}
```

### Aircraft Prediction

```http
POST /predict
```

The endpoint accepts an image file and returns the model's predictions.

Example response:

```json
{
  "aircraft": "Boeing 737",
  "confidence": 0.942,
  "predictions": [
    {
      "aircraft": "Boeing 737",
      "confidence": 0.942
    },
    {
      "aircraft": "Airbus A320",
      "confidence": 0.031
    },
    {
      "aircraft": "Boeing 777",
      "confidence": 0.014
    }
  ]
}
```

The confidence value represents the model's predicted probability for each class.

## Project Structure

```text
aerovision/
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── model.py
│   │   ├── inference.py
│   │   └── schemas.py
│   │
│   ├── models/
│   │   └── aircraft_classifier.pth
│   │
│   ├── train.py
│   └── requirements.txt
│
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   ├── App.tsx
│   │   └── main.tsx
│   │
│   ├── package.json
│   └── ...
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── README.md
└── .gitignore
```

## Running the Project

### Prerequisites

Install:

* Python 3.10+
* Node.js 18+
* npm

A GPU is recommended for model training but is not required for inference.

## Training the Model

Create and activate a Python virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install the backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Place the dataset inside the `data/` directory using the expected class-based structure.

Run the training script:

```bash
python backend/train.py
```

After training, the model weights are saved to:

```text
backend/models/aircraft_classifier.pth
```

## Starting the Backend

From the project root:

```bash
uvicorn backend.app.main:app --reload
```

The API will be available at:

```text
http://localhost:8000
```

FastAPI also provides interactive API documentation at:

```text
http://localhost:8000/docs
```

## Starting the Frontend

Navigate to the frontend directory:

```bash
cd frontend
```

Install dependencies:

```bash
npm install
```

Start the development server:

```bash
npm run dev
```

The frontend communicates with the FastAPI backend to send uploaded images and display the model's predictions.

## Inference Flow

When a user uploads an image:

```text
User
 |
 | Upload aircraft image
 v
React Frontend
 |
 | HTTP POST /predict
 v
FastAPI Backend
 |
 v
Image Preprocessing
 |
 v
PyTorch Model
 |
 v
Class Probabilities
 |
 v
Top-3 Predictions
 |
 v
React Frontend
 |
 v
Aircraft Type + Confidence
```

## What This Project Demonstrates

This project demonstrates the fundamental workflow involved in developing and deploying an image classification model.

### Machine Learning

* Dataset organization
* Train/validation/test splits
* Image preprocessing
* Data augmentation
* Neural network training
* Loss functions
* Backpropagation
* Optimization
* Transfer learning
* Model evaluation
* Model inference

### Software Engineering

* Separating training and inference code
* Loading trained model weights
* Building a REST API around an ML model
* Handling image uploads
* Connecting a machine learning backend with a web frontend
* Structuring an ML application for local deployment

## Limitations

The model's performance depends heavily on the quality, size, and diversity of the training dataset.

Aircraft classification can be particularly difficult when:

* Aircraft are viewed from unusual angles.
* Images have low resolution.
* Aircraft are partially occluded.
* Multiple aircraft appear in the same image.
* Different aircraft variants have very similar appearances.
* The aircraft is not represented sufficiently in the training dataset.

The current version is designed as a classification system for a predefined set of aircraft classes. It is not a general-purpose aircraft recognition system.

## Future Improvements

Potential improvements include:

* Increasing the number of aircraft classes
* Adding more aircraft variants
* Using a larger and more diverse dataset
* Improving data augmentation
* Comparing multiple CNN architectures
* Hyperparameter tuning
* Fine-tuning more layers of the pretrained network
* Adding precision, recall, F1-score, and confusion matrix reporting
* Adding model explainability using Grad-CAM
* Detecting multiple aircraft in a single image
* Moving from image classification to object detection
* Deploying the inference API to a cloud platform
* Optimizing the model for faster inference

## Learning Objectives

AeroVision was built as a practical introduction to computer vision and deep learning.

The main objective is to understand how an image moves through an end-to-end machine learning system:

```text
Raw Image
    |
    v
Preprocessing
    |
    v
Tensor
    |
    v
Neural Network
    |
    v
Class Probabilities
    |
    v
Aircraft Prediction
```

The project provides a foundation for progressing from basic image classification to more advanced computer vision tasks such as object detection, image segmentation, and model deployment.

## License

This project is intended for educational and research purposes.
