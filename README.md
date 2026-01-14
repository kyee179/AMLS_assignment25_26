# ELEC0134 Applied Machine Learning Systems - Project Report

## Project Description
This project benchmarks two distinct machine learning approaches for the classification of medical images using the **BreastMNIST** dataset. It compares a classical pipeline (SVM with HOG features) against a deep learning approach (ResNet-18) to analyze the effects of model complexity, data augmentation, and training budgets.

---

## 1. Project Organization
The project is structured to separate the main execution logic from the model implementations and utility scripts.
* **Root Directory**: Contains the primary executable script.
* **Code Directory**: Divided into modular sub-packages:
    * **A**: Contains the classical machine learning implementation (Model A).
    * **B**: Contains the deep learning implementation (Model B).
    * **utils**: Contains shared resources for data loading and preprocessing.

## 2. File Descriptions
The role of each file in the repository is described below:

### Root Directory
* **`main.py`**: The entry point for the project. It handles command-line arguments to select models (`A` or `B`), triggers training, performs evaluation, and generates learning curve plots.

### Code/A/ (Model A)
* **`model_a.py`**: Implements the `ModelA` class. It encapsulates the Support Vector Machine (SVM) logic, including a `grid_search` method for hyperparameter tuning (Kernel and C) and standard training/evaluation functions.

### Code/B/ (Model B)
* **`model_b.py`**: Implements the `ModelB` class. It defines the training loop, calculates validation metrics, and manages the PyTorch optimizer and loss functions.

### Code/utils/ (Utilities)
* **`dataloader.py`**: Responsible for downloading and loading the BreastMNIST dataset. It provides two functions: `load_breastmnist_numpy` for the SVM (flattened arrays) and `load_breastmnist_torch` for the ResNet (DataLoaders).
* **`preprocess.py`**: Handles feature engineering and data augmentation. It includes:
    * `ImagePreprocessor`: A class for HOG feature extraction and PCA dimensionality reduction.
    * `AddGaussianNoise`: A custom PyTorch transform for tensor augmentation.
    * `NumpyAugmentations`: Static methods for adding noise/contrast to NumPy arrays.

---

## 3. Required Packages

This project requires **Python 3.9**.
To run this code, some important Python packages required are listed below.

* **`numpy`**: For array manipulation and data handling.
* **`torch`** & **`torchvision`**: For building and training the ResNet model (Model B) and data transformations.
* **`scikit-learn`**: For the SVM implementation, PCA, scaling, and metric calculations.
* **`scikit-image`**: For Histogram of Oriented Gradients (HOG) feature extraction.
* **`pandas`**: For handling Grid Search results in Model A.
* **`medmnist`**: To download and access the BreastMNIST dataset.
* **`matplotlib`**: For plotting training and validation learning curves.

To install all dependencies, run the following command:

```bash
pip install -r requirements.txt
```


## 4. Usage Instructions

Run the project from the root directory using the following commands:

Run all experiment:

```bash
python main.py
```

Run Model A (SVM):

```bash
# Standard run with Grid Search
python main.py --model A --grid_search 

# With Data Augmentation
python main.py --model A --augment --grid_search 

# Without HOG Feature Extraction
python main.py --model A --no_feature_extraction
```

Run Model B (ResNet):

```bash
# Run ResNet-18
python main.py --model B --resnet_version resnet18

# Run ResNet-50
python main.py --model B --resnet_version resnet50

# Run ResNet-18 with Data Augmentation
python main.py --model B --augment --resnet_version resnet18
```