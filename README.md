
# Optimizing Early Diabetes Diagnosis through Artificial Neural Networks Project
**DS402 Group 4**

## Project Overview
This project implements a machine learning pipeline to predict early-stage diabetes risk using non-invasive, questionnaire-based health features. A Feedforward Artificial Neural Network (ANN) is developed and benchmarked against several classical machine learning models.

The main objective is to evaluate whether an ANN can serve as a reliable screening model while prioritizing recall, which is especially important in medical applications where missing a positive case may lead to delayed diagnosis.

## Project Structure
.
├── code/
│ └── diabetes_prediction.ipynb # Jupyter Notebook (main code)
│
├── data/
│ └── early_stage_diabetes.csv # Dataset
│
├── result/
│ ├── cv_summary.xlsx
│ ├── test_set_comparison.xlsx
│ ├── ann_details.json
│ ├── ann_roc.png
│ ├── ann_pr.png
│ ├── ann_calibration.png
│ ├── ann_confusion_matrix.png
│ ├── rf_permutation_importance.xlsx
│ └── rf_permutation_importance.png
│
└── README.md

## Dataset
The project uses the **Early Stage Diabetes Risk Prediction** dataset, which contains:

- 520 patient records  
- 16 predictor variables  
- 1 binary target label (`class`: Positive / Negative)  

Feature types:
- One continuous feature: `Age`  
- Binary features: symptom indicators (Yes/No) and `Gender`  

The dataset file must be placed inside the `data/` folder.

## Methodology Summary
- Categorical variables are encoded numerically (Yes/No → 1/0, Male/Female → 1/0).  
- Only the `Age` feature is standardized using `StandardScaler`.  
- Binary symptom features are passed through unchanged using a `ColumnTransformer`.  
- Stratified splitting is applied for training, validation, and testing.  
- Baseline models are evaluated using stratified 5-fold cross-validation.  
- The ANN uses dropout and early stopping to reduce overfitting.  
- Probability calibration is applied using isotonic regression.  
- The ANN decision threshold is tuned on the validation set to prioritize recall.  
- Random Forest permutation importance is used for feature interpretability.

## Models Implemented

### Baseline Models
- Logistic Regression  
- Support Vector Machine (RBF kernel)  
- Random Forest  
- K-Nearest Neighbors (k = 5)  

### Artificial Neural Network
- Feedforward ANN  
- Two hidden layers (32 and 16 neurons)  
- ReLU activation  
- Dropout regularization  
- Sigmoid output layer  
- Adam optimizer with binary cross-entropy loss  
- Early stopping  

## Requirements
Python 3.9 or later is recommended.

Install required libraries:
pip install numpy pandas matplotlib scikit-learn tensorflow openpyxl
Running the Project in Google Colab (Recommended)
Open Google Colab and create a new notebook.

Upload the following folders to Colab:
- code/
- data/

Open the notebook:
```python
code/DS402_G4_Code.ipynb

In the notebook, confirm that the dataset path is set correctly:
CSV_PATH = "data/dataset.csv"
Run all cells in order.

After execution, all results will be automatically saved to:
result/

To download the results folder:
!zip -r result.zip result

Running the Project Locally (Jupyter / VS Code)
Clone or download the repository.

Ensure the folder structure is preserved:
- code/
- data/
- result/

Launch Jupyter: jupyter notebook

Open:
code/diabetes_prediction.ipynb

Verify the dataset path:
CSV_PATH = "data/dataset.csv"
Run all cells from top to bottom.

All outputs will appear in the result/ folder.

Outputs Explained
After running the notebook, the result/ folder will contain:
- Cross-validation summary (cv_summary.xlsx)
- Test-set performance comparison (test_set_comparison.xlsx)
ANN diagnostic plots:
- ROC curve
- Precision–Recall curve
- Calibration curve
- Confusion matrix

ANN details (ann_details.json)

Random Forest feature importance (Excel file and PNG)

Notes for Users
Dataset column names must match the expected names used in the notebook.

The notebook automatically strips whitespace from column names to reduce formatting issues.

TensorFlow warnings during training are normal and do not affect results.

Run all cells sequentially, as later steps depend on earlier outputs.

Authors
DS402 Group 4
