# Diabetes Prediction using Machine Learning Algorithms

This repository demonstrates diabetes prediction using various machine learning algorithms implemented in Python. It includes preprocessing steps, model training, evaluation, and visualization of results using popular libraries such as pandas, scikit-learn, matplotlib, seaborn, and XGBoost.

## Datasets

The dataset used in this project is named `diabetes.csv`, which contains various health-related features along with the target variable indicating diabetes outcome (0: No diabetes, 1: Diabetes). It is loaded into a pandas DataFrame for further processing.

## Preprocessing

The dataset is preprocessed to handle missing values in the 'Insulin' column by replacing them with the mean value. Features are then standardized using StandardScaler to ensure uniformity and better model performance.

## Models Implemented

The following machine learning algorithms are implemented for diabetes prediction:

- Logistic Regression
- Random Forest Classifier
- Decision Tree Classifier
- XGBoost Classifier
- Support Vector Machine (SVM)

Each model is trained on the training data and evaluated using accuracy score and confusion matrix on the test set.

## Visualization

A bar plot and a line plot are used to visualize the accuracy scores of different algorithms. This provides a comparative view of the performance of each algorithm in predicting diabetes.

## Dependencies

- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- XGBoost

You can install the required dependencies using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## Usage

1. Clone the repository:

```bash
git clone https://github.com/yourusername/diabetes-prediction.git
```

2. Navigate to the project directory:

```bash
cd diabetes-prediction
```

3. Run the Python script:

```bash
python diabetes_prediction.py
```

This will execute the script, train the models, evaluate their performance, and visualize the results.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, feel free to open an issue or create a pull request.

