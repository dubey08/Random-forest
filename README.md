

# Diabetes Prediction Using Random Forest

Welcome to the Diabetes Prediction project repository! This project uses the Random Forest algorithm to predict the likelihood of a person having diabetes based on medical data.

## Features

- **Data Preprocessing**: Clean and prepare the dataset for analysis.
- **Model Training**: Train a Random Forest classifier on the dataset.
- **Model Evaluation**: Evaluate the performance of the model using various metrics.
- **Prediction**: Use the trained model to make predictions on new data.

## Technologies Used

- Programming Language: Python
- Libraries: 
  - Pandas
  - NumPy
  - Scikit-learn
  - Matplotlib (for visualizations)

## Dataset

The dataset used in this project is the [PIMA Indians Diabetes Dataset](https://www.kaggle.com/uciml/pima-indians-diabetes-database).

## Getting Started

### Prerequisites

- Python 3.x
- [pip](https://pip.pypa.io/en/stable/)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/diabetes-prediction.git
   cd diabetes-prediction
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Project

1. Ensure the dataset is in the project directory or update the path in the script accordingly.
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `diabetes_prediction.ipynb` and run all cells to see the data preprocessing, model training, and evaluation steps.

## Usage

### Training the Model

The Jupyter Notebook includes the following steps:
1. Load and explore the dataset.
2. Preprocess the data (handle missing values, feature scaling, etc.).
3. Split the data into training and testing sets.
4. Train a Random Forest classifier.
5. Evaluate the model's performance.

### Making Predictions

After training the model, you can use it to make predictions on new data:
1. Load the new data.
2. Preprocess the new data similarly to the training data.
3. Use the trained model to predict the likelihood of diabetes.

Example:
```python
new_data = [[5, 166, 72, 19, 175, 25.8, 0.587, 51]]
prediction = model.predict(new_data)
print("Prediction:", "Diabetic" if prediction[0] == 1 else "Not Diabetic")
```

## Results

The performance of the Random Forest classifier is evaluated using metrics such as accuracy, precision, recall, and F1-score. Confusion matrix and ROC curve visualizations are also provided.


