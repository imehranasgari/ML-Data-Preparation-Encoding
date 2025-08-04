# Machine Learning in Python: Data Preprocessing

## Problem Statement

The project addresses common challenges in data preprocessing for machine learning, specifically focusing on handling missing values and encoding categorical data. The dataset used contains missing values in 'Age' and 'Salary' columns, and categorical features like 'Country' and 'Purchased' need to be converted into numerical representations for machine learning algorithms.

## Solution Approach

The solution involves a series of data preprocessing steps:

*   **Importing Libraries and Dataset**: Essential libraries like `numpy`, `pandas`, `matplotlib.pyplot`, `seaborn`, and `sklearn` are imported. The dataset, `Data.csv`, is loaded into a pandas DataFrame.
*   **Handling Missing Values**:
    *   **Dropping Rows**: A method to remove rows containing any missing values is demonstrated using `dropna()`.
    *   **Filling with Mean**: Missing numerical values are imputed by filling them with the mean of their respective columns using `fillna()`.
    *   **Scikit-learn Imputer**: The `SimpleImputer` from `sklearn.impute` is used to replace missing numerical values with the mean strategy.
*   **Encoding Categorical Data**:
    *   **Independent Variable Encoding (One-Hot Encoding)**: The 'Country' column is transformed into numerical format using `OneHotEncoder` from `sklearn.preprocessing.ColumnTransformer`.
    *   **Dependent Variable Encoding (Label Encoding)**: The 'Purchased' column is converted into numerical labels (0s and 1s) using `LabelEncoder` from `sklearn.preprocessing`.
*   **Splitting the Dataset**: The preprocessed data is split into training and testing sets using `train_test_split` from `sklearn.model_selection` with a test size of 20%.
*   **Feature Scaling**:
    *   **MinMaxScaler (Normalization)**: Numerical features are scaled to a range between 0 and 1 using `MinMaxScaler`.
    *   **StandardScaler (Standardization)**: Numerical features are scaled to have a mean of 0 and a standard deviation of 1 using `StandardScaler`.

## Technologies & Libraries

*   **Python**: Programming language.
*   **NumPy**: For numerical operations, especially with arrays.
*   **Pandas**: For data manipulation and analysis.
*   **Matplotlib.pyplot**: For creating static, interactive, and animated visualizations.
*   **Seaborn**: For statistical data visualization.
*   **Scikit-learn**: A comprehensive machine learning library for:
    *   `SimpleImputer`: Handling missing values.
    *   `ColumnTransformer`: Applying different transformers to different columns.
    *   `OneHotEncoder`: Encoding categorical features.
    *   `LabelEncoder`: Encoding target labels.
    *   `train_test_split`: Splitting data into training and test sets.
    *   `MinMaxScaler`: Normalization for feature scaling.
    *   `StandardScaler`: Standardization for feature scaling.

## Installation & Execution Guide

To run this notebook, ensure you have Python installed along with the necessary libraries.

1.  **Clone the repository (Not provided, assuming local access):**
    If this were a GitHub repository, you would clone it using:
    ```bash
    git clone <repository_url>
    cd <repository_name>
    ```
    Since no repository URL is provided, assume the `Feature-Scaling.ipynb` file and `Data.csv` are in your local directory.

2.  **Install dependencies:**
    ```bash
    pip install numpy pandas matplotlib seaborn scikit-learn
    ```

3.  **Run the Jupyter Notebook:**
    ```bash
    jupyter notebook Feature-Scaling.ipynb
    ```
    This will open the Jupyter environment in your web browser, where you can execute the cells sequentially to see the data preprocessing steps in action.

## Key Results / Performance

The project demonstrates the successful application of various data preprocessing techniques:

*   **Missing Value Handling**: Missing values in 'Age' and 'Salary' columns are identified and successfully imputed using both `fillna()` with mean and `SimpleImputer`.
*   **Categorical Data Encoding**:
    *   The 'Country' column is successfully one-hot encoded, transforming it into numerical representations (e.g., `[1.0 0.0 0.0]` for France).
    *   The 'Purchased' column is label encoded into `0` (No) and `1` (Yes).
*   **Dataset Splitting**: The dataset is correctly split into training (80%) and testing (20%) sets, ready for model training and evaluation.
*   **Feature Scaling**: Both `MinMaxScaler` and `StandardScaler` are applied to the numerical features (Age and Salary) in the training and test sets, resulting in scaled numerical values.

The output of `print(X_train)` and `print(X_test)` after feature scaling shows the transformed numerical values, indicating that the scaling was applied successfully.

## Screenshots / Sample Outputs

**Original DataFrame (`df`):**
```
   Country   Age   Salary Purchased
0   France  44.0  72000.0        No
1    Spain  27.0  48000.0       Yes
2  Germany  30.0  54000.0        No
3    Spain  38.0  61000.0        No
4  Germany  40.0      NaN       Yes
5   France  35.0  58000.0       Yes
6    Spain   NaN  52000.0        No
7   France  48.0  79000.0       Yes
8  Germany  50.0  83000.0        No
9   France  37.0  67000.0       Yes
```

**Missing Values Count (`df.isnull().sum()`):**
```
Country      0
Age          1
Salary       1
Purchased    0
dtype: int64
```

**DataFrame after `fillna()` with mean (`df_fillna`):**
```
   Country        Age        Salary Purchased
0   France  44.000000  72000.000000        No
1    Spain  27.000000  48000.000000       Yes
2  Germany  30.000000  54000.000000        No
3    Spain  38.000000  61000.000000        No
4  Germany  40.000000  63777.777778       Yes
5   France  35.000000  58000.000000       Yes
6    Spain  38.777778  52000.000000        No
7   France  48.000000  79000.000000       Yes
8  Germany  50.000000  83000.000000        No
9   France  37.000000  67000.000000       Yes
```

**Independent Variable `X` after One-Hot Encoding and Imputation:**
```
[[1.0 0.0 0.0 44.0 72000.0]
 [0.0 0.0 1.0 27.0 48000.0]
 [0.0 1.0 0.0 30.0 54000.0]
 [0.0 0.0 1.0 38.0 61000.0]
 [0.0 1.0 0.0 40.0 63777.77777777778]
 [1.0 0.0 0.0 35.0 58000.0]
 [0.0 0.0 1.0 38.77777777777778 52000.0]
 [1.0 0.0 0.0 48.0 79000.0]
 [0.0 1.0 0.0 50.0 83000.0]
 [1.0 0.0 0.0 37.0 67000.0]]
```

**Dependent Variable `y` after Label Encoding:**
```
[0 1 0 0 1 1 0 1 0 1]
```

**`X_train` after StandardScaler:**
```
[[0.0 0.0 1.0 -0.19159184384578537 -1.0781259408412425]
 [0.0 1.0 0.0 -0.014117293757057581 -0.07013167641635436]
 [1.0 0.0 0.0 0.5667085065333245 0.6335624327104541]
 [0.0 0.0 1.0 -0.3045301939022482 -0.3078661727429788]
 [0.0 0.0 1.0 -1.9018011447007983 -1.4204636155515822]
 [1.0 0.0 0.0 1.1475343068237058 1.2326533634535486]
 [0.0 1.0 0.0 1.4379472069688963 1.5749910381638883]
 [1.0 0.0 0.0 -0.740149544120035 -0.5646194287757338]]
```

**`X_test` after StandardScaler:**
```
[[0.0 1.0 0.0 -1.4661817944830116 -0.9069571034860727]
 [1.0 0.0 0.0 -0.4497366439748436 0.20564033932252992]]
```

## Acknowledgements

This project was inspired by educational materials by Soheil Tehranipour. All code, explanations, and documentation have been fully rewritten by mehran Asgari for portfolio and educational purposes.

*   Website: [http://www.iran-machinelearning.ir](http://www.iran-machinelearning.ir)
*   
---

**Author:** mehran Asgari
**Email:** [imehranasgari@gmail.com](mailto:imehranasgari@gmail.com).
**GitHub:** [https://github.com/imehranasgari](https://github.com/imehranasgari).

---

## 📄 License

This project is licensed under the MIT License – see the `LICENSE` file for details.

