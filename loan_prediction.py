import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
import mlflow
import os

# Load dataset
df = pd.read_csv(r"C:\Users\Shreyansh Singh\Desktop\MLfolder\PackageA\prediction_model\datasets\train.csv")

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
categorical_cols.remove('Loan_Status')
categorical_cols.remove('Loan_ID')

# Fill missing values for categorical columns
for cols in categorical_cols:
    df[cols].fillna(df[cols].mode()[0], inplace=True)

# Fill missing values for numerical columns
for cols in numerical_cols:
    df[cols].fillna(df[cols].median(), inplace=True)

# Clip numerical values to reduce outliers
df[numerical_cols] = df[numerical_cols].apply(lambda x: x.clip(*x.quantile([0.05, 0.95])))

# Log transformation of skewed numerical features
df['LoanAmount'] = np.log(df['LoanAmount'])
df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome'] = np.log(df['TotalIncome'])

# Drop redundant columns
df = df.drop(columns=['ApplicantIncome', 'CoapplicantIncome'])

# Encode categorical variables
le = LabelEncoder()
for cols in categorical_cols:
    df[cols] = le.fit_transform(df[cols])

# Encode target variable
df['Loan_Status'] = le.fit_transform(df['Loan_Status'])

# Split data into features and target
X = df.drop(columns=['Loan_Status', 'Loan_ID'])
y = df['Loan_Status']

# Split data into training and testing sets
RANDOM_SEED = 6
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# RandomForestClassifier
rf = RandomForestClassifier(random_state=RANDOM_SEED)
param_grid_forest = {
    'n_estimators': [200, 400, 700],
    'max_depth': [10, 20, 30],
    'criterion': ["gini", "entropy"],
    'max_leaf_nodes': [50, 100]
}

grid_forest = GridSearchCV(
    estimator=rf,
    param_grid=param_grid_forest,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=0
)
model_forest = grid_forest.fit(X_train, y_train)

# LogisticRegression
lr = LogisticRegression()
param_grid_log = {
    'C': [100, 10, 1, 0.1, 0.01],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
grid_log = GridSearchCV(estimator=lr,
                        param_grid=param_grid_log,
                        cv=5,
                        n_jobs=-1,
                        scoring='accuracy',
                        verbose=0)

model_log = grid_log.fit(X_train, y_train)

# DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=RANDOM_SEED)
param_grid_tree = {
    "max_depth": [3, 5, 7, 9, 11, 13],
    'criterion': ["gini", "entropy"]
}
grid_tree = GridSearchCV(
    cv=5,
    param_grid=param_grid_tree,
    verbose=0,
    n_jobs=-1,
    scoring='accuracy',
    estimator=dt
)

model_tree = grid_tree.fit(X_train, y_train)

# Define experiment
mlflow.set_experiment("Loan Prediction")


def eval_metrics(actual, pred):
    accuracy = metrics.accuracy_score(actual, pred)
    f1 = metrics.f1_score(actual, pred)
    fpr, tpr, _ = metrics.roc_curve(actual, pred, pos_label=1)
    auc = metrics.auc(fpr, tpr)
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='blue', label='ROC curve area = %0.2f' % auc)
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([-0.1, 1.1])
    plt.ylim([-0.1, 1.1])
    plt.xlabel('False Positive Rate', size=14)
    plt.ylabel('True Positive Rate', size=14)
    plt.legend(loc='lower right')
    os.makedirs("plots", exist_ok=True)
    plt.savefig("plots/ROC_curve.png")
    plt.close()
    return accuracy, f1, auc


def mlflow_logging(model, X, y, name):
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        mlflow.set_tag("run_id", run_id)
        pred = model.predict(X)
        accuracy, f1, auc = eval_metrics(y, pred)
        mlflow.log_params(model.best_params_)
        mlflow.log_metric("Mean CV score", model.best_score_)
        mlflow.log_metric("Accuracy", accuracy)
        mlflow.log_metric("f1-score", f1)
        mlflow.log_metric("AUC", auc)
        mlflow.log_artifact("plots/ROC_curve.png")
        mlflow.sklearn.log_model(model, name)
        mlflow.end_run()


# Log models and metrics
mlflow_logging(model_tree, X_test, y_test, "DecisionTree")
mlflow_logging(model_log, X_test, y_test, "Logistic")
mlflow_logging(model_forest, X_test, y_test, "RandomForest")
