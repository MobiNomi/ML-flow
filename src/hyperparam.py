from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import mlflow.sklearn

# 🔥 Connect to MLflow server (localhost:5001)
import dagshub 

dagshub.init(repo_owner='MobiNomi', repo_name='ML-flow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/MobiNomi/ML-flow.mlflow")


# Set experiment
mlflow.set_experiment("breast-cancer-rf-hp")

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
rf = RandomForestClassifier(random_state=42)

# Hyperparameters
param_grid = {
    "n_estimators": [10, 50, 100],
    "max_depth": [None, 10, 20, 30],
}

# GridSearch
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=1,
)

# Start MLflow run
with mlflow.start_run():

    # Train
    grid_search.fit(X_train, y_train)

    # 🔁 Log each parameter combination
    for i, params in enumerate(grid_search.cv_results_["params"]):
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric(
                "accuracy",
                grid_search.cv_results_["mean_test_score"][i],
            )

    # Best results
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print("Best Params:", best_params)
    print("Best Score:", best_score)

    # Log best
    mlflow.log_params(best_params)
    mlflow.log_metric("best_accuracy", best_score)

    # 🔹 Log training data
    train_df = X_train.copy()
    train_df["target"] = y_train
    mlflow.log_input(mlflow.data.from_pandas(train_df), "training")

    # 🔹 Log testing data
    test_df = X_test.copy()
    test_df["target"] = y_test
    mlflow.log_input(mlflow.data.from_pandas(test_df), "testing")

    # 🔹 Log model (with input example → fixes warning)
    mlflow.sklearn.log_model(
        grid_search.best_estimator_,
        artifact_path="random_forest",
        input_example=X_train.iloc[:5],
    )

    # Tag
    mlflow.set_tag("author", "Mubashir Hussain")