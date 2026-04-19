import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix as Confusion_matrix
import warnings
import seaborn as sns
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning)
import dagshub 

dagshub.init(repo_owner='MobiNomi', repo_name='ML-flow', mlflow=True)
mlflow.set_tracking_uri("https://dagshub.com/MobiNomi/ML-flow.mlflow")

mlflow.set_experiment("Wine_Classification_Experiment")  # Set the experiment name

wine = load_wine()
X = wine.data
y = wine.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

max_depth = 5
n_estimators = 8

with mlflow.start_run():
    model = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    mlflow.set_tags({"Author": "Mubashir Hussain" , "Model": "RandomForestClassifier"})
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")
    # creating the confusion matrix and logging it as an artifact
    cm = Confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("confusion_matrix.png")
    
    mlflow.log_artifact("confusion_matrix.png")
    
    mlflow.log_artifact(__file__ ) # Log the current script as an artifact
    
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Run logged successfully!")
    
    
    