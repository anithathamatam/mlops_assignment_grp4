import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Set MLflow tracking URI to project root
mlflow.set_tracking_uri("file:../mlruns")

# Load preprocessed data
df = pd.read_csv("../dataset/raw/iris.csv")
X = df.drop("target", axis=1)
y = df["target"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ✅ Scale only the training data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42)
}

best_model = None
best_accuracy = 0.0
best_run_id = None

mlflow.set_experiment("iris_classification")

# Train and track each model
for model_name, model in models.items():
    with mlflow.start_run(run_name=model_name) as run:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        # Log params and metrics
        if model_name == "LogisticRegression":
            mlflow.log_param("max_iter", model.max_iter)
        elif model_name == "RandomForest":
            mlflow.log_param("n_estimators", model.n_estimators)

        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        print(f"🔍 {model_name} Accuracy: {acc:.4f}")

        # Save best model info
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = model
            best_run_id = run.info.run_id

# Register best model
if best_run_id:
    model_uri = f"runs:/{best_run_id}/model"
    registered_model = mlflow.register_model(model_uri=model_uri, name="IrisClassifier")
    print(f"✅ Registered model: {registered_model.name}, version: {registered_model.version}")
    print(f"🏆 Best model: {registered_model.name}, Accuracy: {best_accuracy:.4f}")


