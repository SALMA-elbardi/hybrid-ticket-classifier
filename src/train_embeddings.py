import time
import yaml
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from src.embeddings import TextEmbedder


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_and_evaluate():
    config = load_config()
    exp_name = "ticket_classifier_embeddings_experiments"
    mlflow.set_experiment(exp_name)

    print(" Chargement des données prétraitées...")
    train_df = pd.read_csv("data/processed/train.csv")
    test_df = pd.read_csv("data/processed/test.csv")

    X_train_raw = train_df["text"].astype(str).tolist()
    y_train = np.array(train_df["category"].tolist(), dtype=str)

    X_test_raw = test_df["text"].astype(str).tolist()
    y_test = np.array(test_df["category"].tolist(), dtype=str)

    # 1. Extraction des Embeddings Denses
    print(" Génération des embeddings denses avec all-MiniLM-L6-v2...")
    embedder = TextEmbedder()

    t0_enc = time.perf_counter()
    X_train = embedder.encode(X_train_raw, show_progress=True)
    X_test = embedder.encode(X_test_raw, show_progress=False)
    enc_latency_per_sample_ms = ((time.perf_counter() - t0_enc) / len(X_train_raw)) * 1000

    print(f" Encodage terminé. Dimension: {X_train.shape[1]} | Latence moyenne: {enc_latency_per_sample_ms:.2f} ms/ticket")

    # 2. Candidats supervisés
    models = {
        "Dense_LogisticRegression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        ),
        "Dense_Calibrated_LinearSVC": CalibratedClassifierCV(
            estimator=LinearSVC(C=1.0, random_state=42, max_iter=2000),
            cv=3,
            method="sigmoid"
        ),
        "Dense_RandomForest": RandomForestClassifier(
            n_estimators=150,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        )
    }

    results = []

    # 3. Entraînement et Traçabilité MLflow
    for model_name, clf in models.items():
        with mlflow.start_run(run_name=model_name):
            print(f"\n Entraînement de {model_name}...")
            
            t0 = time.perf_counter()
            clf.fit(X_train, y_train)
            train_time = time.perf_counter() - t0

            # Évaluation
            t0_inf = time.perf_counter()
            y_pred = clf.predict(X_test)
            inf_latency_ms = ((time.perf_counter() - t0_inf) / len(X_test)) * 1000

            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")

            # Logs des hyperparamètres et métriques
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("feature_extractor", "all-MiniLM-L6-v2")
            mlflow.log_param("embedding_dimension", 384)
            
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_macro", f1_macro)
            mlflow.log_metric("train_time_sec", train_time)
            mlflow.log_metric("inference_latency_ms", inf_latency_ms)
            mlflow.log_metric("encoding_latency_ms", enc_latency_per_sample_ms)
            mlflow.log_metric("total_fast_path_latency_ms", enc_latency_per_sample_ms + inf_latency_ms)

            # Sérialisation via cloudpickle pour supporter la calibration
            mlflow.sklearn.log_model(
                clf,
                artifact_path="model",
                serialization_format="cloudpickle"
            )

            print(f" {model_name} -> Accuracy: {acc * 100:.2f}% | F1-Macro: {f1_macro:.4f} | Latence totale: {(enc_latency_per_sample_ms + inf_latency_ms):.1f} ms")

            results.append({
                "name": model_name,
                "model": clf,
                "f1_macro": f1_macro,
                "accuracy": acc,
                "run_id": mlflow.active_run().info.run_id
            })

    # 4. Sélection et promotion du Champion
    champion = max(results, key=lambda x: x["f1_macro"])
    print(f"\n CHAMPION DES EMBEDDINGS : {champion['name']} (F1-Macro: {champion['f1_macro']:.4f})")

    client = mlflow.tracking.MlflowClient()
    reg_name = "DenseTicketClassifierChampion"
    model_uri = f"runs:/{champion['run_id']}/model"
    
    mv = mlflow.register_model(model_uri=model_uri, name=reg_name)
    client.set_registered_model_alias(name=reg_name, alias="champion", version=mv.version)
    print(f" Champion enregistré dans le Registry MLflow sous le nom '{reg_name}' (alias: champion).")


if __name__ == "__main__":
    train_and_evaluate()