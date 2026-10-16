import os
import time
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.preprocessor import TextCleaner


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(y_true, y_pred, labels, output_path):
    """Génère et sauvegarde l'image de la matrice de confusion."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title("Matrice de Confusion")
    plt.xlabel("Prédiction")
    plt.ylabel("Vérité terrain")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def train_and_track():
    config = load_config()
    ml_cfg = config["ml"]
    data_cfg = config["data"]

    # 1. Chargement des données
    train_path = os.path.join(data_cfg["processed_dir"], "train.csv")
    test_path = os.path.join(data_cfg["processed_dir"], "test.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train, y_train = train_df["text"], train_df["category"]
    X_test, y_test = test_df["text"], test_df["category"]
    labels = sorted(y_train.unique().tolist())

    # 2. Configuration de l'expérience MLflow
    mlflow.set_experiment(ml_cfg["experiment_name"])
    
    # 3. Définition des 3 modèles à comparer
    models_to_test = {
        "LogisticRegression": {
            "classifier": LogisticRegression(C=2.0, max_iter=500, random_state=42),
            "params": {"model_type": "LogisticRegression", "C": 2.0, "max_iter": 500},
        },
        "Calibrated_LinearSVC": {
            "classifier": CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, max_iter=2000, random_state=42),
                cv=3
            ),
            "params": {"model_type": "LinearSVC_Calibrated", "C": 1.0, "cv": 3},
        },
        "RandomForest": {
            "classifier": RandomForestClassifier(n_estimators=150, max_depth=20, random_state=42),
            "params": {"model_type": "RandomForest", "n_estimators": 150, "max_depth": 20},
        },
    }

    best_f1 = -1.0
    best_run_id = None
    best_model_name = ""

    os.makedirs("artifacts", exist_ok=True)

    print(" Début de l'entraînement et du tracking MLflow...\n")

    for model_name, config_item in models_to_test.items():
        with mlflow.start_run(run_name=model_name) as run:
            print(f" Entraînement : {model_name}")

            # Création du pipeline complet : Nettoyage -> Vectorisation -> Modèle
            pipeline = Pipeline([
                ("cleaner", TextCleaner()),
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
                ("clf", config_item["classifier"]),
            ])

            # Entraînement
            t_train_start = time.perf_counter()
            pipeline.fit(X_train, y_train)
            train_duration = time.perf_counter() - t_train_start

            # Évaluation & Mesure de la latence d'inférence
            t_infer_start = time.perf_counter()
            y_pred = pipeline.predict(X_test)
            total_infer_time = time.perf_counter() - t_infer_start
            latency_ms = (total_infer_time / len(X_test)) * 1000

            # Calcul des métriques
            acc = accuracy_score(y_test, y_pred)
            f1_macro = f1_score(y_test, y_pred, average="macro")
            f1_weighted = f1_score(y_test, y_pred, average="weighted")
            prec_macro = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec_macro = recall_score(y_test, y_pred, average="macro")

            print(f"   Accuracy    : {acc:.4f}")
            print(f"   F1 Macro    : {f1_macro:.4f}")
            print(f"   Latence Moy : {latency_ms:.2f} ms/requête\n")

            # Logging paramètres et métriques
            mlflow.log_params(config_item["params"])
            mlflow.log_param("ngram_range", "(1, 2)")
            mlflow.log_param("max_features", 5000)

            mlflow.log_metrics({
                "accuracy": acc,
                "f1_macro": f1_macro,
                "f1_weighted": f1_weighted,
                "precision_macro": prec_macro,
                "recall_macro": rec_macro,
                "latency_ms": latency_ms,
                "train_duration_sec": train_duration,
            })

            # Logging artefact : Matrice de confusion
            cm_path = f"artifacts/cm_{model_name}.png"
            plot_confusion_matrix(y_test, y_pred, labels, cm_path)
            mlflow.log_artifact(cm_path)

            # Logging du pipeline modèle
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                serialization_format="pickle",
                skops_trusted_types=["src.preprocessor.TextCleaner"],
            )

            # Sélection du meilleur modèle basé sur le F1-Score Macro
            if f1_macro > best_f1:
                best_f1 = f1_macro
                best_run_id = run.info.run_id
                best_model_name = model_name

    # 4. Promotion du modèle Champion dans le Model Registry MLflow
    print("==================================================")
    print(f" Meilleur modèle retenu : {best_model_name} (F1 Macro = {best_f1:.4f})")
    print(f" Enregistrement dans le Model Registry sous : '{ml_cfg['registered_model_name']}'...")

    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    reg_name = ml_cfg["registered_model_name"]

    model_details = mlflow.register_model(model_uri=model_uri, name=reg_name)

    # Attribution du tag 'champion'
    client.set_model_version_tag(
        name=reg_name,
        version=model_details.version,
        key="status",
        value="champion",
    )
    print(f" Modèle enregistré (Version {model_details.version}) avec le tag 'champion' !")
    print("==================================================")


if __name__ == "__main__":
    train_and_track()