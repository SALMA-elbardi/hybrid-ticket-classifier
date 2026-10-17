import os
import time
import yaml
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.pipeline import HybridTicketClassifier


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_hybrid_evaluation(sample_size=80):
    config = load_config()
    test_path = os.path.join(config["data"]["processed_dir"], "test.csv")
    test_df = pd.read_csv(test_path)

    # Vérification et normalisation des colonnes
    if "category" not in test_df.columns:
        test_df.columns = [c.strip().lower() for c in test_df.columns]

    # Échantillonnage stratifié direct et propre
    if sample_size and len(test_df) > sample_size:
        eval_df, _ = train_test_split(
            test_df,
            train_size=sample_size,
            stratify=test_df["category"],
            random_state=config["data"]["random_state"],
        )
        eval_df = eval_df.reset_index(drop=True)
    else:
        eval_df = test_df.copy()

    print(f" Début de l'évaluation sur {len(eval_df)} tickets de test...")

    classifier = HybridTicketClassifier()

    y_true = eval_df["category"].tolist()
    y_pred_ml_only = []
    y_pred_hybrid = []
    latencies_ml_only = []
    latencies_hybrid = []
    routed_by_list = []

    for idx, row in eval_df.iterrows():
        text = str(row["text"])

        # 1. Évaluation pure ML
        t0 = time.perf_counter()
        probs = classifier.ml_model.predict_proba([text])[0]
        max_idx = int(np.argmax(probs))
        ml_cat = classifier.classes[max_idx]
        lat_ml = (time.perf_counter() - t0) * 1000
        y_pred_ml_only.append(ml_cat)
        latencies_ml_only.append(lat_ml)

        # 2. Évaluation Hybride (Cascade)
        res = classifier.predict(text)
        y_pred_hybrid.append(res["category"])
        latencies_hybrid.append(res["latency_ms"])
        routed_by_list.append("ML" if "ML_Champion" in res["routed_by"] else "LLM")

        if (idx + 1) % 10 == 0 or (idx + 1) == len(eval_df):
            print(f"Progression : {idx + 1}/{len(eval_df)} requêtes traitées...")

    # Calcul des métriques
    acc_ml = accuracy_score(y_true, y_pred_ml_only)
    f1_ml = f1_score(y_true, y_pred_ml_only, average="macro")
    avg_lat_ml = float(np.mean(latencies_ml_only))

    acc_hybrid = accuracy_score(y_true, y_pred_hybrid)
    f1_hybrid = f1_score(y_true, y_pred_hybrid, average="macro")
    avg_lat_hybrid = float(np.mean(latencies_hybrid))

    ml_count = routed_by_list.count("ML")
    llm_count = routed_by_list.count("LLM")
    ml_ratio = (ml_count / len(routed_by_list)) * 100
    llm_ratio = (llm_count / len(routed_by_list)) * 100

    print("\n" + "=" * 60)
    print(" RÉSULTATS DU BENCHMARK COMPARATIF")
    print("=" * 60)
    print(f"Nombre de tickets évalués : {len(eval_df)}")
    print(f"Routage du trafic         : ML = {ml_ratio:.1f}% ({ml_count}) | LLM = {llm_ratio:.1f}% ({llm_count})")
    print("-" * 60)
    print(f"Modèle ML Seul            : F1-Macro = {f1_ml:.4f} | Accuracy = {acc_ml:.4f} | Latence Moy = {avg_lat_ml:.2f} ms")
    print(f"Pipeline Hybride          : F1-Macro = {f1_hybrid:.4f} | Accuracy = {acc_hybrid:.4f} | Latence Moy = {avg_lat_hybrid:.2f} ms")
    print("=" * 60)

    # Génération du graphique comparatif
    os.makedirs("artifacts", exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Graphique 1 : F1-Score & Accuracy
    metrics_data = {
        "ML Seul": [acc_ml, f1_ml],
        "Hybride": [acc_hybrid, f1_hybrid],
    }
    x = np.arange(2)
    width = 0.35
    axes[0].bar(x - width / 2, metrics_data["ML Seul"], width, label="ML Seul", color="#2b5c8f")
    axes[0].bar(x + width / 2, metrics_data["Hybride"], width, label="Hybride (ML+LLM)", color="#00a86b")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(["Accuracy", "F1 Macro"])
    axes[0].set_ylim(0.8, 1.0)
    axes[0].set_title("Qualité Prédictive")
    axes[0].legend()
    axes[0].grid(axis="y", linestyle="--", alpha=0.5)

    # Graphique 2 : Répartition du trafic
    axes[1].pie(
        [ml_ratio, llm_ratio],
        labels=["Fast-Path (ML)", "Slow-Path (LLM)"],
        autopct="%1.1f%%",
        colors=["#2b5c8f", "#f39c12"],
        startangle=140,
    )
    axes[1].set_title(f"Routage Trafic (Latence Moy : {avg_lat_hybrid:.1f} ms)")

    plt.tight_layout()
    chart_path = "artifacts/hybrid_benchmark_tradeoff.png"
    plt.savefig(chart_path)
    plt.close()
    print(f" Graphique sauvegardé -> {chart_path}\n")


if __name__ == "__main__":
    run_hybrid_evaluation(sample_size=80)