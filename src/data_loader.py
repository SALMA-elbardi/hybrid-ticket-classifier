import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def fetch_and_prepare_data(config_path="config.yaml"):
    config = load_config(config_path)
    print(" Téléchargement direct des CSV Banking77 depuis GitHub...")

    # URLs fiables des CSV bruts officiels de PolyAI
    base_url = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
    train_url = f"{base_url}/train.csv"
    test_url = f"{base_url}/test.csv"

    # Lecture directe via Pandas
    df_train = pd.read_csv(train_url)
    df_test = pd.read_csv(test_url)
    df_all = pd.concat([df_train, df_test], ignore_index=True)

    # Standardisation des colonnes
    df_all.rename(columns={"text": "text", "category": "category"}, inplace=True)
    df_all["category"] = df_all["category"].astype(str)

    # Filtrage sur les catégories cibles définies dans config.yaml
    selected_cats = config["data"]["target_categories"]
    df_filtered = df_all[df_all["category"].isin(selected_cats)].copy()

    # Sécurité au cas où les libellés diffèrent
    if len(df_filtered) == 0:
        print(" Catégories configurées non trouvées, sélection des 8 plus fréquentes...")
        top_cats = df_all["category"].value_counts().head(8).index.tolist()
        df_filtered = df_all[df_all["category"].isin(top_cats)].copy()
        print(f"Catégories retenues : {top_cats}")

    # Sauvegarde des données brutes filtrées
    os.makedirs(config["data"]["raw_dir"], exist_ok=True)
    raw_path = os.path.join(config["data"]["raw_dir"], "banking77_filtered.csv")
    df_filtered.to_csv(raw_path, index=False)
    print(f" Données brutes sauvegardées ({len(df_filtered)} lignes) -> {raw_path}")

    # Split Stratifié Train / Test (80% / 20%)
    train_df, test_df = train_test_split(
        df_filtered,
        test_size=config["data"]["test_size"],
        random_state=config["data"]["random_state"],
        stratify=df_filtered["category"]
    )

    # Sauvegarde dans data/processed/
    os.makedirs(config["data"]["processed_dir"], exist_ok=True)
    train_path = os.path.join(config["data"]["processed_dir"], "train.csv")
    test_path = os.path.join(config["data"]["processed_dir"], "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f" Données d'entraînement : {len(train_df)} lignes -> {train_path}")
    print(f" Données de test : {len(test_df)} lignes -> {test_path}")


if __name__ == "__main__":
    fetch_and_prepare_data()