import re
from sklearn.base import BaseEstimator, TransformerMixin

# Dictionnaire de décontraction pour l'anglais bancaire
CONTRACTIONS = {
    r"can\'t": "can not",
    r"won\'t": "will not",
    r"n\'t": " not",
    r"\'re": " are",
    r"\'s": " is",
    r"\'d": " would",
    r"\'ll": " will",
    r"\'ve": " have",
    r"\'m": " am",
}


def clean_text(text: str) -> str:
    """
    Nettoie et normalise une chaîne de caractères brute :
    - Mise en minuscules
    - Décontraction des formes verbales et négatives
    - Normalisation des montants monétaires -> amount_token
    - Normalisation des formats de dates -> date_token
    - Normalisation des nombres restants -> number_token
    - Nettoyage des caractères spéciaux et des espaces multiples
    """
    if not isinstance(text, str):
        return ""

    # 1. Mise en minuscules
    text = text.lower()

    # 2. Décontraction
    for pattern, replacement in CONTRACTIONS.items():
        text = re.sub(pattern, replacement, text)

    # 3. Remplacement des montants financiers (ex: $45.99, 50€, £10, 30 usd)
    text = re.sub(
        r"[\$\£\€]\s*\d+([.,]\d+)?|\d+([.,]\d+)?\s*[\$\£\€]|\b\d+([.,]\d+)?\s*(usd|eur|gbp)\b",
        " amount_token ",
        text,
    )

    # 4. Remplacement des dates (ex: 12/04, 2024-05-01, 12/04/2024)
    text = re.sub(r"\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b", " date_token ", text)

    # 5. Remplacement des nombres isolés restants (ex: codes PIN, numéros de compte)
    text = re.sub(r"\b\d+\b", " number_token ", text)

    # 6. Suppression de la ponctuation et caractères non désirés (garde lettres et tokens avec '_')
    text = re.sub(r"[^a-zA-Z_\s]", " ", text)

    # 7. Suppression des espaces multiples et trim
    text = re.sub(r"\s+", " ", text).strip()

    return text


class TextCleaner(BaseEstimator, TransformerMixin):
    """
    Transformeur Scikit-Learn personnalisé pour insérer clean_text
    directement dans un Pipeline MLflow sans fuite de données.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if hasattr(X, "apply"):
            return X.apply(clean_text)
        return [clean_text(doc) for doc in X]


if __name__ == "__main__":
    test_samples = [
        "I can't link my card!! It's not working...",
        "Where's my pending transfer? Won't arrive on time.",
        "Extra charge on statement: $45.99 on 12/04.",
        "I transferred 150 eur but my pin is 4452.",
    ]
    print("--- Test unitaire du préprocesseur  ---")
    cleaner = TextCleaner()
    cleaned = cleaner.transform(test_samples)
    for original, clean in zip(test_samples, cleaned):
        print(f"Brut   : {original}")
        print(f"Nettoyé: {clean}\n")