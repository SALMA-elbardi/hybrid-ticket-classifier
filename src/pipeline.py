import os
import re
import time
import yaml
import mlflow.sklearn
import numpy as np

from src.llm_fallback import LLMFallbackRouter


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HybridTicketClassifier:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        
        # Récupération sécurisée depuis le bloc 'ml' (ou 'model' / 'mlflow' par repli)
        ml_cfg = self.config.get("ml", self.config.get("model", {}))
        self.threshold = ml_cfg.get("confidence_threshold", 0.75)
        self.model_name = ml_cfg.get(
            "registered_model_name",
            self.config.get("mlflow", {}).get("registered_model_name", "TicketClassifierChampion")
        )
        
        # Chargement du modèle Champion depuis MLflow
        model_uri = f"models:/{self.model_name}@champion"
        try:
            self.ml_model = mlflow.sklearn.load_model(model_uri)
        except Exception:
            # Repli si l'alias champion n'est pas encore posé
            self.ml_model = mlflow.sklearn.load_model(f"models:/{self.model_name}/1")

        self.classes = self.ml_model.classes_
        self.llm_router = LLMFallbackRouter(config_path=config_path)

    def is_deterministic_out_of_scope(self, text: str) -> tuple[bool, str]:
        cleaned = text.strip()
        
        # 1. Texte vide
        if not cleaned:
            return True, "Ticket vide ou ne contenant que des espaces."

        # 2. Caractères spéciaux sans texte informatif
        if not re.search(r"[a-zA-Z0-9]", cleaned):
            return True, "Ticket composé exclusivement de symboles ou caractères spéciaux."

        # 3. Uniquement des chiffres
        if re.fullmatch(r"[\d\s,.-]+", cleaned):
            return True, "Ticket composé uniquement de valeurs numériques sans contexte explicatif."

        # 4. Mots isolés ou salutations directes (< 2 mots ou < 8 caractères utiles)
        tokens = cleaned.split()
        if len(tokens) <= 1:
            return True, "Message contenant un seul mot ou nom isolé sans demande métier explicite."

        common_greetings = {"hello", "hi", "hey", "bonjour", "salut", "good morning", "good evening", "thanks", "merci"}
        if cleaned.lower() in common_greetings:
            return True, "Message composé uniquement d'une salutation sans question bancaire."

        return False, ""

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()

        # Garde-fou Niveau 1 : Détection immédiate hors-domaine
        is_oos, oos_reason = self.is_deterministic_out_of_scope(text)
        if is_oos:
            return {
                "ticket": text,
                "category": "out_of_scope",
                "confidence": 0.0,
                "routed_by": "Rule_Engine_Guardrail",
                "reason": oos_reason,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
            }

        # Niveau 2 : Inférence statistique ML
        probabilities = self.ml_model.predict_proba([text])[0]
        max_idx = int(np.argmax(probabilities))
        max_confidence = float(probabilities[max_idx])
        predicted_category = self.classes[max_idx]

        # Préparation du top-2 des candidats ML
        sorted_indices = np.argsort(probabilities)[::-1]
        top_candidates = [
            {"category": self.classes[idx], "score": round(float(probabilities[idx]), 4)}
            for idx in sorted_indices[:2]
        ]

        # Fast-Path si la confiance dépasse le seuil
        if max_confidence >= self.threshold:
            lat = (time.perf_counter() - t0) * 1000
            return {
                "ticket": text,
                "category": predicted_category,
                "confidence": round(max_confidence, 4),
                "routed_by": "ML_Champion",
                "reason": f"Confiance statistique élevée ({max_confidence:.2%}) >= seuil ({self.threshold:.0%}).",
                "latency_ms": round(lat, 2),
            }

        # Niveau 3 : Slow-Path LLM Fallback (Désambiguïsation ou détection sémantique out_of_scope)
        llm_decision = self.llm_router.route_ticket(
            text=text,
            allowed_categories=list(self.classes),
            top_candidates=top_candidates,
        )

        lat = (time.perf_counter() - t0) * 1000
        return {
            "ticket": text,
            "category": llm_decision.get("category", "out_of_scope"),
            "confidence": round(max_confidence, 4),
            "routed_by": "LLM_Fallback",
            "reason": llm_decision.get("reason", "Arbitrage contextuel."),
            "latency_ms": round(lat, 2),
        }