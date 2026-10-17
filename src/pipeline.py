import os
import re
import time
import yaml
import mlflow.sklearn
import numpy as np

from src.llm_fallback import LLMFallbackRouter
from src.embeddings import TextEmbedder


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HybridTicketClassifier:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        ml_cfg = self.config.get("ml", self.config.get("model", {}))
        self.threshold = ml_cfg.get("confidence_threshold", 0.75)
        
        # 1. Chargement du Champion TF-IDF
        tfidf_model_name = ml_cfg.get("registered_model_name", "TicketClassifierChampion")
        try:
            self.tfidf_model = mlflow.sklearn.load_model(f"models:/{tfidf_model_name}@champion")
        except Exception:
            self.tfidf_model = mlflow.sklearn.load_model(f"models:/{tfidf_model_name}/1")

        # 2. Chargement du Champion Embeddings
        dense_model_name = "DenseTicketClassifierChampion"
        try:
            self.dense_model = mlflow.sklearn.load_model(f"models:/{dense_model_name}@champion")
        except Exception:
            self.dense_model = mlflow.sklearn.load_model(f"models:/{dense_model_name}/1")

        self.embedder = TextEmbedder()
        self.classes = self.tfidf_model.classes_
        self.llm_router = LLMFallbackRouter(config_path=config_path)

    def is_deterministic_out_of_scope(self, text: str) -> tuple[bool, str]:
        cleaned = text.strip()
        if not cleaned:
            return True, "Ticket vide ou ne contenant que des espaces."
        if not re.search(r"[a-zA-Z0-9]", cleaned):
            return True, "Ticket composé exclusivement de symboles ou caractères spéciaux."
        if re.fullmatch(r"[\d\s,.-]+", cleaned):
            return True, "Ticket composé uniquement de valeurs numériques sans contexte explicatif."
        if len(cleaned.split()) < 2:
            return True, "Message trop court (mot isolé) sans intention bancaire explicite."
        return False, ""

    def predict(self, text: str, engine: str = "tfidf") -> dict:
        t0 = time.perf_counter()

        # Niveau 1 : Garde-fou déterministe
        is_oos, oos_reason = self.is_deterministic_out_of_scope(text)
        if is_oos:
            return {
                "ticket": text,
                "category": "out_of_scope",
                "confidence": 0.0,
                "routed_by": "Rule_Engine_Guardrail",
                "reason": oos_reason,
                "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
                "engine_used": engine,
            }

        # Niveau 2 : Inférence ML selon le moteur sélectionné
        if engine == "dense":
            # Encodage sémantique dense (384d)
            dense_vec = self.embedder.encode([text], show_progress=False)
            probabilities = self.dense_model.predict_proba(dense_vec)[0]
            engine_label = "Dense_MiniLM_Champion"
        else:
            # Pipeline Pipeline TF-IDF
            probabilities = self.tfidf_model.predict_proba([text])[0]
            engine_label = "TFIDF_LinearSVC_Champion"

        max_idx = int(np.argmax(probabilities))
        max_confidence = float(probabilities[max_idx])
        predicted_category = self.classes[max_idx]

        sorted_indices = np.argsort(probabilities)[::-1]
        top_candidates = [
            {"category": self.classes[idx], "score": round(float(probabilities[idx]), 4)}
            for idx in sorted_indices[:2]
        ]

        # Fast-Path si confiance suffisante
        if max_confidence >= self.threshold:
            lat = (time.perf_counter() - t0) * 1000
            return {
                "ticket": text,
                "category": predicted_category,
                "confidence": round(max_confidence, 4),
                "routed_by": engine_label,
                "reason": f"Confiance statistique élevée ({max_confidence:.2%}) >= seuil ({self.threshold:.0%}).",
                "latency_ms": round(lat, 2),
                "engine_used": engine,
            }

        # Niveau 3 : Slow-Path LLM Fallback
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
            "engine_used": engine,
        }