import time
import yaml
import numpy as np
import mlflow.sklearn

from src.llm_fallback import LLMFallbackRouter


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class HybridTicketClassifier:
    def __init__(self, config_path="config.yaml"):
        self.config = load_config(config_path)
        self.threshold = self.config["ml"]["confidence_threshold"]
        model_name = self.config["ml"]["registered_model_name"]

        print(f" Chargement du modèle Champion depuis MLflow ('{model_name}')...")
        # Chargement direct de la dernière version du modèle enregistré
        self.ml_model = mlflow.sklearn.load_model(f"models:/{model_name}/latest")
        self.classes = list(self.ml_model.classes_)

        print(" Initialisation du Fallback LLM (Ollama)...")
        self.llm_router = LLMFallbackRouter(config_path=config_path)
        print(" Pipeline Hybride opérationnel !\n")

    def predict(self, text: str) -> dict:
        t0 = time.perf_counter()

        # 1. Calcul des probabilités par le modèle ML
        probabilities = self.ml_model.predict_proba([text])[0]
        max_idx = int(np.argmax(probabilities))
        confidence = float(probabilities[max_idx])
        ml_predicted_class = self.classes[max_idx]

        # Extraction des deux meilleures options en cas d'arbitrage
        top_2_indices = np.argsort(probabilities)[::-1][:2]
        top_candidates = [
            {"category": self.classes[idx], "score": round(float(probabilities[idx]), 3)}
            for idx in top_2_indices
        ]

        # 2. Règle de décision : Fast-Path (ML) vs Slow-Path (LLM)
        if confidence >= self.threshold:
            latency = (time.perf_counter() - t0) * 1000
            return {
                "ticket": text,
                "category": ml_predicted_class,
                "confidence": round(confidence, 3),
                "routed_by": "ML_Champion (LinearSVC)",
                "reason": "High confidence prediction",
                "latency_ms": round(latency, 2),
            }
        else:
            llm_decision = self.llm_router.route_ticket(
                text=text,
                allowed_categories=self.classes,
                top_candidates=top_candidates,
            )
            latency = (time.perf_counter() - t0) * 1000
            return {
                "ticket": text,
                "category": llm_decision.get("category"),
                "confidence": round(confidence, 3),
                "routed_by": "LLM_Fallback (llama3.2)",
                "reason": llm_decision.get("reason"),
                "latency_ms": round(latency, 2),
            }


if __name__ == "__main__":
    classifier = HybridTicketClassifier()

    test_tickets = [
        "When will my new credit card arrive in the mail?",
        "I was billed an extra charge on my card but the app still says transfer pending.",
    ]

    print("--- DÉMONSTRATION DU ROUTAGE HYBRIDE ---")
    for ticket in test_tickets:
        res = classifier.predict(ticket)
        print(f"\nTicket      : {res['ticket']}")
        print(f"Catégorie   : {res['category']}")
        print(f"Confiance ML: {res['confidence']}")
        print(f"Routé par   : {res['routed_by']}")
        print(f"Raison      : {res['reason']}")
        print(f"Latence     : {res['latency_ms']} ms")