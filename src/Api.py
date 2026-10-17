import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, make_asgi_app

from src.pipeline import HybridTicketClassifier

app = FastAPI(
    title="Hybrid-TicketClassifier API",
    description="Production API with asymmetric routing (Fast-Path ML vs Slow-Path LLM) and Prometheus metrics",
    version="1.0.0"
)

# --- Métriques Prometheus ---
REQUEST_COUNT = Counter(
    "ticket_classifier_requests_total",
    "Nombre total de requêtes de classification",
    ["engine", "routed_by", "category"]
)

REQUEST_LATENCY = Histogram(
    "ticket_classifier_latency_seconds",
    "Latence de bout en bout de l'inférence en secondes",
    ["routed_by"],
    buckets=[0.005, 0.015, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

# Chargement du classifieur
classifier = HybridTicketClassifier()

# Montage du endpoint /metrics pour Prometheus
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


class TicketRequest(BaseModel):
    text: str = Field(..., min_length=1, example="When will my new plastic card arrive in the mail?")
    engine: str = Field("dense", example="dense", description="'dense' (MiniLM) ou 'tfidf' (N-grams)")


class TicketResponse(BaseModel):
    ticket: str
    category: str
    confidence: float
    routed_by: str
    reason: str
    latency_ms: float
    engine_used: str


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "Hybrid-TicketClassifier"}


@app.post("/predict", response_model=TicketResponse, tags=["Inference"])
def predict_ticket(request: TicketRequest):
    if request.engine not in ["dense", "tfidf"]:
        raise HTTPException(status_code=400, detail="Moteur invalide. Utilisez 'dense' ou 'tfidf'.")

    t0 = time.perf_counter()
    result = classifier.predict(request.text, engine=request.engine)
    latency_sec = time.perf_counter() - t0

    # Incrémentation des métriques
    REQUEST_COUNT.labels(
        engine=result.get("engine_used", request.engine),
        routed_by=result.get("routed_by", "unknown"),
        category=result.get("category", "unknown")
    ).inc()

    REQUEST_LATENCY.labels(
        routed_by=result.get("routed_by", "unknown")
    ).observe(latency_sec)

    return result