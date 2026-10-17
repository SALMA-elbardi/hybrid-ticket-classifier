import time
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from src.pipeline import HybridTicketClassifier

# Initialisation de l'application FastAPI
app = FastAPI(
    title="Hybrid Ticket Classifier API",
    description="Microservice de routage intelligent de tickets bancaires (MLflow Champion + Ollama Fallback)",
    version="1.0.0",
)

# Chargement du pipeline hybride en mémoire 
classifier = None

@app.on_event("startup")
def startup_event():
    global classifier
    classifier = HybridTicketClassifier()

# ---------------------------------------------------------
# Métriques d'observabilité Prometheus
# ---------------------------------------------------------
REQUEST_COUNT = Counter(
    "ticket_classifier_requests_total",
    "Nombre total de requêtes de classification reçues",
    ["routed_by", "category"]
)

LATENCY_HISTOGRAM = Histogram(
    "ticket_classifier_latency_seconds",
    "Latence de traitement par composant (secondes)",
    ["routed_by"],
    buckets=[0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
)

# ---------------------------------------------------------
# Schémas de validation Pydantic
# ---------------------------------------------------------
class TicketRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=5,
        max_length=1000,
        example="When will my new credit card arrive in the mail?"
    )

class TicketResponse(BaseModel):
    ticket: str
    category: str
    confidence: float
    routed_by: str
    reason: str
    latency_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    registered_model_version: str

# ---------------------------------------------------------
# Endpoints de l'API
# ---------------------------------------------------------
@app.get("/health", response_model=HealthResponse, tags=["Observability"])
def health_check():
    """Vérifie la santé de l'API et la disponibilité du modèle ML."""
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le pipeline de classification n'est pas prêt."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "registered_model_version": "TicketClassifierChampion"
    }

@app.post("/api/v1/predict", response_model=TicketResponse, tags=["Inference"])
def predict_ticket(request: TicketRequest):
    """
    Point d'entrée principal pour classifier un ticket bancaire.
    - Fast-Path (ML) : utilisé si confiance >= 0.75
    - Slow-Path (LLM) : sollicité pour arbitrage contextuel si confiance < 0.75
    """
    if classifier is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le classifieur est en cours d'initialisation."
        )

    t0 = time.perf_counter()
    result = classifier.predict(request.text)
    latency_sec = time.perf_counter() - t0

    # Mise à jour des métriques Prometheus
    engine = "ML" if "ML_Champion" in result["routed_by"] else "LLM"
    REQUEST_COUNT.labels(routed_by=engine, category=result["category"]).inc()
    LATENCY_HISTOGRAM.labels(routed_by=engine).observe(latency_sec)

    return result

@app.get("/metrics", tags=["Observability"])
def metrics():
    """Expose les compteurs et histogrammes au format standard Prometheus."""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)