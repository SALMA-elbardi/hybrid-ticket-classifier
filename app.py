import streamlit as st
import time
from src.pipeline import HybridTicketClassifier

# Configuration générale de la page
st.set_page_config(
    page_title="Hybrid-TicketClassifier | Support IA",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Feuilles de style pour le rendu visuel
st.markdown(
    """
<style>
    .badge-fast {
        background-color: #d4edda;
        color: #155724;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.9rem;
    }
    .badge-slow {
        background-color: #fff3cd;
        color: #856404;
        padding: 6px 14px;
        border-radius: 6px;
        font-weight: 700;
        display: inline-block;
        font-size: 0.9rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Chargement unique et mise en cache du pipeline hybride
@st.cache_resource
def load_classifier():
    return HybridTicketClassifier()

with st.spinner("Initialisation du système de classification hybride..."):
    classifier = load_classifier()

# --- SIDEBAR : Supervision & Paramètres Techniques ---
st.sidebar.title("Paramètres & Architecture")
st.sidebar.caption("Supervision du routage en cascade")

st.sidebar.markdown("### ⚙️ Modèles Actifs")
st.sidebar.info(f"""
- **Seuil d'aiguillage ($\tau$)** : `{classifier.threshold * 100:.0f} %`
- **Fast-Path** : `LinearSVC (MLflow Registry)`
- **Slow-Path** : `Llama 3.2:3b (Ollama)`
""")

st.sidebar.markdown("### 📊 Résultats du Benchmark Réel")
st.sidebar.markdown("""
- **Absorption Fast-Path** : `82.5 %`
- **Recours Fallback LLM** : `17.5 %`
- **Latence Fast-Path** : `~8 ms`
- **Accuracy globale** : `95.0 %`
""")

# --- CORPS PRINCIPAL ---
st.title("🏦 Hybrid-TicketClassifier — Portail Support Client")
st.markdown("""
Architecture en cascade : traitement haute fréquence en quelques millisecondes 
via le modèle statistique supervisé, et délégation vers le modèle de langage local pour désambiguïser les cas complexes.
""")

st.divider()

col_input, col_output = st.columns([1, 1], gap="large")

with col_input:
    st.subheader("📥 Réception du Ticket")

    preset_tickets = [
        "--- Sélectionner un ticket type ---",
        "When will my new plastic card arrive in the mail?",
        "I was charged an extra fee on my statement that I didn't authorize.",
        "I sent money to a friend yesterday but it is still showing as pending.",
        "How do I link my virtual card with my phone payment app?",
        "I tried to transfer money but got an unexpected fee and the payment is delayed.",
    ]

    selected_preset = st.selectbox("Tickets d'exemples pré-configurés :", preset_tickets)

    initial_text = "" if selected_preset == preset_tickets[0] else selected_preset
    user_text = st.text_area(
        "Texte du message client :",
        value=initial_text,
        height=140,
        placeholder="Écrivez le message reçu...",
    )

    btn_classify = st.button("Classifier & Router le Ticket 🚀", type="primary", use_container_width=True)

with col_output:
    st.subheader("📤 Décision & Routage")

    if btn_classify:
        if not user_text.strip():
            st.warning("Veuillez saisir un texte de ticket à analyser.")
        else:
            with st.spinner("Analyse et routage en cours..."):
                result = classifier.predict(user_text.strip())

            is_ml = "ML_Champion" in result["routed_by"]

            # Badge d'aiguillage
            if is_ml:
                st.markdown(
                    '<span class="badge-fast">⚡ FAST-PATH ACTIVÉ (Scikit-Learn MLflow)</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="badge-slow">🤖 SLOW-PATH ACTIVÉ (Ollama LLM Fallback)</span>',
                    unsafe_allow_html=True,
                )

            st.write("")

            # Formatage de la latence (secondes si >= 1000 ms, millisecondes sinon)
            latency_val = result["latency_ms"]
            latency_str = f"{latency_val / 1000:.2f} s" if latency_val >= 1000 else f"{latency_val:.1f} ms"

            # Cartes métriques responsives sans troncature
            st.markdown(
                f"""
            <div style="display: grid; grid-template-columns: 1.5fr 1fr 1fr; gap: 12px; margin: 15px 0;">
                <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px;">
                    <div style="font-size: 0.8rem; color: #6c757d; text-transform: uppercase; font-weight: 600;">Catégorie prédite</div>
                    <div style="font-size: 1.05rem; font-weight: bold; color: #1a1a1a; word-break: break-word; margin-top: 4px;">
                        {result['category']}
                    </div>
                </div>
                <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px;">
                    <div style="font-size: 0.8rem; color: #6c757d; text-transform: uppercase; font-weight: 600;">Confiance ML</div>
                    <div style="font-size: 1.25rem; font-weight: bold; color: #2b5c8f; margin-top: 4px;">
                        {result['confidence'] * 100:.1f} %
                    </div>
                </div>
                <div style="background-color: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 12px;">
                    <div style="font-size: 0.8rem; color: #6c757d; text-transform: uppercase; font-weight: 600;">Latence totale</div>
                    <div style="font-size: 1.25rem; font-weight: bold; color: #1a1a1a; margin-top: 4px;">
                        {latency_str}
                    </div>
                </div>
            </div>
            """,
                unsafe_allow_html=True,
            )

            # Jauge visuelle de confiance
            st.write(f"**Indice de certitude statistique (Seuil : {classifier.threshold * 100:.0f} %)**")
            st.progress(min(max(float(result["confidence"]), 0.0), 1.0))

            # Explication détaillée du routage
            st.markdown("### 📋 Explication du Routage")
            if is_ml:
                st.info(
                    f"**Règle métier :** Le score de confiance ({result['confidence'] * 100:.1f} %) "
                    f"dépasse le seuil de {classifier.threshold * 100:.0f} %. Traité directement avec une latence minimale."
                )
            else:
                st.warning(
                    f"**Arbitrage LLM :** Confiance ML insuffisante ({result['confidence'] * 100:.1f} % < {classifier.threshold * 100:.0f} %).\n\n"
                    f"*{result['reason']}*"
                )
    else:
        st.info("Sélectionnez ou écrivez un ticket à gauche, puis cliquez sur **Classifier & Router le Ticket**.")

st.divider()

# Détails techniques sous accordéon
with st.expander("🔍 Voir les détails du modèle en production (Model Registry MLflow)"):
    st.markdown("""
    - **Modèle déployé** : `Calibrated_LinearSVC` (Version 1 - Champion)
    - **Prétraitement** : Nettoyage regex, normalisation de tokens (`amount_token`, `date_token`, `number_token`), vectorisation TF-IDF.
    - **Calibration de probabilités** : `CalibratedClassifierCV` (méthode sigmoïde / Platt Scaling).
    - **Gouvernance & Registry** : Modèle versionné et servi dynamiquement depuis les artefacts MLflow.
    """)