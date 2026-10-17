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
    .badge-dense {
        background-color: #d1ecf1;
        color: #0c5460;
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
    .badge-guardrail {
        background-color: #e2e3e5;
        color: #383d41;
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

st.sidebar.markdown("### ⚙️ Choix du Fast-Path")
selected_engine = st.sidebar.radio(
    "Moteur d'extraction de caractéristiques :",
    options=["dense", "tfidf"],
    format_func=lambda x: "🧠 Dense Embeddings (MiniLM-L6-v2)" if x == "dense" else "⚡ TF-IDF (N-grammes creux)",
    help="Comparez en direct la robustesse sémantique et la latence d'inférence."
)

st.sidebar.markdown("### ⚙️ Configuration Actuelle")
st.sidebar.info(f"""
- **Seuil d'aiguillage ($\tau$)** : `{classifier.threshold * 100:.0f} %`
- **Moteur Fast-Path Actif** : `{"Sentence-Transformers (MiniLM)" if selected_engine == "dense" else "TF-IDF + LinearSVC"}`
- **Fallback Slow-Path** : `Llama 3.2:3b (Ollama)`
""")

st.sidebar.markdown("### 📊 Benchmark des Champions MLflow")
if selected_engine == "dense":
    st.sidebar.markdown("""
    - **Modèle Champion** : `Dense LogisticRegression`
    - **Accuracy / F1-Macro** : `96.1 %` / `0.962`
    - **Latence Fast-Path** : `~7-15 ms`
    - **Résistance Sémantique** : `Élevée (comprend le contexte)`
    """)
else:
    st.sidebar.markdown("""
    - **Modèle Champion** : `Calibrated LinearSVC`
    - **Accuracy / F1-Macro** : `95.0 %` / `0.952`
    - **Latence Fast-Path** : `~1-2 ms`
    - **Résistance Sémantique** : `Vulnérable aux mots isolés`
    """)

# --- CORPS PRINCIPAL ---
st.title("🏦 Hybrid-TicketClassifier — Portail Support Client")
st.markdown("""
Architecture en cascade : traitement haute fréquence en quelques millisecondes 
via le modèle statistique supervisé, et délégation vers le modèle de langage local pour désambiguïser les cas complexes ou hors-domaine.
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
        "hey my name is card",
        "my card is not arrived yet ",
        "I deposited a cheque but my balance hasn't updated.",
        "i have an extra fee on my bank statement"
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
                result = classifier.predict(user_text.strip(), engine=selected_engine)

            routed_by = result["routed_by"]

            # Badges conditionnels selon le moteur ayant tranché
            if "Rule_Engine" in routed_by:
                st.markdown(
                    '<span class="badge-guardrail"> Out-of-Scope</span>',
                    unsafe_allow_html=True,
                )
            elif "Dense" in routed_by:
                st.markdown(
                    '<span class="badge-dense">🧠 FAST-PATH ACTIVÉ (Dense MiniLM + Champion MLflow)</span>',
                    unsafe_allow_html=True,
                )
            elif "TFIDF" in routed_by or "ML_Champion" in routed_by:
                st.markdown(
                    '<span class="badge-fast">⚡ FAST-PATH ACTIVÉ (TF-IDF + LinearSVC MLflow)</span>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    '<span class="badge-slow">🤖 SLOW-PATH ACTIVÉ (Ollama LLM Fallback)</span>',
                    unsafe_allow_html=True,
                )

            st.write("")

            # Formatage de la latence
            latency_val = result["latency_ms"]
            latency_str = f"{latency_val / 1000:.2f} s" if latency_val >= 1000 else f"{latency_val:.1f} ms"

            # Cartes métriques responsives
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
            st.markdown("###  Explication du Routage")
            if "Rule_Engine" in routed_by:
                st.warning(f"**Interception Déterministe :** {result['reason']}")
            elif "ML" in routed_by or "TFIDF" in routed_by or "Dense" in routed_by:
                st.info(
                    f"**Règle métier ({routed_by}) :** Le score de confiance ({result['confidence'] * 100:.1f} %) "
                    f"dépasse le seuil de {classifier.threshold * 100:.0f} %. Traité directement sans solliciter le LLM."
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
with st.expander("🔍 Voir les détails des modèles comparés en production (MLflow Registry)"):
    st.markdown("""
    - **Approche 1 — TF-IDF (Baseline Haute Fréquence)** :
      - Modèle : `Calibrated_LinearSVC` (Version 1 - Champion)
      - Vectorisation : N-grammes creux (1, 2)
      - Latence : `< 2 ms`
      - Limite : Sensible aux mots-clés bancaires hors-contexte.
    - **Approche 2 — Dense Embeddings (Compréhension Sémantique)** :
      - Modèle : `Dense_LogisticRegression` (Enregistré dans `DenseTicketClassifierChampion`)
      - Embedder : `sentence-transformers/all-MiniLM-L6-v2` (Vecteurs denses 384d)
      - Latence : `~8 à 15 ms`
      - Avantage : Distingue le sens global de la phrase et élimine les faux positifs lexicaux.
    - **Slow-Path de Repli** : `Llama 3.2:3b` via Ollama avec formatage JSON contraint et gestion du hors-domaine (`out_of_scope`).
    """)