import json
import yaml
from openai import OpenAI


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# Dictionnaire de Few-Shot contextuel issu du dataset Banking77
CATEGORY_DEFINITIONS_AND_EXAMPLES = {
    "card_arrival": "When will my new physical plastic card arrive in the mail or courier?",
    "card_linking": "How do I link, connect, or attach my existing card to the app account?",
    "extra_charge_on_statement": "There is an unknown, duplicate, or extra charge appearing on my account statement.",
    "pending_transfer": "I sent a money transfer or wire, but the status is still pending or delayed.",
    "transfer_timing": "How long does an international or standard transfer take to arrive?",
    "balance_not_updated_after_cheque_or_cash_deposit": "I deposited funds or a cheque but my current balance did not update.",
    "card_not_working": "My physical or virtual card is declined or not working at the terminal.",
    "top_up_reverted": "My top-up was canceled, reversed, or sent back to my source account.",
}


class LLMFallbackRouter:
    """
    Routeur d'arbitrage exploitant Ollama (llama3.2:3b)
    avec Few-Shot contextuel et désambiguïsation strictement contrainte.
    """

    def __init__(self, config_path="config.yaml"):
        config = load_config(config_path)
        self.model_name = config["llm"]["model_name"]
        self.base_url = config["llm"]["base_url"]
        self.client = OpenAI(base_url=self.base_url, api_key="ollama")

    def route_ticket(self, text: str, allowed_categories: list[str], top_candidates: list[dict]) -> dict:
        # Règle 2 : Constrained Disambiguation
        # Si le modèle ML a identifié au moins 2 candidats, on force le LLM à choisir UNIQUEMENT entre eux
        if top_candidates and len(top_candidates) >= 2:
            candidate_names = [c["category"] for c in top_candidates[:2]]
        elif top_candidates:
            candidate_names = [top_candidates[0]["category"]]
        else:
            candidate_names = allowed_categories

        # Règle 1 : Few-Shot Prompting ciblé sur les classes candidates
        few_shot_context = "\n".join(
            [f'- "{cat}": reference intent example -> "{CATEGORY_DEFINITIONS_AND_EXAMPLES.get(cat, "")}"'
             for cat in candidate_names]
        )

        system_prompt = (
            "You are a specialized banking intent arbitration agent.\n"
            "An ML model detected ambiguity and narrowed the prediction down to two candidate classes.\n"
            f"Candidate classes with reference definitions:\n{few_shot_context}\n\n"
            f"Constraint: You MUST select exactly ONE category strictly from this restricted list: {json.dumps(candidate_names)}.\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            '{"category": "<chosen_category>", "reason": "<one concise sentence explaining why this candidate matches better>"}'
        )

        user_content = (
            f'Customer Ticket: "{text}"\n'
            f"Candidate scores from ML: {json.dumps(top_candidates[:2])}\n"
            "Decide the true category based on the exact intent."
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,  # Déterminisme maximal pour éviter les hallucinations
            )

            raw_reply = response.choices[0].message.content.strip()

            # Nettoyage des balises Markdown
            if "```" in raw_reply:
                raw_reply = raw_reply.split("```")[1]
                if raw_reply.startswith("json"):
                    raw_reply = raw_reply[4:].strip()

            parsed = json.loads(raw_reply)

            # Sécurité stricte : si la catégorie n'est pas dans les candidats autorisés, repli sur le top-1 ML
            if parsed.get("category") not in candidate_names:
                parsed["category"] = candidate_names[0]
                parsed["reason"] = "Forced fallback to highest ML candidate due to unconstrained LLM output."

            return parsed

        except Exception as e:
            return {
                "category": candidate_names[0] if candidate_names else allowed_categories[0],
                "reason": f"Fallback error recovery: {str(e)}",
            }


if __name__ == "__main__":
    router = LLMFallbackRouter()
    sample_text = "I asked for a refund last week, but yesterday I saw another unknown charge."
    top_candidates = [
        {"category": "extra_charge_on_statement", "score": 0.44},
        {"category": "pending_transfer", "score": 0.38},
    ]

    print("--- Test LLMFallbackRouter avec Few-Shot et Constrained Disambiguation ---")
    result = router.route_ticket(sample_text, list(CATEGORY_DEFINITIONS_AND_EXAMPLES.keys()), top_candidates)
    print(json.dumps(result, indent=2))