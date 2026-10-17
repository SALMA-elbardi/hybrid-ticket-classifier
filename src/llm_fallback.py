import json
import yaml
from openai import OpenAI


def load_config(config_path="config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


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
    def __init__(self, config_path="config.yaml"):
        config = load_config(config_path)
        self.model_name = config["llm"]["model_name"]
        self.base_url = config["llm"]["base_url"]
        self.client = OpenAI(base_url=self.base_url, api_key="ollama")

    def route_ticket(self, text: str, allowed_categories: list[str], top_candidates: list[dict]) -> dict:
        if top_candidates and len(top_candidates) >= 2:
            candidate_names = [c["category"] for c in top_candidates[:2]]
        elif top_candidates:
            candidate_names = [top_candidates[0]["category"]]
        else:
            candidate_names = allowed_categories

        candidate_names_with_oos = candidate_names + ["out_of_scope"]

        few_shot_context = "\n".join(
            [f'- "{cat}": reference intent -> "{CATEGORY_DEFINITIONS_AND_EXAMPLES.get(cat, "")}"'
             for cat in candidate_names]
        )

        system_prompt = (
            "You are a strict banking ticket classifier.\n"
            "Your priority is to identify whether the text contains an actual, explicit banking request.\n\n"
            "Available categories:\n"
            f"{few_shot_context}\n"
            '- "out_of_scope": ANY text that does NOT clearly ask for a banking action (e.g. personal names, chit-chat, greetings, questions about other topics, random sentences).\n\n'
            "Strict Rules:\n"
            '1. If the message does not contain a clear problem or question related to the banking categories, you MUST classify it as "out_of_scope".\n'
            '2. Do NOT force a banking category onto irrelevant text.\n'
            f"3. You must select strictly ONE category from this list: {json.dumps(candidate_names_with_oos)}.\n\n"
            'Return ONLY valid JSON: {"category": "<chosen_category>", "reason": "<short justification>"}'
        )

        user_content = f'Customer Message to evaluate: "{text}"'

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.0,
            )

            raw_reply = response.choices[0].message.content.strip()

            if "```" in raw_reply:
                raw_reply = raw_reply.split("```")[1]
                if raw_reply.startswith("json"):
                    raw_reply = raw_reply[4:].strip()

            parsed = json.loads(raw_reply)

            if parsed.get("category") not in candidate_names_with_oos:
                parsed["category"] = "out_of_scope"
                parsed["reason"] = "Text does not match any valid banking domain intent."

            return parsed

        except Exception as e:
            return {
                "category": "out_of_scope",
                "reason": f"Fallback error recovery: {str(e)}",
            }