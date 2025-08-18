from typing import Dict, List


def build_context_for_llm(
    user_input: str,
    symptom_names: List[str],
    retrieved_docs: List[Dict],
    patient_profile: Dict,
) -> str:
    """Compose a compact retrieval-augmented context string for LLM prompts."""
    context_lines = []
    if symptom_names:
        context_lines.append(f"Reported symptoms: {', '.join(symptom_names)}")
    if patient_profile:
        context_lines.append(f"Patient profile: {patient_profile}")
    if user_input:
        context_lines.append(f"Latest user message: {user_input}")

    if retrieved_docs:
        context_lines.append("Relevant medical snippets:")
        for d in retrieved_docs[:5]:
            snippet = (d.get('text') or '')[:220]
            context_lines.append(f"- {d.get('title')}: {snippet}")

    return "\n".join(context_lines)


