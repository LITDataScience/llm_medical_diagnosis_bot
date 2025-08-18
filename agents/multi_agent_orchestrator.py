import os
import re
from typing import Dict, List, Optional, Tuple

try:
    from litellm import completion
    LLM_AVAILABLE = True
except Exception:
    LLM_AVAILABLE = False
from rag.context_engineering import build_context_for_llm


class ConversationOrchestrator:
    """
    Orchestrates a multi-agent pipeline for the medical chat:
    - Symptom extraction (LLM-assisted with heuristic fallback)
    - Follow-up question generation (LLM + heuristic)
    - Diagnosis and explanation (model + RAG + LLM)
    """

    def __init__(self, chatbot, retriever):
        self.chatbot = chatbot
        self.retriever = retriever
        self.min_turns_before_diagnosis = 2
        self.min_symptoms_before_diagnosis = 2

    def route_turn(self, user_input: str, conversation_state: Dict) -> Dict:
        """Route a user turn to either ask a follow-up question or perform diagnosis."""
        extracted_symptoms = self.extract_symptoms_llm_first(user_input)
        conversation_state['mentioned_symptoms'].extend(extracted_symptoms)
        conversation_state['turn_count'] += 1
        # Extract structured intake facts (e.g., duration) from this turn
        self._update_intake_facts_from_text(user_input, conversation_state)

        # Decide if ready to diagnose
        ready = self._is_ready_for_diagnosis(conversation_state)
        conversation_state['ready_to_diagnose'] = ready

        if not ready:
            question = self.generate_follow_up_question(conversation_state, extracted_symptoms)
            conversation_state['pending_question'] = question
            return {
                'response': question,
                'state': conversation_state,
                'extracted_symptoms': extracted_symptoms,
                'diagnosis': None,
            }

        # Diagnose
        diagnosis_result = self.chatbot.get_diagnosis(
            text_description=user_input,
            symptom_ids=None,
            patient_history=conversation_state.get('patient_history', {})
        )
        conversation_state['diagnosis'] = diagnosis_result

        # Build RAG context and explanation with LLM
        explanation = self.generate_explanation_with_rag(user_input, diagnosis_result, conversation_state)
        response = self.chatbot.format_diagnosis_response(diagnosis_result)
        if explanation:
            response += f"\n\n{explanation}"

        return {
            'response': response,
            'state': conversation_state,
            'extracted_symptoms': extracted_symptoms,
            'diagnosis': diagnosis_result,
        }

    def extract_symptoms_llm_first(self, user_input: str) -> List[int]:
        """Use LLM to extract symptoms mapped to known list; fallback to heuristic extractor."""
        # Access the simple list
        symptom_list = getattr(self.chatbot, 'symptom_list', [])
        if not symptom_list:
            return self.chatbot.extract_symptoms(user_input)

        if not LLM_AVAILABLE or os.environ.get('OPENAI_API_KEY') is None and os.environ.get('LITELLM_MODEL') is None:
            return self.chatbot.extract_symptoms(user_input)

        system_prompt = (
            "You are a medical intake assistant. From the given user text, identify which of the provided "
            "canonical symptoms are explicitly present. Return a comma-separated list of the exact matching "
            "canonical symptom names only. If none match, return an empty string."
        )
        user_prompt = f"User text: {user_input}\nCanonical symptoms: {', '.join(symptom_list[:200])}"

        try:
            model = os.environ.get('LITELLM_MODEL', 'gpt-4o-mini')
            resp = completion(model=model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ], max_tokens=128, temperature=0.1)
            text = resp['choices'][0]['message']['content'] if isinstance(resp, dict) else str(resp)
            names = [n.strip() for n in text.split(',') if n.strip()]
            indices = []
            name_to_idx = {name.lower(): i for i, name in enumerate(symptom_list)}
            for n in names:
                idx = name_to_idx.get(n.lower())
                if idx is not None:
                    indices.append(idx)
            # Fallback to heuristic if LLM returned nothing
            if not indices:
                indices = self.chatbot.extract_symptoms(user_input)
            return indices
        except Exception:
            return self.chatbot.extract_symptoms(user_input)

    def generate_follow_up_question(self, conversation_state: Dict, extracted_symptoms: List[int]) -> str:
        """Create a targeted follow-up question using dynamic slot-filling logic + RAG/LLM."""
        # Retrieve related info for context
        try:
            context_docs = self.retriever.retrieve_for_symptoms(extracted_symptoms, top_k=3)
        except Exception:
            context_docs = []

        # Dynamic intake slots (ask what's still missing)
        facts = conversation_state.get('intake_facts', {})
        prioritized_questions = [
            ("duration_days", "Besides the cough, do you have a fever, runny nose, or shortness of breath?"),
            ("fever_present", "Have you had a fever with the cough?"),
            ("breath_difficulty", "Are you experiencing shortness of breath or wheezing?"),
            ("sputum", "Is your cough producing any phlegm? If so, what color?"),
            ("exposure", "Any recent travel or close contact with someone who was ill?"),
        ]
        # If duration is missing, we should ask about it; if present, skip it
        if 'duration_days' not in facts:
            dynamic_question = "How long have you been experiencing these symptoms?"
        else:
            # Pick first unfilled slot after duration
            dynamic_question = None
            for key, q in prioritized_questions[1:]:
                if key not in facts:
                    dynamic_question = q
                    break
            if dynamic_question is None:
                dynamic_question = "Can you describe any other symptoms or anything that makes it better or worse?"

        base_question = dynamic_question

        if not LLM_AVAILABLE or os.environ.get('OPENAI_API_KEY') is None and os.environ.get('LITELLM_MODEL') is None:
            return base_question

        try:
            model = os.environ.get('LITELLM_MODEL', 'gpt-4o-mini')
            system_prompt = (
                "You are a triage agent. Ask one short, specific medical follow-up question to clarify the case. "
                "Prefer questions that disambiguate the most likely respiratory/viral/infectious/other differentials. "
                "Keep it to a single sentence."
            )
            symptom_names = [self.chatbot.symptom_list[i] for i in extracted_symptoms if i < len(self.chatbot.symptom_list)]
            context_text = build_context_for_llm(
                user_input="",
                symptom_names=symptom_names,
                retrieved_docs=context_docs,
                patient_profile=conversation_state.get('patient_history', {}),
            )
            user_prompt = (
                f"{context_text}\nFacts known: {conversation_state.get('intake_facts', {})}\nDefault question if unsure: {base_question}"
            )
            resp = completion(model=model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ], max_tokens=96, temperature=0.2)
            question = resp['choices'][0]['message']['content'].strip()
            # Ensure it's one sentence and not too long
            return question.split('\n')[0][:220]
        except Exception:
            return base_question

    def generate_explanation_with_rag(self, user_input: str, diagnosis_result: Dict, conversation_state: Dict) -> Optional[str]:
        if not LLM_AVAILABLE or os.environ.get('OPENAI_API_KEY') is None and os.environ.get('LITELLM_MODEL') is None:
            return None

        # Build small context from RAG for top diseases and symptoms
        top_indices = [idx.item() for idx in diagnosis_result['top_diseases'][0][:3]]
        disease_names = []
        model_disease_names = getattr(self.chatbot.enhanced_model, 'disease_names', None)
        for idx in top_indices:
            if model_disease_names and idx < len(model_disease_names):
                disease_names.append(model_disease_names[idx])
            elif idx < len(self.chatbot.disease_list):
                disease_names.append(self.chatbot.disease_list[idx])

        docs = self.retriever.retrieve_for_diseases_by_name(disease_names, top_k=3)
        context_text = build_context_for_llm(
            user_input=user_input,
            symptom_names=[self.chatbot.symptom_list[i] for i in conversation_state.get('mentioned_symptoms', []) if i < len(self.chatbot.symptom_list)],
            retrieved_docs=docs,
            patient_profile=conversation_state.get('patient_history', {}),
        )

        try:
            model = os.environ.get('LITELLM_MODEL', 'gpt-4o-mini')
            system_prompt = (
                "You are a clinical AI assistant. Given top candidate diagnoses and brief medical context, "
                "explain succinctly why these conditions are plausible based on the reported symptoms and profile. "
                "Keep it under 4 sentences and avoid prescribing. Add a brief safety notice."
            )
            user_prompt = (
                f"Top differentials: {', '.join(disease_names)}\n{context_text}"
            )
            resp = completion(model=model, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ], max_tokens=180, temperature=0.2)
            return resp['choices'][0]['message']['content'].strip()
        except Exception:
            return None

    def _is_ready_for_diagnosis(self, conversation_state: Dict) -> bool:
        if conversation_state.get('turn_count', 0) < self.min_turns_before_diagnosis:
            return False
        if len(conversation_state.get('mentioned_symptoms', [])) < self.min_symptoms_before_diagnosis:
            return False
        return True

    # --------- Intake extraction helpers ---------
    def _update_intake_facts_from_text(self, text: str, conversation_state: Dict) -> None:
        facts = conversation_state.setdefault('intake_facts', {})
        new_facts = self._parse_intake_facts(text)
        for k, v in new_facts.items():
            # Do not overwrite existing facts unless they are empty
            if k not in facts and v is not None:
                facts[k] = v

    def _parse_intake_facts(self, text: str) -> Dict:
        text_l = text.lower()
        facts: Dict[str, Optional[object]] = {}

        # Duration extraction (e.g., "for 10 days", "since 2 weeks", "3 months", "a week")
        duration_patterns = [
            r"(?:for|from|since|about|around|over|past)?\s*(\d+(?:\.\d+)?)\s*(day|days|week|weeks|month|months|hour|hours|hr|hrs)",
            r"a\s*(day|week|month)",
            r"couple\s*of\s*(days|weeks|months)",
            r"few\s*(days|weeks|months)",
        ]
        duration_days = None
        for pat in duration_patterns:
            m = re.search(pat, text_l)
            if not m:
                continue
            if pat.startswith('a'):
                unit = m.group(1)
                num = 1.0
            elif 'couple' in pat:
                unit = m.group(1)
                num = 2.0
            elif 'few' in pat:
                unit = m.group(1)
                num = 3.0
            else:
                num = float(m.group(1))
                unit = m.group(2)

            if unit.startswith('day'):
                duration_days = num
            elif unit.startswith('week'):
                duration_days = num * 7
            elif unit.startswith('month'):
                duration_days = num * 30
            elif unit in {"hour", "hours", "hr", "hrs"}:
                duration_days = max(1.0/24.0, num/24.0)
            if duration_days is not None:
                break

        if duration_days is not None:
            try:
                facts['duration_days'] = float(duration_days)
            except Exception:
                pass

        # Fever presence
        if 'fever' in text_l:
            if re.search(r"no\s+fever|without\s+fever|afebrile", text_l):
                facts['fever_present'] = False
            else:
                facts.setdefault('fever_present', True)

        # Shortness of breath / wheeze
        if any(k in text_l for k in ['shortness of breath', 'breathless', 'breathlessness', 'wheezing', 'wheeze']):
            facts['breath_difficulty'] = True

        # Sputum
        if any(k in text_l for k in ['phlegm', 'sputum', 'mucus']):
            facts['sputum'] = True

        return facts


