from typing import Dict, List, Optional


class SimpleMedicalRetriever:
    """
    Extremely lightweight retriever over the model's built-in knowledge graph.
    Uses token overlap scoring to return small context snippets for RAG.
    """

    def __init__(self):
        self.docs: List[Dict] = []
        self._name_to_doc: Dict[str, Dict] = {}

    def index_from_model(self, enhanced_model) -> None:
        kg = getattr(enhanced_model, 'knowledge_graph', None)
        if kg is None:
            return
        self.docs = []
        self._name_to_doc = {}
        for entity in kg.entities.values():
            title = entity.name
            text = entity.description or ''
            synonyms = ', '.join(entity.synonyms or [])
            payload = {
                'id': entity.id,
                'title': title,
                'text': f"{text} Synonyms: {synonyms}".strip(),
                'type': entity.category,
            }
            self.docs.append(payload)
            self._name_to_doc[title.lower()] = payload

    def _score(self, query: str, text: str) -> int:
        q_tokens = set([t.lower() for t in query.split() if t.isalpha() or t.replace('-', '').isalpha()])
        t_tokens = set([t.lower() for t in text.split() if t.isalpha() or t.replace('-', '').isalpha()])
        return len(q_tokens & t_tokens)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        scores = [(self._score(query, d['title'] + ' ' + d['text']), d) for d in self.docs]
        scores.sort(key=lambda x: x[0], reverse=True)
        return [d for s, d in scores[:top_k] if s > 0]

    def retrieve_for_symptoms(self, symptom_indices: List[int], top_k: int = 3) -> List[Dict]:
        # Fallback: return any symptom docs if indices unknown
        candidates = [d for d in self.docs if d['type'] == 'symptom']
        return candidates[:top_k]

    def retrieve_for_diseases_by_name(self, disease_names: List[str], top_k: int = 3) -> List[Dict]:
        results: List[Dict] = []
        for name in disease_names:
            doc = self._name_to_doc.get(name.lower())
            if doc:
                results.append(doc)
        if len(results) < top_k:
            extra = [d for d in self.docs if d['type'] == 'disease' and d not in results]
            results.extend(extra[: max(0, top_k - len(results))])
        return results[:top_k]


