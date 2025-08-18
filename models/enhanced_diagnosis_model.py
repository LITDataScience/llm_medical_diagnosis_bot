"""
Enhanced Medical Diagnosis Model
This model uses state-of-the-art transformer architecture combined with
medical knowledge graphs to diagnose a wide range of diseases based on symptoms.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
try:
    from transformers import AutoModel, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not installed. Some features will be limited.")
import json
import pickle
from dataclasses import dataclass
from collections import defaultdict
import os


@dataclass
class MedicalEntity:
    """Represents a medical entity (disease, symptom, etc.)"""
    id: str
    name: str
    category: str
    description: str = ""
    icd_code: Optional[str] = None
    synonyms: List[str] = None
    
    def __post_init__(self):
        if self.synonyms is None:
            self.synonyms = []


class MedicalKnowledgeGraph:
    """
    Medical knowledge graph containing relationships between
    diseases, symptoms, risk factors, and treatments
    """
    
    def __init__(self):
        self.entities = {}
        self.relations = defaultdict(lambda: defaultdict(list))
        self.symptom_disease_matrix = None
        self.disease_embeddings = None
        self.symptom_embeddings = None
        
    def add_entity(self, entity: MedicalEntity):
        """Add a medical entity to the knowledge graph"""
        self.entities[entity.id] = entity
        
    def add_relation(self, source_id: str, relation_type: str, target_id: str, weight: float = 1.0):
        """Add a relation between two entities"""
        self.relations[source_id][relation_type].append((target_id, weight))
        
    def build_symptom_disease_matrix(self):
        """Build a matrix of symptom-disease associations"""
        symptoms = [e for e in self.entities.values() if e.category == "symptom"]
        diseases = [e for e in self.entities.values() if e.category == "disease"]
        
        matrix = np.zeros((len(symptoms), len(diseases)))
        symptom_idx = {s.id: i for i, s in enumerate(symptoms)}
        disease_idx = {d.id: i for i, d in enumerate(diseases)}
        
        for symptom in symptoms:
            if "associated_with" in self.relations[symptom.id]:
                for disease_id, weight in self.relations[symptom.id]["associated_with"]:
                    if disease_id in disease_idx:
                        matrix[symptom_idx[symptom.id], disease_idx[disease_id]] = weight
                        
        self.symptom_disease_matrix = matrix
        return matrix
    
    def get_disease_probabilities(self, symptom_ids: List[str]) -> Dict[str, float]:
        """Calculate disease probabilities given symptoms"""
        if self.symptom_disease_matrix is None:
            self.build_symptom_disease_matrix()
            
        symptom_vector = np.zeros(len(self.symptom_disease_matrix))
        symptom_idx = {s.id: i for i, s in enumerate([e for e in self.entities.values() if e.category == "symptom"])}
        
        for sid in symptom_ids:
            if sid in symptom_idx:
                symptom_vector[symptom_idx[sid]] = 1.0
                
        # Calculate disease scores
        disease_scores = np.dot(symptom_vector, self.symptom_disease_matrix)
        disease_probs = F.softmax(torch.tensor(disease_scores), dim=0).numpy()
        
        diseases = [e for e in self.entities.values() if e.category == "disease"]
        return {d.id: float(disease_probs[i]) for i, d in enumerate(diseases)}


class TransformerDiagnosisEncoder(nn.Module):
    """
    Transformer-based encoder for medical text and symptoms
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        hidden_size: int = 768,
        num_diseases: int = 1000,
        num_symptoms: int = 500,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Load pre-trained biomedical language model
        if TRANSFORMERS_AVAILABLE:
            self.bert = AutoModel.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            self.bert = None
            self.tokenizer = None
        
        # Symptom embedding layer
        self.symptom_embeddings = nn.Embedding(num_symptoms, hidden_size)
        
        # Disease prediction layers
        self.disease_classifier = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, num_diseases)
        )
        
        # Confidence estimation
        self.confidence_estimator = nn.Sequential(
            nn.Linear(hidden_size * 2, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for symptom importance
        self.symptom_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout
        )
        
    def forward(
        self,
        text_input: Optional[Dict[str, torch.Tensor]] = None,
        symptom_ids: Optional[torch.Tensor] = None,
        symptom_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Returns:
            disease_logits: (batch_size, num_diseases)
            confidence: (batch_size, 1)
            symptom_attention_weights: (batch_size, num_symptoms)
        """
        batch_size = symptom_ids.size(0) if symptom_ids is not None else text_input['input_ids'].size(0)
        
        # Encode text description if provided
        text_features = None
        if text_input is not None:
            bert_output = self.bert(**text_input)
            text_features = bert_output.pooler_output  # (batch_size, hidden_size)
            
        # Encode symptoms
        symptom_features = None
        attention_weights = None
        if symptom_ids is not None:
            symptom_embeds = self.symptom_embeddings(symptom_ids)  # (batch_size, num_symptoms, hidden_size)
            
            # Apply attention to aggregate symptom information
            attended_symptoms, attention_weights = self.symptom_attention(
                symptom_embeds.transpose(0, 1),
                symptom_embeds.transpose(0, 1),
                symptom_embeds.transpose(0, 1),
                key_padding_mask=~symptom_mask if symptom_mask is not None else None
            )
            symptom_features = attended_symptoms.mean(dim=0)  # (batch_size, hidden_size)
            
        # Combine features
        if text_features is not None and symptom_features is not None:
            combined_features = torch.cat([text_features, symptom_features], dim=-1)
        elif text_features is not None:
            combined_features = torch.cat([text_features, text_features], dim=-1)
        else:
            combined_features = torch.cat([symptom_features, symptom_features], dim=-1)
            
        # Predict diseases
        disease_logits = self.disease_classifier(combined_features)
        
        # Estimate confidence
        confidence = self.confidence_estimator(combined_features)
        
        return disease_logits, confidence, attention_weights


class EnhancedMedicalDiagnosisModel(nn.Module):
    """
    Enhanced medical diagnosis model combining transformers,
    knowledge graphs, and clinical decision rules
    """
    
    def __init__(
        self,
        knowledge_graph: MedicalKnowledgeGraph,
        num_diseases: int = 1000,
        num_symptoms: int = 500,
        hidden_size: int = 768,
        use_differential_diagnosis: bool = True
    ):
        super().__init__()
        
        self.knowledge_graph = knowledge_graph
        self.num_diseases = num_diseases
        self.num_symptoms = num_symptoms
        self.use_differential_diagnosis = use_differential_diagnosis
        
        # Transformer encoder
        self.encoder = TransformerDiagnosisEncoder(
            hidden_size=hidden_size,
            num_diseases=num_diseases,
            num_symptoms=num_symptoms
        )
        
        # Graph neural network for knowledge graph reasoning
        self.graph_encoder = GraphAttentionNetwork(
            num_nodes=len(knowledge_graph.entities),
            input_dim=hidden_size,
            hidden_dim=256,
            output_dim=128,
            num_heads=4
        )
        
        # Differential diagnosis module
        if use_differential_diagnosis:
            self.differential_diagnosis = DifferentialDiagnosisModule(
                num_diseases=num_diseases,
                hidden_size=256
            )
            
        # Clinical rule engine
        self.rule_engine = ClinicalRuleEngine()
        
        # Final disease prediction layer
        self.final_predictor = nn.Sequential(
            nn.Linear(num_diseases * 3 if use_differential_diagnosis else num_diseases * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_diseases)
        )
        
    def forward(
        self,
        text_description: Optional[str] = None,
        symptom_ids: Optional[List[int]] = None,
        patient_history: Optional[Dict] = None,
        apply_rules: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Perform medical diagnosis
        
        Args:
            text_description: Natural language description of symptoms
            symptom_ids: List of symptom IDs
            patient_history: Patient medical history
            apply_rules: Whether to apply clinical decision rules
            
        Returns:
            Dictionary containing:
                - disease_probabilities: Probability distribution over diseases
                - confidence: Model confidence score
                - differential_diagnoses: Top differential diagnoses
                - explanations: Reasoning explanations
        """
        device = next(self.parameters()).device
        
        # Prepare inputs
        text_input = None
        if text_description:
            text_input = self.encoder.tokenizer(
                text_description,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(device)
            
        symptom_tensor = None
        if symptom_ids:
            symptom_tensor = torch.tensor(symptom_ids).unsqueeze(0).to(device)
            
        # Get transformer predictions
        disease_logits, confidence, symptom_attention = self.encoder(
            text_input=text_input,
            symptom_ids=symptom_tensor
        )
        
        # Get knowledge graph predictions
        kg_disease_probs = self._get_kg_predictions(symptom_ids)
        kg_tensor = torch.tensor(list(kg_disease_probs.values())).unsqueeze(0).to(device)
        
        # Combine predictions
        combined_features = torch.cat([
            disease_logits,
            kg_tensor
        ], dim=-1)
        
        # Apply differential diagnosis if enabled
        if self.use_differential_diagnosis and hasattr(self, 'differential_diagnosis'):
            diff_diagnosis = self.differential_diagnosis(disease_logits, patient_history)
            combined_features = torch.cat([combined_features, diff_diagnosis], dim=-1)
            
        # Final prediction
        final_logits = self.final_predictor(combined_features)
        disease_probs = F.softmax(final_logits, dim=-1)
        
        # Apply clinical rules if requested
        if apply_rules:
            disease_probs = self.rule_engine.apply_rules(
                disease_probs,
                symptom_ids,
                patient_history
            )
            
        # Get top diagnoses
        top_k = 5
        top_probs, top_indices = torch.topk(disease_probs, k=top_k, dim=-1)
        
        # Generate explanations
        explanations = self._generate_explanations(
            top_indices,
            symptom_attention,
            kg_disease_probs
        )
        
        return {
            "disease_probabilities": disease_probs,
            "confidence": confidence,
            "top_diseases": top_indices,
            "top_probabilities": top_probs,
            "explanations": explanations,
            "symptom_importance": symptom_attention
        }
        
    def _get_kg_predictions(self, symptom_ids: Optional[List[int]]) -> Dict[str, float]:
        """Get predictions from knowledge graph"""
        if symptom_ids is None:
            return {str(i): 0.0 for i in range(self.num_diseases)}
            
        # Convert symptom IDs to entity IDs
        symptom_entities = []
        for sid in symptom_ids:
            if str(sid) in self.knowledge_graph.entities:
                symptom_entities.append(str(sid))
                
        return self.knowledge_graph.get_disease_probabilities(symptom_entities)
        
    def _generate_explanations(
        self,
        top_diseases: torch.Tensor,
        symptom_attention: Optional[torch.Tensor],
        kg_predictions: Dict[str, float]
    ) -> List[str]:
        """Generate explanations for diagnoses"""
        explanations = []
        
        for disease_idx in top_diseases[0].cpu().numpy():
            disease_id = str(disease_idx)
            if disease_id in self.knowledge_graph.entities:
                disease = self.knowledge_graph.entities[disease_id]
                explanation = f"Diagnosis: {disease.name}\n"
                
                # Add symptom-based reasoning
                if symptom_attention is not None:
                    top_symptoms = torch.topk(symptom_attention[0], k=3).indices
                    explanation += "Key symptoms: "
                    for sym_idx in top_symptoms:
                        if str(sym_idx.item()) in self.knowledge_graph.entities:
                            explanation += f"{self.knowledge_graph.entities[str(sym_idx.item())].name}, "
                    explanation = explanation.rstrip(", ") + "\n"
                    
                # Add knowledge graph confidence
                if disease_id in kg_predictions:
                    explanation += f"Knowledge-based confidence: {kg_predictions[disease_id]:.2f}\n"
                    
                explanations.append(explanation)
                
        return explanations


class GraphAttentionNetwork(nn.Module):
    """Graph Attention Network for medical knowledge graph encoding"""
    
    def __init__(self, num_nodes: int, input_dim: int, hidden_dim: int, output_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_nodes = num_nodes
        self.attention_heads = nn.ModuleList([
            GraphAttentionLayer(input_dim, hidden_dim)
            for _ in range(num_heads)
        ])
        self.output_layer = nn.Linear(hidden_dim * num_heads, output_dim)
        
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Multi-head attention
        head_outputs = []
        for attention in self.attention_heads:
            head_outputs.append(attention(node_features, adjacency_matrix))
        
        # Concatenate heads
        combined = torch.cat(head_outputs, dim=-1)
        
        # Output projection
        return self.output_layer(combined)


class GraphAttentionLayer(nn.Module):
    """Single graph attention layer"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.W = nn.Linear(input_dim, output_dim, bias=False)
        self.a = nn.Parameter(torch.randn(2 * output_dim, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, node_features: torch.Tensor, adjacency_matrix: torch.Tensor) -> torch.Tensor:
        # Linear transformation
        h = self.W(node_features)
        batch_size, num_nodes, feat_dim = h.size()
        
        # Attention mechanism
        h_i = h.unsqueeze(2).expand(batch_size, num_nodes, num_nodes, feat_dim)
        h_j = h.unsqueeze(1).expand(batch_size, num_nodes, num_nodes, feat_dim)
        
        attention_input = torch.cat([h_i, h_j], dim=-1)
        attention_scores = self.leaky_relu(torch.matmul(attention_input, self.a).squeeze(-1))
        
        # Mask based on adjacency matrix
        attention_scores = attention_scores.masked_fill(adjacency_matrix == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # Apply attention
        return torch.matmul(attention_weights, h)


class DifferentialDiagnosisModule(nn.Module):
    """Module for differential diagnosis reasoning"""
    
    def __init__(self, num_diseases: int, hidden_size: int = 256):
        super().__init__()
        self.disease_similarity = nn.Parameter(torch.randn(num_diseases, num_diseases))
        self.differential_encoder = nn.Sequential(
            nn.Linear(num_diseases * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_diseases)
        )
        
    def forward(self, disease_logits: torch.Tensor, patient_history: Optional[Dict] = None) -> torch.Tensor:
        # Compute disease similarities
        disease_probs = F.softmax(disease_logits, dim=-1)
        similarity_scores = torch.matmul(disease_probs, self.disease_similarity)
        
        # Combine with original predictions
        combined = torch.cat([disease_logits, similarity_scores], dim=-1)
        
        # Generate differential diagnosis scores
        return self.differential_encoder(combined)


class ClinicalRuleEngine:
    """Engine for applying clinical decision rules"""
    
    def __init__(self):
        self.rules = []
        self._load_default_rules()
        
    def _load_default_rules(self):
        """Load default clinical rules"""
        # Example rules - in practice, these would be loaded from a medical database
        self.rules.append({
            "name": "Fever + Cough -> Respiratory Infection",
            "symptoms": ["fever", "cough"],
            "boost_diseases": ["upper_respiratory_infection", "pneumonia"],
            "weight": 0.2
        })
        
    def apply_rules(
        self,
        disease_probs: torch.Tensor,
        symptom_ids: List[int],
        patient_history: Optional[Dict] = None
    ) -> torch.Tensor:
        """Apply clinical rules to adjust disease probabilities"""
        adjusted_probs = disease_probs.clone()
        
        for rule in self.rules:
            if self._check_rule_conditions(rule, symptom_ids, patient_history):
                for disease in rule.get("boost_diseases", []):
                    # Boost probability for specific diseases
                    # In practice, map disease names to indices
                    pass
                    
        # Re-normalize probabilities
        return F.softmax(adjusted_probs, dim=-1)
        
    def _check_rule_conditions(
        self,
        rule: Dict,
        symptom_ids: List[int],
        patient_history: Optional[Dict]
    ) -> bool:
        """Check if rule conditions are met"""
        # Simplified implementation
        return True


def create_enhanced_diagnosis_model(
    disease_data_path: str = "dataset/diseases_extended.json",
    symptom_data_path: str = "dataset/symptoms_extended.json",
    pretrained_path: Optional[str] = None
) -> EnhancedMedicalDiagnosisModel:
    """
    Create and initialize the enhanced diagnosis model
    """
    # Load medical knowledge graph
    kg = MedicalKnowledgeGraph()
    
    # Load disease and symptom data
    # In practice, this would load from comprehensive medical databases
    # For now, create sample data
    
    # Add sample diseases
    diseases = [
        MedicalEntity("d001", "COVID-19", "disease", "Coronavirus disease 2019", "U07.1"),
        MedicalEntity("d002", "Influenza", "disease", "Seasonal flu", "J11"),
        MedicalEntity("d003", "Pneumonia", "disease", "Lung infection", "J18"),
        MedicalEntity("d004", "Common Cold", "disease", "Upper respiratory infection", "J00"),
        MedicalEntity("d005", "Asthma", "disease", "Chronic respiratory condition", "J45"),
        # Add more diseases...
    ]
    
    # Add sample symptoms
    symptoms = [
        MedicalEntity("s001", "Fever", "symptom", "Elevated body temperature"),
        MedicalEntity("s002", "Cough", "symptom", "Reflex to clear airways"),
        MedicalEntity("s003", "Shortness of breath", "symptom", "Difficulty breathing"),
        MedicalEntity("s004", "Fatigue", "symptom", "Extreme tiredness"),
        MedicalEntity("s005", "Headache", "symptom", "Pain in head region"),
        # Add more symptoms...
    ]
    
    # Add entities to knowledge graph
    for disease in diseases:
        kg.add_entity(disease)
    for symptom in symptoms:
        kg.add_entity(symptom)
        
    # Add relationships
    kg.add_relation("s001", "associated_with", "d001", 0.9)  # Fever -> COVID-19
    kg.add_relation("s002", "associated_with", "d001", 0.8)  # Cough -> COVID-19
    kg.add_relation("s003", "associated_with", "d001", 0.7)  # Shortness of breath -> COVID-19
    # Add more relationships...
    
    # Create model
    model = EnhancedMedicalDiagnosisModel(
        knowledge_graph=kg,
        num_diseases=len(diseases),
        num_symptoms=len(symptoms)
    )
    # Expose ordered name lists aligned to model output indices
    model.disease_names = [d.name for d in diseases]
    model.symptom_names = [s.name for s in symptoms]
    
    # Load pretrained weights if available
    if pretrained_path and os.path.exists(pretrained_path):
        model.load_state_dict(torch.load(pretrained_path))
        
    return model


if __name__ == "__main__":
    # Test the model
    model = create_enhanced_diagnosis_model()
    
    # Example diagnosis
    result = model(
        text_description="I have a high fever, dry cough, and difficulty breathing",
        symptom_ids=[0, 1, 2],  # Fever, Cough, Shortness of breath
        patient_history={"age": 45, "conditions": ["hypertension"]}
    )
    
    print("Top diagnoses:")
    for i, (disease_idx, prob) in enumerate(zip(result["top_diseases"][0], result["top_probabilities"][0])):
        print(f"{i+1}. Disease {disease_idx}: {prob:.2%}")
        
    print("\nExplanations:")
    for exp in result["explanations"]:
        print(exp)