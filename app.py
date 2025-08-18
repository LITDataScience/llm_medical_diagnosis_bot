"""
Medical Diagnosis Bot - Chat Interface
A modern, user-friendly chat interface for medical diagnosis using AI
"""

import streamlit as st
import streamlit.components.v1 as components
# from streamlit_chat import message  # Optional: install if needed
import torch
import numpy as np
import pickle
import os
import json
from datetime import datetime
import uuid
from typing import Dict, List, Optional, Tuple
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Import our models and components
from agents.agent import AgentDQN
from dialog_system.dialog_manager import DialogManager
from models.enhanced_diagnosis_model import create_enhanced_diagnosis_model, EnhancedMedicalDiagnosisModel
from rag.retriever import SimpleMedicalRetriever
from agents.multi_agent_orchestrator import ConversationOrchestrator
import dialog_config
from federated_learning.fl_client import create_federated_client


# Page configuration
st.set_page_config(
    page_title="AI Medical Diagnosis Assistant",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful UI
st.markdown("""
<style>
    /* Main container */
    .main {
        padding: 0;
        max-width: 100%;
    }
    
    /* Chat messages */
    .stChatMessage {
        background-color: #f7f7f7;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    
    /* User message */
    .user-message {
        background-color: #e3f2fd;
        border-radius: 15px;
        padding: 12px 18px;
        margin: 5px 0;
        align-self: flex-end;
        max-width: 70%;
    }
    
    /* Bot message */
    .bot-message {
        background-color: #ffffff;
        border-radius: 15px;
        padding: 12px 18px;
        margin: 5px 0;
        align-self: flex-start;
        max-width: 70%;
        border: 1px solid #e0e0e0;
    }
    
    /* Sidebar styling */
    .sidebar .element-container {
        padding: 5px 0;
    }
    
    /* Diagnosis card */
    .diagnosis-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 15px;
        margin: 10px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Symptom chip */
    .symptom-chip {
        display: inline-block;
        background-color: #4CAF50;
        color: white;
        padding: 5px 10px;
        border-radius: 20px;
        margin: 2px;
        font-size: 14px;
    }
    
    /* Confidence meter */
    .confidence-meter {
        height: 20px;
        background-color: #f0f0f0;
        border-radius: 10px;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        background: linear-gradient(90deg, #4CAF50 0%, #8BC34A 100%);
        transition: width 0.5s ease;
    }
    
    /* Animation */
    @keyframes pulse {
        0% { opacity: 0.6; }
        50% { opacity: 1; }
        100% { opacity: 0.6; }
    }
    
    .thinking {
        animation: pulse 1.5s infinite;
    }
</style>
""", unsafe_allow_html=True)


class MedicalDiagnosisChatBot:
    """
    Main chat bot class that handles the medical diagnosis conversation
    """
    
    def __init__(self):
        self.initialize_models()
        self.symptom_list = self.load_symptoms()
        self.disease_list = self.load_diseases()
        # Build lightweight retriever index from the enhanced model's knowledge graph
        self.retriever = SimpleMedicalRetriever()
        try:
            self.retriever.index_from_model(self.enhanced_model)
        except Exception:
            pass
        # Multi-agent orchestrator (LLM-assisted)
        self.orchestrator = ConversationOrchestrator(self, self.retriever)
        
    def initialize_models(self):
        """Initialize all AI models"""
        # Load enhanced diagnosis model
        self.enhanced_model = create_enhanced_diagnosis_model()
        self.enhanced_model.eval()
        
        # Load dialog components
        self.load_dialog_components()
        
    def load_dialog_components(self):
        """Load dialog system components"""
        data_folder = "dataset"
        
        # Load data files
        try:
            self.slot_set = pickle.load(open(os.path.join(data_folder, 'slot_set.txt'), 'rb'))
        except:
            # If pickle fails, load as text
            with open(os.path.join(data_folder, 'slot_set.txt'), 'r') as f:
                lines = f.readlines()
                self.slot_set = {line.strip(): i for i, line in enumerate(lines)}
                
        # Load symptoms dictionary
        try:
            self.sym_dict = pickle.load(open(os.path.join(data_folder, 'symptoms.txt'), 'rb'))
        except:
            with open(os.path.join(data_folder, 'symptoms.txt'), 'r') as f:
                lines = f.readlines()
                self.sym_dict = {i: line.strip() for i, line in enumerate(lines)}
                
        # Load diseases dictionary  
        try:
            self.dise_dict = pickle.load(open(os.path.join(data_folder, 'diseases.txt'), 'rb'))
        except:
            with open(os.path.join(data_folder, 'diseases.txt'), 'r') as f:
                lines = f.readlines()
                self.dise_dict = {i: line.strip() for i, line in enumerate(lines)}
        
        # Load additional required data files for AgentDQN
        try:
            self.req_dise_sym_dict = pickle.load(open(os.path.join(data_folder, 'req_dise_sym_dict.p'), 'rb'))
        except:
            self.req_dise_sym_dict = {}
            
        try:
            self.dise_sym_num_dict = pickle.load(open(os.path.join(data_folder, 'dise_sym_num_dict.p'), 'rb'))
        except:
            self.dise_sym_num_dict = {}
            
        # Load transition matrix (action_mat.txt)
        try:
            with open(os.path.join(data_folder, 'action_mat.txt'), 'r') as f:
                lines = f.readlines()
                tran_mat = []
                for line in lines:
                    if line.strip():
                        row = [float(x) for x in line.strip().split()]
                        tran_mat.append(row)
                self.tran_mat = np.array(tran_mat)
        except:
            # Create a default transition matrix if file not found
            self.tran_mat = np.zeros((100, 100))  # Default size
            
        # Load symptom-disease probability matrix
        try:
            with open(os.path.join(data_folder, 'sym_dise_pro.txt'), 'r') as f:
                lines = f.readlines()
                sym_dise_pro = []
                for line in lines:
                    if line.strip():
                        row = [float(x) for x in line.strip().split()]
                        sym_dise_pro.append(row)
                self.sym_dise_pro = np.array(sym_dise_pro)
        except:
            self.sym_dise_pro = np.zeros((100, 100))  # Default size
            
        # Load disease-symptom probability matrix
        try:
            with open(os.path.join(data_folder, 'dise_sym_pro.txt'), 'r') as f:
                lines = f.readlines()
                dise_sym_pro = []
                for line in lines:
                    if line.strip():
                        row = [float(x) for x in line.strip().split()]
                        dise_sym_pro.append(row)
                self.dise_sym_pro = np.array(dise_sym_pro)
        except:
            self.dise_sym_pro = np.zeros((100, 100))  # Default size
            
        # Load symptom priority
        try:
            with open(os.path.join(data_folder, 'sym_prio.txt'), 'r') as f:
                line = f.readline().strip()
                self.sym_prio = np.array([float(x) for x in line.split()])
        except:
            self.sym_prio = np.zeros(100)  # Default size
        
        # Initialize agent
        params = {
            'dqn_hidden_size': 128,
            'lr': 0.01,
            'epsilon': 0.0,
            'experience_replay_size': 10000,
            'batch_size': 16,
            'gamma': 0.9,
            'target_net_update_freq': 1,
            'warm_start': 2,
            'max_turn': 22,
            'fix_buffer': 0,
            'priority_replay': False,
            'trained_model_path': None,
            'predict_mode': True
        }
        
        # Create act_set dictionary from feasible_actions
        act_set = {}
        for i, action in enumerate(dialog_config.feasible_actions):
            act_set[action['diaact']] = i
            
        # Create agent
        self.agent = AgentDQN(
            sym_dict=self.sym_dict,
            dise_dict=self.dise_dict,
            req_dise_sym_dict=self.req_dise_sym_dict,
            dise_sym_num_dict=self.dise_sym_num_dict,
            tran_mat=self.tran_mat,
            sym_dise_pro=self.sym_dise_pro,
            dise_sym_pro=self.dise_sym_pro,
            sym_prio=self.sym_prio,
            act_set=act_set,
            slot_set=self.slot_set,
            params=params
        )
        
        # Load pre-trained model if available
        # model_path = "./checkpoints/exp_models/KR-DQN/test_0.739.pth.tar"
        # if os.path.exists(model_path):
        #     checkpoint = torch.load(model_path, map_location=dialog_config.device)
        #     self.agent.model.load_state_dict(checkpoint['state_dict'])
        #     self.agent.target_model.load_state_dict(self.agent.model.state_dict())
        #     self.agent.predict_mode = True
        #     self.agent.warm_start = 2
        
    def load_symptoms(self) -> List[str]:
        """Load symptom list"""
        symptom_file = "dataset/symptoms.txt"
        if os.path.exists(symptom_file):
            with open(symptom_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []
        
    def load_diseases(self) -> List[str]:
        """Load disease list"""
        disease_file = "dataset/diseases.txt"
        if os.path.exists(disease_file):
            with open(disease_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        return []
        
    def process_user_input(self, user_input: str, conversation_state: Dict) -> Dict:
        """Process a user turn through the multi-agent orchestrator.
        The orchestrator decides whether to ask follow-up or to diagnose.
        """
        return self.orchestrator.route_turn(user_input, conversation_state)
        
    def extract_symptoms(self, text: str) -> List[int]:
        """Extract symptom IDs from user text"""
        # Simple keyword matching - in production, use NLP
        extracted = []
        text_lower = text.lower()
        
        for idx, symptom in enumerate(self.symptom_list):
            if symptom.lower() in text_lower:
                extracted.append(idx)
                
        return extracted
        
    def get_diagnosis(self, text_description: str, symptom_ids: List[int], patient_history: Dict) -> Dict:
        """Get diagnosis from enhanced model"""
        with torch.no_grad():
            result = self.enhanced_model(
                text_description=text_description,
                symptom_ids=symptom_ids,
                patient_history=patient_history
            )
            
        return result
        
    def format_diagnosis_response(self, diagnosis_result: Dict) -> str:
        """Format diagnosis results into readable response"""
        confidence = float(diagnosis_result['confidence'][0].item())
        top_diseases = diagnosis_result['top_diseases'][0]
        top_probs = diagnosis_result['top_probabilities'][0]
        
        response = f"Based on your symptoms, here are the most likely diagnoses:\n\n"
        
        # Prefer disease names from the enhanced model for correct alignment
        model_disease_names = getattr(self.enhanced_model, 'disease_names', None)
        for i, (disease_idx, prob) in enumerate(zip(top_diseases[:3], top_probs[:3])):
            idx = disease_idx.item()
            if model_disease_names and idx < len(model_disease_names):
                disease_name = model_disease_names[idx]
            else:
                disease_name = self.disease_list[idx] if idx < len(self.disease_list) else f"Disease {idx}"
            response += f"**{i+1}. {disease_name}** - {float(prob.item())*100:.1f}% probability\n"
            
        response += f"\n**Overall Confidence**: {confidence*100:.1f}%\n\n"
        response += "⚠️ **Important**: This is an AI-assisted diagnosis for reference only. Please consult a healthcare professional for accurate medical advice."
        
        return response
        
    def generate_follow_up_question(self, conversation_state: Dict) -> str:
        """Generate relevant follow-up questions"""
        questions = [
            "Can you describe any other symptoms you're experiencing?",
            "How long have you been experiencing these symptoms?",
            "Do you have any chronic conditions or are you taking any medications?",
            "Have you had any recent travel or exposure to sick individuals?",
            "Is there any specific area of discomfort or pain?",
            "Have you noticed any patterns in when the symptoms occur?"
        ]
        
        # Select question based on conversation context
        turn = conversation_state['turn_count'] % len(questions)
        return questions[turn]


def init_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
        
    if 'conversation_state' not in st.session_state:
        st.session_state.conversation_state = {
            'session_id': str(uuid.uuid4()),
            'mentioned_symptoms': [],
            'turn_count': 0,
            'patient_history': {},
            'diagnosis': None
        }
        
    if 'chatbot' not in st.session_state:
        with st.spinner("🤖 Initializing AI Medical Assistant..."):
            st.session_state.chatbot = MedicalDiagnosisChatBot()


def display_sidebar():
    """Display sidebar with patient information and settings"""
    with st.sidebar:
        st.markdown("## 🏥 AI Medical Assistant")
        st.markdown("---")
        
        # Patient Information
        st.markdown("### 👤 Patient Information")
        age = st.number_input("Age", min_value=0, max_value=120, value=30)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        # Medical History
        st.markdown("### 📋 Medical History")
        conditions = st.multiselect(
            "Existing Conditions",
            ["Diabetes", "Hypertension", "Asthma", "Heart Disease", "None"]
        )
        
        allergies = st.text_input("Allergies (comma-separated)")
        
        # Update patient history
        st.session_state.conversation_state['patient_history'] = {
            'age': age,
            'gender': gender,
            'conditions': conditions,
            'allergies': allergies.split(',') if allergies else []
        }
        
        # Settings
        st.markdown("### ⚙️ Settings")
        enable_federated = st.checkbox("Enable Federated Learning", value=False)
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        
        # Session Info
        st.markdown("### 📊 Session Info")
        st.info(f"Session ID: {st.session_state.conversation_state['session_id'][:8]}...")
        st.metric("Symptoms Mentioned", len(st.session_state.conversation_state['mentioned_symptoms']))
        st.metric("Conversation Turns", st.session_state.conversation_state['turn_count'])
        
        # Clear conversation
        if st.button("🔄 Start New Consultation", type="secondary"):
            st.session_state.messages = []
            st.session_state.conversation_state = {
                'session_id': str(uuid.uuid4()),
                'mentioned_symptoms': [],
                'turn_count': 0,
                'patient_history': {},
                'diagnosis': None
            }
            st.rerun()


def display_diagnosis_visualization():
    """Display diagnosis visualization if available"""
    if st.session_state.conversation_state.get('diagnosis'):
        diagnosis = st.session_state.conversation_state['diagnosis']
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Disease probability chart
            disease_names = []
            probabilities = []
            
            model_disease_names = getattr(st.session_state.chatbot.enhanced_model, 'disease_names', None)
            for i, (disease_idx, prob) in enumerate(zip(
                diagnosis['top_diseases'][0][:5], 
                diagnosis['top_probabilities'][0][:5]
            )):
                idx = disease_idx.item()
                if model_disease_names and idx < len(model_disease_names):
                    disease_name = model_disease_names[idx]
                else:
                    disease_name = st.session_state.chatbot.disease_list[idx] \
                        if idx < len(st.session_state.chatbot.disease_list) \
                        else f"Disease {idx}"
                disease_names.append(disease_name)
                probabilities.append(float(prob.item()) * 100)
                
            fig = go.Figure(data=[
                go.Bar(
                    x=probabilities,
                    y=disease_names,
                    orientation='h',
                    marker_color=['#4CAF50', '#8BC34A', '#CDDC39', '#FFC107', '#FF9800']
                )
            ])
            
            fig.update_layout(
                title="Top Diagnosis Probabilities",
                xaxis_title="Probability (%)",
                yaxis_title="Disease",
                height=400,
                template="plotly_white"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Confidence gauge
            confidence = float(diagnosis['confidence'][0].item()) * 100
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=confidence,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Model Confidence"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#4CAF50"},
                    'steps': [
                        {'range': [0, 50], 'color': "#ffebee"},
                        {'range': [50, 80], 'color': "#fff3e0"},
                        {'range': [80, 100], 'color': "#e8f5e9"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
        # Symptom importance
        if diagnosis.get('symptom_importance') is not None:
            st.markdown("### 🔍 Symptom Importance Analysis")
            symptom_importance = diagnosis['symptom_importance']
            # Visualization code here


def main():
    """Main application function"""
    init_session_state()
    display_sidebar()
    
    # Main chat interface
    st.markdown("# 🏥 AI Medical Diagnosis Assistant")
    st.markdown("Welcome! I'm your AI-powered medical assistant. Describe your symptoms and I'll help identify potential conditions.")
    
    # Display diagnosis visualization if available
    display_diagnosis_visualization()
    
    # Chat container
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Display extracted symptoms if available
                if message.get("symptoms"):
                    st.markdown("**Detected symptoms:**")
                    for symptom_id in message["symptoms"]:
                        symptom_name = st.session_state.chatbot.symptom_list[symptom_id] \
                            if symptom_id < len(st.session_state.chatbot.symptom_list) \
                            else f"Symptom {symptom_id}"
                        st.markdown(f'<span class="symptom-chip">{symptom_name}</span>', unsafe_allow_html=True)
    
    # User input
    user_input = st.chat_input("Describe your symptoms...")
    
    if user_input:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.chat_message("user"):
            st.markdown(user_input)
            
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Analyzing your symptoms..."):
                result = st.session_state.chatbot.process_user_input(
                    user_input,
                    st.session_state.conversation_state
                )
                
            st.markdown(result['response'])
            
            # Display extracted symptoms
            if result['extracted_symptoms']:
                st.markdown("**Detected symptoms:**")
                for symptom_id in result['extracted_symptoms']:
                    symptom_name = st.session_state.chatbot.symptom_list[symptom_id] \
                        if symptom_id < len(st.session_state.chatbot.symptom_list) \
                        else f"Symptom {symptom_id}"
                    st.markdown(f'<span class="symptom-chip">{symptom_name}</span>', unsafe_allow_html=True)
                    
        # Add assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": result['response'],
            "symptoms": result['extracted_symptoms']
        })
        
        # Update conversation state
        st.session_state.conversation_state = result['state']
        
        # Rerun to update the UI
        st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666;'>
            <p>⚠️ This AI assistant is for educational purposes only. Always consult healthcare professionals for medical advice.</p>
            <p>Powered by Advanced AI with Federated Learning | Privacy-Preserving Medical Diagnosis</p>
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()