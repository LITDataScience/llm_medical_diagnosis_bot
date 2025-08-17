"""
Federated Learning Client for Medical Diagnosis Bot
This client trains the diagnosis model locally on hospital/clinic data
and communicates with the federated server while preserving data privacy.
"""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
import pickle
import os
import copy
import json
from collections import OrderedDict

from agents.agent import AgentDQN
from dialog_system.dialog_manager import DialogManager
from usersim.usersim_rule import RuleSimulator
from utils.utils import *
import dialog_config


class FederatedDiagnosisClient(fl.client.NumPyClient):
    """
    Federated learning client for medical diagnosis that trains locally
    """
    
    def __init__(
        self,
        client_id: str,
        agent: AgentDQN,
        dialog_manager: DialogManager,
        user_sim: RuleSimulator,
        local_data_path: str,
        privacy_multiplier: float = 1.0,
        clip_norm: float = 1.0,
        num_local_epochs: int = 5,
        local_batch_size: int = 16
    ):
        self.client_id = client_id
        self.agent = agent
        self.dialog_manager = dialog_manager
        self.user_sim = user_sim
        self.local_data_path = local_data_path
        self.privacy_multiplier = privacy_multiplier
        self.clip_norm = clip_norm
        self.num_local_epochs = num_local_epochs
        self.local_batch_size = local_batch_size
        
        # Load local dataset
        self._load_local_data()
        
        # Training metrics
        self.training_history = {
            "loss": [],
            "accuracy": [],
            "success_rate": [],
            "avg_turns": []
        }
        
    def _load_local_data(self):
        """
        Load client's local medical diagnosis data
        """
        # Load goal set for training
        goal_set_path = os.path.join(self.local_data_path, f"{self.client_id}_goals.p")
        if os.path.exists(goal_set_path):
            self.goal_set = pickle.load(open(goal_set_path, 'rb'))
        else:
            # Use default goal set
            default_path = os.path.join(self.local_data_path, "train_goal_dict.p")
            self.goal_set = pickle.load(open(default_path, 'rb'))
            
    def get_parameters(self, config: Dict[str, any]) -> List[np.ndarray]:
        """
        Get current model parameters
        """
        return self._get_model_parameters()
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from server
        """
        self._set_model_parameters(parameters)
    
    def fit(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, any]
    ) -> Tuple[List[np.ndarray], int, Dict[str, float]]:
        """
        Train model locally on client data with privacy preservation
        """
        # Set global model parameters
        self.set_parameters(parameters)
        
        # Local training configuration
        num_rounds = config.get("num_rounds", self.num_local_epochs)
        
        # Training metrics
        total_loss = 0
        success_count = 0
        total_turns = 0
        num_episodes = 0
        
        for epoch in range(num_rounds):
            # Sample batch of local data
            batch_goals = self._sample_batch(self.local_batch_size)
            
            for goal in batch_goals:
                # Reset dialog
                self.user_sim.init_dialog(goal)
                self.agent.initialize_episode()
                self.dialog_manager.initialize_episode()
                
                episode_over = False
                dialog_turn = 0
                
                while not episode_over and dialog_turn < dialog_config.max_turn:
                    # User simulation
                    user_action = self.user_sim.next_response()
                    
                    # Agent response
                    agent_action = self.agent.next(
                        user_action, 
                        turn=dialog_turn,
                        greedy=False
                    )
                    
                    # Dialog management
                    episode_over, reward = self.dialog_manager.next_turn(
                        user_action,
                        agent_action
                    )
                    
                    # Update agent
                    if self.agent.warm_start == 0:
                        self.agent.update(reward, episode_over)
                    
                    dialog_turn += 1
                
                # Collect metrics
                if reward > 0:
                    success_count += 1
                total_turns += dialog_turn
                num_episodes += 1
                
                # Get loss from agent
                if hasattr(self.agent, 'get_recent_loss'):
                    total_loss += self.agent.get_recent_loss()
        
        # Apply gradient clipping for privacy
        self._apply_gradient_clipping()
        
        # Compute metrics
        metrics = {
            "loss": total_loss / max(num_episodes, 1),
            "accuracy": success_count / max(num_episodes, 1),
            "success_rate": success_count / max(num_episodes, 1),
            "avg_turns": total_turns / max(num_episodes, 1),
            "num_episodes": num_episodes,
            "client_id": self.client_id
        }
        
        # Update training history
        for key in ["loss", "accuracy", "success_rate", "avg_turns"]:
            self.training_history[key].append(metrics[key])
        
        # Get updated parameters
        updated_parameters = self.get_parameters(config)
        
        return updated_parameters, num_episodes, metrics
    
    def evaluate(
        self, 
        parameters: List[np.ndarray], 
        config: Dict[str, any]
    ) -> Tuple[float, int, Dict[str, float]]:
        """
        Evaluate model on local test data
        """
        self.set_parameters(parameters)
        
        # Use test goal set for evaluation
        test_goals = self._get_test_goals()
        
        success_count = 0
        total_turns = 0
        num_episodes = len(test_goals)
        
        for goal in test_goals:
            # Reset dialog
            self.user_sim.init_dialog(goal)
            self.agent.initialize_episode()
            self.dialog_manager.initialize_episode()
            
            episode_over = False
            dialog_turn = 0
            
            while not episode_over and dialog_turn < dialog_config.max_turn:
                # User simulation
                user_action = self.user_sim.next_response()
                
                # Agent response (greedy for evaluation)
                agent_action = self.agent.next(
                    user_action,
                    turn=dialog_turn,
                    greedy=True
                )
                
                # Dialog management
                episode_over, reward = self.dialog_manager.next_turn(
                    user_action,
                    agent_action
                )
                
                dialog_turn += 1
            
            if reward > 0:
                success_count += 1
            total_turns += dialog_turn
        
        # Compute evaluation metrics
        accuracy = success_count / max(num_episodes, 1)
        avg_turns = total_turns / max(num_episodes, 1)
        
        metrics = {
            "accuracy": accuracy,
            "success_rate": accuracy,
            "avg_turns": avg_turns,
            "num_episodes": num_episodes,
            "client_id": self.client_id
        }
        
        # Loss is negative accuracy for federated averaging
        loss = 1.0 - accuracy
        
        return loss, num_episodes, metrics
    
    def _get_model_parameters(self) -> List[np.ndarray]:
        """
        Extract model parameters as numpy arrays
        """
        if hasattr(self.agent, 'dqn'):
            # Get DQN parameters
            state_dict = self.agent.dqn.state_dict()
            parameters = [param.cpu().numpy() for param in state_dict.values()]
            
            # Also get target network if exists
            if hasattr(self.agent, 'target_dqn'):
                target_state_dict = self.agent.target_dqn.state_dict()
                target_params = [param.cpu().numpy() for param in target_state_dict.values()]
                parameters.extend(target_params)
                
            return parameters
        else:
            raise ValueError("Agent does not have DQN model")
    
    def _set_model_parameters(self, parameters: List[np.ndarray]) -> None:
        """
        Set model parameters from numpy arrays
        """
        if hasattr(self.agent, 'dqn'):
            # Set DQN parameters
            state_dict = self.agent.dqn.state_dict()
            param_idx = 0
            
            for key in state_dict.keys():
                state_dict[key] = torch.from_numpy(parameters[param_idx]).to(dialog_config.device)
                param_idx += 1
            
            self.agent.dqn.load_state_dict(state_dict)
            
            # Set target network if exists
            if hasattr(self.agent, 'target_dqn') and param_idx < len(parameters):
                target_state_dict = self.agent.target_dqn.state_dict()
                for key in target_state_dict.keys():
                    target_state_dict[key] = torch.from_numpy(parameters[param_idx]).to(dialog_config.device)
                    param_idx += 1
                self.agent.target_dqn.load_state_dict(target_state_dict)
    
    def _sample_batch(self, batch_size: int) -> List[dict]:
        """
        Sample a batch of goals from local data
        """
        indices = np.random.choice(len(self.goal_set), size=batch_size, replace=True)
        return [self.goal_set[i] for i in indices]
    
    def _get_test_goals(self) -> List[dict]:
        """
        Get test goals for evaluation
        """
        test_path = os.path.join(self.local_data_path, f"{self.client_id}_test_goals.p")
        if os.path.exists(test_path):
            return pickle.load(open(test_path, 'rb'))
        else:
            # Use default test set
            default_test_path = os.path.join(self.local_data_path, "test_goal_dict.p")
            if os.path.exists(default_test_path):
                return pickle.load(open(default_test_path, 'rb'))
            else:
                # Use a subset of training data for testing
                return self.goal_set[:50]
    
    def _apply_gradient_clipping(self):
        """
        Apply gradient clipping for differential privacy
        """
        if hasattr(self.agent, 'dqn') and self.clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.agent.dqn.parameters(), 
                self.clip_norm
            )
    
    def save_local_checkpoint(self, round_num: int):
        """
        Save local model checkpoint
        """
        checkpoint_dir = os.path.join("checkpoints", "federated", self.client_id)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"local_model_round_{round_num}.pth"
        )
        
        checkpoint = {
            "round": round_num,
            "model_state": self.agent.dqn.state_dict() if hasattr(self.agent, 'dqn') else None,
            "training_history": self.training_history,
            "client_id": self.client_id
        }
        
        torch.save(checkpoint, checkpoint_path)


def create_federated_client(
    client_id: str,
    data_folder: str = "dataset",
    dqn_hidden_size: int = 128,
    lr: float = 0.01,
    epsilon: float = 0.1,
    **kwargs
) -> FederatedDiagnosisClient:
    """
    Create a federated learning client with initialized components
    """
    # Load data components
    slot_set = pickle.load(open(os.path.join(data_folder, 'slot_set.txt'), 'rb'))
    sym_dict = pickle.load(open(os.path.join(data_folder, 'symptoms.txt'), 'rb'))
    dise_dict = pickle.load(open(os.path.join(data_folder, 'diseases.txt'), 'rb'))
    req_dise_sym_dict = pickle.load(open(os.path.join(data_folder, 'req_dise_sym_dict.p'), 'rb'))
    dise_sym_num_dict = pickle.load(open(os.path.join(data_folder, 'dise_sym_num_dict.p'), 'rb'))
    
    # Load probability matrices
    action_mat_path = os.path.join(data_folder, 'action_mat.txt')
    with open(action_mat_path, 'r') as f:
        tran_mat = json.load(f)
    tran_mat = np.array(tran_mat)
    
    # Initialize components
    params = {
        'dqn_hidden_size': dqn_hidden_size,
        'lr': lr,
        'epsilon': epsilon,
        'experience_replay_size': 10000,
        'batch_size': 16,
        'gamma': 0.9,
        'target_net_update_freq': 1,
        'warm_start': 0,
        'max_turn': dialog_config.max_turn,
        'fix_buffer': 0,
        'priority_replay': False,
        **kwargs
    }
    
    # Create agent
    agent = AgentDQN(
        sym_dict=sym_dict,
        dise_dict=dise_dict,
        req_dise_sym_dict=req_dise_sym_dict,
        dise_sym_num_dict=dise_sym_num_dict,
        tran_mat=tran_mat,
        act_set=dialog_config.feasible_actions,
        slot_set=slot_set,
        params=params
    )
    
    # Create user simulator
    user_sim = RuleSimulator(
        sym_dict=sym_dict,
        dise_dict=dise_dict,
        req_dise_sym_dict=req_dise_sym_dict,
        dise_sym_num_dict=dise_sym_num_dict,
        tran_mat=tran_mat,
        slot_set=slot_set
    )
    
    # Create dialog manager
    dialog_manager = DialogManager(
        agent=agent,
        user_sim=user_sim,
        slot_set=slot_set,
        act_set=dialog_config.feasible_actions
    )
    
    # Create federated client
    client = FederatedDiagnosisClient(
        client_id=client_id,
        agent=agent,
        dialog_manager=dialog_manager,
        user_sim=user_sim,
        local_data_path=data_folder,
        **kwargs
    )
    
    return client


def start_client(
    server_address: str = "localhost:8080",
    client_id: str = None,
    **kwargs
):
    """
    Start a federated learning client
    """
    if client_id is None:
        import uuid
        client_id = f"client_{uuid.uuid4().hex[:8]}"
    
    # Create client
    client = create_federated_client(client_id=client_id, **kwargs)
    
    # Start Flower client
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Client for Medical Diagnosis")
    parser.add_argument("--server", type=str, default="localhost:8080", help="Server address")
    parser.add_argument("--client-id", type=str, default=None, help="Client ID")
    parser.add_argument("--data-folder", type=str, default="dataset", help="Path to data folder")
    parser.add_argument("--privacy-multiplier", type=float, default=1.0, help="Privacy noise multiplier")
    parser.add_argument("--clip-norm", type=float, default=1.0, help="Gradient clipping norm")
    parser.add_argument("--local-epochs", type=int, default=5, help="Number of local epochs")
    
    args = parser.parse_args()
    
    start_client(
        server_address=args.server,
        client_id=args.client_id,
        data_folder=args.data_folder,
        privacy_multiplier=args.privacy_multiplier,
        clip_norm=args.clip_norm,
        num_local_epochs=args.local_epochs
    )