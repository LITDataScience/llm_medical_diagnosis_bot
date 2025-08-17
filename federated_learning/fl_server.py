"""
Federated Learning Server for Medical Diagnosis Bot
This server coordinates the training process across multiple healthcare institutions
while maintaining data privacy through federated learning.
"""

import flwr as fl
import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Optional, Dict
import pickle
import os
from datetime import datetime
import json

from agents.agent import AgentDQN
from qlearning.dqn_prior import KR_DQN
import dialog_config


class FederatedDiagnosisStrategy(fl.server.strategy.FedAvg):
    """
    Custom federated averaging strategy for medical diagnosis with enhanced privacy features
    """
    
    def __init__(
        self,
        fraction_fit: float = 0.5,
        fraction_evaluate: float = 0.5,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        model_path: str = "./checkpoints/federated/",
        privacy_budget: float = 10.0,
        differential_privacy: bool = True,
        **kwargs
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            **kwargs
        )
        self.model_path = model_path
        self.privacy_budget = privacy_budget
        self.differential_privacy = differential_privacy
        self.global_metrics = {
            "accuracy": [],
            "success_rate": [],
            "avg_turns": [],
            "privacy_spent": []
        }
        
        # Create model directory if not exists
        os.makedirs(model_path, exist_ok=True)
        
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ) -> Tuple[Optional[fl.common.Parameters], Dict[str, float]]:
        """
        Aggregate model weights from multiple clients with differential privacy
        """
        if not results:
            return None, {}
            
        # Extract weights and metrics
        weights_results = []
        metrics_list = []
        
        for client, fit_res in results:
            weights = fl.common.parameters_to_ndarrays(fit_res.parameters)
            num_examples = fit_res.num_examples
            weights_results.append((weights, num_examples))
            
            if fit_res.metrics:
                metrics_list.append(fit_res.metrics)
        
        # Apply differential privacy if enabled
        if self.differential_privacy:
            weights_results = self._add_differential_privacy(weights_results, server_round)
        
        # Perform weighted averaging
        aggregated_weights = self._weighted_average(weights_results)
        
        # Save checkpoint
        self._save_checkpoint(aggregated_weights, server_round, metrics_list)
        
        # Aggregate metrics
        aggregated_metrics = self._aggregate_metrics(metrics_list, server_round)
        
        return fl.common.ndarrays_to_parameters(aggregated_weights), aggregated_metrics
    
    def _add_differential_privacy(
        self, 
        weights_results: List[Tuple[List[np.ndarray], int]], 
        server_round: int
    ) -> List[Tuple[List[np.ndarray], int]]:
        """
        Add Gaussian noise for differential privacy
        """
        epsilon = self.privacy_budget / server_round  # Adaptive privacy budget
        sensitivity = 1.0  # L2 sensitivity bound
        
        noisy_weights_results = []
        for weights, num_examples in weights_results:
            noisy_weights = []
            for w in weights:
                # Add Gaussian noise scaled by sensitivity and epsilon
                noise_scale = sensitivity / epsilon
                noise = np.random.normal(0, noise_scale, w.shape)
                noisy_w = w + noise
                noisy_weights.append(noisy_w)
            noisy_weights_results.append((noisy_weights, num_examples))
            
        return noisy_weights_results
    
    def _weighted_average(
        self, 
        weights_results: List[Tuple[List[np.ndarray], int]]
    ) -> List[np.ndarray]:
        """
        Compute weighted average of model parameters
        """
        num_examples_total = sum([num_examples for _, num_examples in weights_results])
        weighted_weights = []
        
        for i in range(len(weights_results[0][0])):
            weighted_sum = np.zeros_like(weights_results[0][0][i])
            for weights, num_examples in weights_results:
                weighted_sum += weights[i] * num_examples / num_examples_total
            weighted_weights.append(weighted_sum)
            
        return weighted_weights
    
    def _aggregate_metrics(
        self, 
        metrics_list: List[Dict[str, float]], 
        server_round: int
    ) -> Dict[str, float]:
        """
        Aggregate evaluation metrics from all clients
        """
        if not metrics_list:
            return {}
            
        aggregated = {}
        for key in metrics_list[0].keys():
            values = [m[key] for m in metrics_list if key in m]
            aggregated[f"avg_{key}"] = np.mean(values)
            aggregated[f"std_{key}"] = np.std(values)
            
        # Update global metrics history
        for key in ["accuracy", "success_rate", "avg_turns"]:
            if f"avg_{key}" in aggregated:
                self.global_metrics[key].append(aggregated[f"avg_{key}"])
                
        aggregated["server_round"] = server_round
        aggregated["num_clients"] = len(metrics_list)
        
        return aggregated
    
    def _save_checkpoint(
        self, 
        weights: List[np.ndarray], 
        server_round: int,
        metrics: List[Dict[str, float]]
    ):
        """
        Save model checkpoint and training history
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model weights
        checkpoint_path = os.path.join(
            self.model_path, 
            f"federated_round_{server_round}_{timestamp}.pkl"
        )
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(weights, f)
            
        # Save metrics history
        metrics_path = os.path.join(
            self.model_path,
            f"metrics_history.json"
        )
        with open(metrics_path, 'w') as f:
            json.dump({
                "round": server_round,
                "timestamp": timestamp,
                "global_metrics": self.global_metrics,
                "client_metrics": metrics
            }, f, indent=2)


class SecureAggregator:
    """
    Implements secure multi-party computation for enhanced privacy
    """
    
    def __init__(self, num_clients: int, threshold: int = None):
        self.num_clients = num_clients
        self.threshold = threshold or (num_clients // 2 + 1)
        
    def generate_masks(self, shape: Tuple[int, ...]) -> List[np.ndarray]:
        """
        Generate random masks for secure aggregation
        """
        masks = []
        base_mask = np.random.randn(*shape)
        
        for i in range(self.num_clients - 1):
            mask = np.random.randn(*shape)
            masks.append(mask)
            base_mask -= mask
            
        masks.append(base_mask)
        return masks
    
    def aggregate_with_masks(
        self, 
        masked_updates: List[np.ndarray],
        dropout_clients: List[int] = None
    ) -> np.ndarray:
        """
        Securely aggregate masked updates
        """
        if dropout_clients:
            # Handle client dropouts
            active_updates = [
                update for i, update in enumerate(masked_updates) 
                if i not in dropout_clients
            ]
        else:
            active_updates = masked_updates
            
        # Sum all masked updates (masks cancel out)
        return np.sum(active_updates, axis=0)


def create_federated_server(
    num_rounds: int = 100,
    fraction_fit: float = 0.5,
    min_clients: int = 2,
    model_path: str = "./checkpoints/federated/",
    differential_privacy: bool = True,
    privacy_budget: float = 10.0
) -> fl.server.Server:
    """
    Create and configure the federated learning server
    """
    strategy = FederatedDiagnosisStrategy(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=min_clients,
        min_evaluate_clients=min_clients,
        min_available_clients=min_clients,
        model_path=model_path,
        differential_privacy=differential_privacy,
        privacy_budget=privacy_budget,
        evaluate_metrics_aggregation_fn=weighted_average_metrics,
    )
    
    # Configure the server
    config = fl.server.ServerConfig(num_rounds=num_rounds)
    
    return fl.server.Server(
        client_manager=fl.server.SimpleClientManager(),
        strategy=strategy,
        config=config,
    )


def weighted_average_metrics(metrics: List[Tuple[int, Dict[str, float]]]) -> Dict[str, float]:
    """
    Weighted average of evaluation metrics
    """
    total_examples = sum([num_examples for num_examples, _ in metrics])
    
    aggregated_metrics = {}
    for num_examples, client_metrics in metrics:
        for key, value in client_metrics.items():
            if key not in aggregated_metrics:
                aggregated_metrics[key] = 0.0
            aggregated_metrics[key] += (num_examples / total_examples) * value
            
    return aggregated_metrics


def start_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds: int = 100,
    **kwargs
):
    """
    Start the federated learning server
    """
    server = create_federated_server(num_rounds=num_rounds, **kwargs)
    
    fl.server.start_server(
        server_address=server_address,
        server=server,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Federated Learning Server for Medical Diagnosis")
    parser.add_argument("--rounds", type=int, default=100, help="Number of federated rounds")
    parser.add_argument("--min-clients", type=int, default=2, help="Minimum number of clients")
    parser.add_argument("--fraction-fit", type=float, default=0.5, help="Fraction of clients for training")
    parser.add_argument("--address", type=str, default="0.0.0.0:8080", help="Server address")
    parser.add_argument("--privacy-budget", type=float, default=10.0, help="Differential privacy budget")
    parser.add_argument("--no-dp", action="store_true", help="Disable differential privacy")
    
    args = parser.parse_args()
    
    start_server(
        server_address=args.address,
        num_rounds=args.rounds,
        min_clients=args.min_clients,
        fraction_fit=args.fraction_fit,
        differential_privacy=not args.no_dp,
        privacy_budget=args.privacy_budget
    )