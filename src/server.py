"""
This module sets up and runs a federated learning server using the Flower framework.

Author: Anh Nguyen, aln4739@rit.edu, Ananya Misra, am4063@g.rit.edu;
"""

import csv
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from flwr.common import Context, FitIns, Metrics, ndarrays_to_parameters, Parameters, parameters_to_ndarrays, Scalar
from flwr.common.typing import FitRes
from flwr.server import ClientManager, ServerApp, ServerAppComponents, ServerConfig
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from .main import get_device
from .model import get_model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """Calculate weighted average of accuracy and loss metrics."""
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples)
    }


class MetricLogger:
    """Log and save metrics during federated learning rounds."""
    def __init__(self, is_poisoned, num_rounds, config_string):
        self.folder = "results/attack" if is_poisoned else "results/no_attack"
        self.filename = f"{self.folder}/global_metrics({config_string}).csv"
        os.makedirs(self.folder, exist_ok=True)
        self.metrics = []
        self.num_rounds = num_rounds

    def log_metrics(self, round_number, metrics):
        self.metrics.append({"round": round_number, **metrics})
        if round_number == self.num_rounds:
            self.save_metrics()
            print(f"Metrics saved successfully to {self.filename}")

    def save_metrics(self):
        with open(self.filename, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "loss", "accuracy"])
            writer.writeheader()
            writer.writerows(self.metrics)


class FedAvgTrust(FedAvg):
    """FedAvg with trust and reputation mechanism."""

    def __init__(
            self,
            fraction_fit: float,
            fraction_evaluate: float,
            min_available_clients: int,
            initial_parameters: Parameters,
            fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn,
            trust_threshold: float = 0.5,
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_available_clients=min_available_clients,
            initial_parameters=initial_parameters,
            fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn=evaluate_metrics_aggregation_fn,
        )
        self.trust_threshold = trust_threshold
        self.client_reputations = {}
        self.client_distances = {}
        self.alpha = 0.5
        self.t = 10.0

    @staticmethod
    def _calculate_l2_distance(client_params, mean_params) -> float:
        total_distance = 0
        for client_layer, mean_layer in zip(client_params, mean_params):
            diff = client_layer - mean_layer
            total_distance += float(np.sum(diff * diff))
        return np.sqrt(total_distance)

    def calc_reputation(self, client_id: ClientProxy, d: float, t: int) -> float:
        """Calculate reputation using provided formula."""
        R_prev = self.client_reputations.get(client_id, 1.0)
        if d < self.alpha:
            R = (R_prev + d) - (R_prev / t)
        else:
            R = (R_prev + d) * np.exp(-(1 - d) * (R_prev / t))
        return max(0.0, min(1.0, R))

    def calc_trust(self, R: float, d: float) -> float:
        """Calculate trust score from reputation and distance."""
        trust = np.sqrt((R * R) + (d * d)) - np.sqrt((1 - R) * (1 - R) + (1 - d) * (1 - d))
        if trust >= self.trust_threshold:
            return 1.0
        return 0.0

    def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[ClientProxy, FitRes]],
            failures: List[BaseException],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        all_params = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        mean_params = []
        for i in range(len(all_params[0])):
            layer_params = [p[i] for p in all_params]
            mean_params.append(np.mean(layer_params, axis=0))

        max_distance = 1e-8
        for client, fit_res in results:
            client_params = parameters_to_ndarrays(fit_res.parameters)
            distance = self._calculate_l2_distance(client_params, mean_params)
            max_distance = max(max_distance, distance)

        trusted_results = []
        for client, fit_res in results:
            normalized_distance = self._calculate_l2_distance(
                parameters_to_ndarrays(fit_res.parameters), 
                mean_params
            ) / max_distance
            
            reputation = (1.0 - normalized_distance if server_round == 1 
                        else self.calc_reputation(client, normalized_distance, server_round))
            
            self.client_reputations[client] = reputation
            self.client_distances[client] = normalized_distance

            if self.calc_trust(reputation, normalized_distance) > 0:
                trusted_results.append((client, fit_res))

        if not trusted_results and results:
            best_client = max(results, key=lambda x: self.client_reputations[x[0]])
            trusted_results = [best_client]

        # Print reputations for each round
        print(f"\nRound {server_round} Reputations:")
        for client in self.client_reputations:
            print(f"Client {client.cid}: {self.client_reputations[client]:.4f}")

        return super().aggregate_fit(server_round, trusted_results, failures)

    def configure_fit(
            self,
            server_round: int,
            parameters: Parameters,
            client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure fit round and remove untrusted clients."""
        for client, reputation in self.client_reputations.items():
            if self.calc_trust(reputation, self.client_distances[client]) < self.trust_threshold:
                client_manager.unregister(client)
        return super().configure_fit(server_round, parameters, client_manager)


def server_fn(context: Context):
    """Initialize and configure the federated learning server."""
    num_rounds = context.run_config['num-server-rounds']
    fraction_fit = context.run_config['fraction-fit']
    fraction_evaluate = context.run_config['fraction-evaluate']
    local_epochs = context.run_config['local-epochs']
    batch_size = context.run_config['batch-size']
    is_poisoned = context.run_config['poison']

    config_string = f"num_rounds={num_rounds}, local_epochs={local_epochs}, batch_size={batch_size}"
    metric_logger = MetricLogger(is_poisoned, num_rounds, config_string)
    device = get_device()
    model = get_model(device)
    params = ndarrays_to_parameters(model.state_dict().values())

    def fit_metrics_aggregation_fn(metrics):
        aggregated = weighted_average(metrics)
        metric_logger.log_metrics(len(metric_logger.metrics) + 1, aggregated)
        return {
            "val_loss": aggregated["loss"],
            "val_accuracy": aggregated["accuracy"]
        }

    strategy = FedAvgTrust(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=params,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
        trust_threshold=0.5,
    )

    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)