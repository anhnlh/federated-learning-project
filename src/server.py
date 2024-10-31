"""
This module sets up and runs a federated learning server using the Flower framework.

Author: Anh Nguyen, aln4739@rit.edu, Ananya Misra, am4063@g.rit.edu;
"""

import csv
import os
from typing import List, Tuple

from flwr.common import Context, Metrics, ndarrays_to_parameters, Parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from .main import get_device
from .model import get_model


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Calculate the weighted average of accuracy and loss metrics. Function to be
    used as the aggregation function in the FedAvg strategy.

    Args:
        metrics (List[Tuple[int, Metrics]]): A list of tuples where each tuple contains
                                             the number of examples and a dictionary of metrics.

    Returns:
        Metrics: A dictionary containing the weighted average of accuracy and loss.
    """
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples)
    }


class MetricLogger:
    """
    A class used to log and save metrics during federated learning rounds.

    Attributes:
        folder (str): The directory where metrics will be saved.
        filename (str): The name of the file where metrics will be saved.
        metrics (list): A list to store metrics for each round.
        num_rounds (int): The total number of rounds to log metrics for.
    """

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
            print(
                f"Metrics saved successfully to {self.filename}")

    def save_metrics(self):
        with open(self.filename, "w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=["round", "loss", "accuracy"])
            writer.writeheader()
            writer.writerows(self.metrics)


class FedAvgTrust(FedAvg):
    """
    Custom Federated Averaging strategy that employs Trust and Reputation calculations to
    determine the trustworthiness of clients and remove malicious clients from the training.
    """

    def __init__(
            self,
            fraction_fit: float,
            fraction_evaluate: float,
            min_available_clients: int,
            initial_parameters: Parameters,
            fit_metrics_aggregation_fn,
            evaluate_metrics_aggregation_fn,
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
        self.t = 10.0  
        self.alpha = 0.5

    def _calculate_l2_distance(self, client_params, mean_params) -> float:
        
        total_distance = 0
        for client_layer, mean_layer in zip(client_params, mean_layer):
            diff = client_layer - mean_layer
            total_distance += float(np.sum(diff * diff))
        return np.sqrt(total_distance)
    
    def calc_reputation(self, client_id: str, distance: float) -> float:
        """
        Calculate the reputation of each client based on the metrics received from the client.
        TODO: Vectorize this if possible.
        :return:
        """
        R_prev = self.client_reputations.get(client_id, 1.0)
        d = distance    
        if d < self.alpha:
            R = (R_prev + d) - (R_prev / self.t)
        else:
            R = (R_prev + d) * np.exp(-(1 - d) * (R_prev / self.t)) 
        R = max(0.0, min(1.0, R))
        return R

    def calc_trust(self, R: float, d: float) -> float:
        """
        Calculate trust score based on reputation and distance.
        
        Args:
            R: Reputation score
            d: L2 distance
        Returns:
            float: Trust score between 0 and 1
        """
        trust = np.sqrt((R * R) + (d * d)) - np.sqrt((1 - R) * (1 - R) + (1 - d) * (1 - d))
        
        if trust >= self.trust_threshold:
            return 1.0
        return 0.0


def server_fn(context: Context):
    """
    Initialize and configure the federated learning server.

    Args:
        context (Context): The context object containing the run configuration.

    Returns:
        ServerAppComponents: The components required to run the federated learning server.
    """
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
    )

    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


app = ServerApp(server_fn=server_fn)
