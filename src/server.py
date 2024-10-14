"""
This module sets up and runs a federated learning server using the Flower framework.

Author: Anh Nguyen, aln4739@rit.edu, Ananya Misra, am4063@g.rit.edu;
"""

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple
import csv
import os

from src.model import get_model
from src.main import get_device

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    losses = [num_examples * m["loss"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {
        "accuracy": sum(accuracies) / sum(examples),
        "loss": sum(losses) / sum(examples)
    }

class MetricLogger:
    def __init__(self, is_poisoned):
        self.folder = "results/attack" if is_poisoned else "results/no_attack"
        os.makedirs(self.folder, exist_ok=True)
        self.metrics = []

    def log_metrics(self, round_number, metrics):
        self.metrics.append({"round": round_number, **metrics})

    def save_metrics(self):
        filename = f"{self.folder}/global_metrics.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["round", "loss", "accuracy"])
            writer.writeheader()
            writer.writerows(self.metrics)

def server_fn(context: Context):
    num_rounds = context.run_config['num-server-rounds']
    fraction_fit = context.run_config['fraction-fit']
    fraction_evaluate = context.run_config['fraction-evaluate']
    is_poisoned = context.run_config.get('poison', False)

    device = get_device()
    model = get_model(device)
    params = ndarrays_to_parameters(model.state_dict().values())

    metric_logger = MetricLogger(is_poisoned)

    def fit_metrics_aggregation_fn(metrics):
        aggregated = weighted_average(metrics)
        metric_logger.log_metrics(len(metric_logger.metrics) + 1, aggregated)
        return aggregated

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=params,
        fit_metrics_aggregation_fn=fit_metrics_aggregation_fn,
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    config = ServerConfig(num_rounds=num_rounds)

    return strategy, config, [metric_logger.save_metrics]

app = ServerApp(server_fn)