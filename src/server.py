"""
This module sets up and runs a federated learning server using the Flower framework.

Author: Anh Nguyen, aln4739@rit.edu
"""

from flwr.common import Context, Metrics, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from typing import List, Tuple

from src.main import get_weights
from src.model import FemnistModel


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m['accuracy'] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {'accuracy': sum(accuracies) / sum(examples)}


def server_fn(context: Context):
    num_rounds = context.run_config['num-server-rounds']
    fraction_fit = context.run_config['fraction-fit']
    fraction_evaluate = context.run_config['fraction-evaluate']

    ndarrays = get_weights(FemnistModel())
    params = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_evaluate,
        min_available_clients=2,
        initial_parameters=params,
        evaluate_metrics_aggregation_fn=weighted_average,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Start server
app = ServerApp(server_fn=server_fn)
