from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg

from src.main import get_weights
from src.model import FemnistModel


def server_fn(context: Context):
    num_rounds = context.run_config['num-server-rounds']
    fraction_fit = context.run_config['fraction-fit']

    ndarrays = get_weights(FemnistModel())
    params = ndarrays_to_parameters(ndarrays)

    strategy = FedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=params,
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Start server
app = ServerApp(server_fn=server_fn)
