import argparse
from logging import DEBUG, INFO

from xmlrpc.client import boolean
import flwr as fl
from flwr.common.typing import Scalar
from flwr.server.server import Server
from flwr.server.client_manager import ClientManager, SimpleClientManager
from flwr.common.logger import log

import ray
import torch
import torchvision
import numpy as np
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Callable, Optional, Tuple, List

from divide_data import select_dataset
from dataset_utils import get_cifar_10, do_fl_partitioning, get_dataloader, \
    get_femnist, do_femnist_partitioning, \
    femnistTransformation, cifar10Transformation

from utils import Net, train, test


from resnet_torch import resnet18 as resnet18_torch


parser = argparse.ArgumentParser(description="Flower Simulation with PyTorch")

parser.add_argument("--num_client_cpus", type=int, default=8)
parser.add_argument("--num_rounds", type=int, default=1000)

# parser.add_argument("--data_dir", type=str, default="/home/chaoyanghe/FedScale/benchmark/dataset/data/femnist/data")
parser.add_argument("--data_dir", type=str, default="/home/chaoyanghe/FedScale/benchmark/dataset/data/femnist")
parser.add_argument("--num_class", type=int, default=62)
parser.add_argument("--num_participants", type=int, default=32)
parser.add_argument("--data_map_file", type=str, default="/home/chaoyanghe/FedScale/benchmark/dataset/data/femnist/client_data_mapping/train.csv")
parser.add_argument("--task", type=str, default="cv")

parser.add_argument("--model", type=str, default="resnet18")

parser.add_argument("--client_optimizer", type=str, default="sgd")

parser.add_argument("--dataset", type=str, default="femnist")
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--run_name", type=str, default="fedml_optim_bench")

parser.add_argument("--learning_rate", type=float, default=0.05)
parser.add_argument("--frequency_of_the_test", type=int, default=10)
parser.add_argument("--federated_optimizer", type=str, default="FedAvg")
parser.add_argument("--num_loaders", type=int, default=0)

# parser.add_argument("--enable_wandb", type=boolean, default=False)
parser.add_argument("--wandb_entity", type=str, default="automl")
parser.add_argument("--wandb_key", type=str, default="ee0b5f53d949c84cee7decbe7a629e63fb2f8408")
parser.add_argument("--wandb_project", type=str, default="bench_optim")
parser.add_argument("--wandb_name", type=str, default="fedml_optim_bench")

parser.add_argument("--worker_num", type=int, default=8)
# worker_num





def Net(args):
    # if args.dataset == "femnist":
    #     return resnet18_torch(num_classes=62, in_channels=1)
    # else:
    return resnet18_torch(num_classes=62, in_channels=3)


# Flower client, adapted from Pytorch quickstart example
class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid: str, fed_dir_data: str, args):
        self.cid = cid
        self.fed_dir = Path(fed_dir_data)
        self.properties: Dict[str, Scalar] = {"tensor_type": "numpy.ndarray"}
        self.args = args

        # Instantiate model
        self.net = Net(args)

        # Determine device
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


    def get_parameters(self, config):
        return get_params(self.net)

    def fit(self, parameters, config):
        set_params(self.net, parameters)

        # # raise RuntimeError
        # # Load data for this client and get trainloader
        # num_workers = len(ray.worker.get_resource_ids()["CPU"])
        num_workers = args.num_loaders
        trainloader = get_dataloader(
            self.fed_dir,
            self.cid,
            is_train=True,
            batch_size=config["batch_size"],
            workers=num_workers,
            transform=femnistTransformation()
        )

        # # Send model to device
        self.net.to(self.device)

        # # Train
        train(self.net, trainloader, epochs=config["epochs"], device=self.device)
        # log(INFO, f"Client {self.cid} Finish Training")

        # Return local model and statistics
        return get_params(self.net), len(trainloader.dataset), {}
        # return get_params(self.net), len([]), {}

    def evaluate(self, parameters, config):
        return float(0.0), 10, {"accuracy": float(0.1)}
        # set_params(self.net, parameters)

        # Load data for this client and get trainloader
        # num_workers = len(ray.worker.get_resource_ids()["CPU"])
        # valloader = get_dataloader(
        #     self.fed_dir, self.cid, is_train=False, batch_size=50, workers=num_workers
        # )
        # valloader = select_dataset(self.cid, self.testing_sets,
        #                                 batch_size=config["batch_size"], args=self.args,
        #                                 collate_fn=self.collate_fn
        #                                 )
        # # Send model to device
        self.net.to(self.device)

        # Evaluate
        loss, accuracy = test(self.net, valloader, device=self.device)

        # Return statistics
        return float(loss), len(valloader.dataset), {"accuracy": float(accuracy)}


def fit_config(server_round: int) -> Dict[str, Scalar]:
    """Return a configuration with static batch size and (local) epochs."""
    config = {
        "epochs": args.epochs,  # number of local epochs
        "batch_size": args.batch_size,
        "args": args,
    }
    return config


def get_params(model: torch.nn.ModuleList) -> List[np.ndarray]:
    """Get model weights as a list of NumPy ndarrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def set_params(model: torch.nn.ModuleList, params: List[np.ndarray]):
    """Set model weights from a list of NumPy ndarrays."""
    params_dict = zip(model.state_dict().keys(), params)
    state_dict = OrderedDict({k: torch.from_numpy(np.copy(v)) for k, v in params_dict})
    model.load_state_dict(state_dict, strict=True)


def get_evaluate_fn(
    testset: torchvision.datasets.CIFAR10, args=None
) -> Callable[[fl.common.NDArrays], Optional[Tuple[float, float]]]:
    """Return an evaluation function for centralized evaluation."""

    def evaluate(
        server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, float]]:
        """Use the entire CIFAR-10 test set for evaluation."""

        # determine device
        # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        device = torch.device("cpu")

        model = Net(args)
        set_params(model, parameters)
        model.to(device)

        testloader = torch.utils.data.DataLoader(testset, batch_size=50)
        loss, accuracy = test(model, testloader, device=device)

        # return statistics
        return loss, {"accuracy": accuracy}
    return evaluate

    # def nothing(
    #     server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]
    # ):
    #     return 0.0, {"accuracy": 0.0}

    # return nothing


# Start simulation (a _default server_ will be created)
# This example does:
# 1. Downloads CIFAR-10
# 2. Partitions the dataset into N splits, where N is the total number of
#    clients. We refere to this as `pool_size`. The partition can be IID or non-IID
# 3. Starts a simulation where a % of clients are sample each round.
# 4. After the M rounds end, the global model is evaluated on the entire testset.
#    Also, the global model is evaluated on the valset partition residing in each
#    client. This is useful to get a sense on how well the global model can generalise
#    to each client's data.
if __name__ == "__main__":

    # parse input arguments
    args = parser.parse_args()
    args.enable_wandb = True
    # args.enable_wandb = False


    pool_size = 2800  # number of dataset partions (= number of total clients)
    client_resources = {
        "num_cpus": args.num_client_cpus,
        "num_gpus": 1,
    }  # each client will get allocated 1 CPUs
    # client_resources = {
    #     "num_cpus": args.num_client_cpus,
    # }  # each client will get allocated 1 CPUs

    # Download CIFAR-10 dataset
    train_path, testset = get_femnist(path_to_data="./data", args=args)

    # partition dataset (use a large `alpha` to make it IID;
    # a small value (e.g. 1) will make it non-IID)
    # This will create a new directory called "federated": in the directory where
    # CIFAR-10 lives. Inside it, there will be N=pool_size sub-directories each with
    # its own train/set split.
    # fed_dir = do_fl_partitioning(
    #     train_path, pool_size=pool_size, alpha=1000, num_classes=10, val_ratio=0.1
    # )

    # fed_dir = do_femnist_partitioning(
    #     train_path, pool_size=pool_size, alpha=1000, num_classes=10, args=args, val_ratio=0.1
    # )
    fed_dir = train_path.parent / "federated"


    # configure the strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.001,
        fraction_evaluate=0.001,
        min_fit_clients=args.num_participants,
        min_evaluate_clients=10,
        min_available_clients=pool_size,  # All clients should be available
        on_fit_config_fn=fit_config,
        evaluate_fn=get_evaluate_fn(testset, args),  # centralised evaluation of global model
        args=args,
    )
    # strategy = fl.server.strategy.FedAvg(
    #     fraction_fit=0.1,
    #     fraction_evaluate=0.1,
    #     min_fit_clients=100,
    #     min_evaluate_clients=10,
    #     min_available_clients=pool_size,  # All clients should be available
    #     on_fit_config_fn=fit_config,
    #     evaluate_fn=get_evaluate_fn(""),  # centralised evaluation of global model
    # )

    def client_fn(cid: str):
        # create a single client instance
        return FlowerClient(cid, fed_dir, args)

    # (optional) specify Ray config
    ray_init_args = {"include_dashboard": False}

    client_manager = SimpleClientManager()
    server = Server(client_manager=client_manager, strategy=strategy)
    server.set_max_workers(args.worker_num)
    # start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=pool_size,
        client_resources=client_resources,
        config=fl.server.ServerConfig(num_rounds=args.num_rounds),
        strategy=strategy,
        ray_init_args=ray_init_args,
        server=server,
    )


    # client = FlowerClient("1", fed_dir, args)

    # trainloader = get_dataloader(
    #     client.fed_dir,
    #     client.cid,
    #     is_train=True,
    #     batch_size=10,
    #     workers=2,
    #     transform=femnistTransformation()
    # )

    # print(f"trainloader.dataset: {trainloader.dataset}")
    # print(f"trainloader.dataset.data: {trainloader.dataset.data}")
    # print(f"trainloader.dataset.targets: {trainloader.dataset.targets}")

    # for img, label in trainloader:
    #     print(f"img: {img.shape}, label: {label}")

    # num_workers = args.num_loaders
    # trainloader = get_dataloader(
    #     client.fed_dir,
    #     client.cid,
    #     is_train=True,
    #     batch_size=20,
    #     workers=num_workers,
    #     transform=femnistTransformation()
    # )

    # # # Send model to device
    # client.net.to(client.device)

    # # # Train
    # train(client.net, trainloader, epochs=5, device=client.device)







