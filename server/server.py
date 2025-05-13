import time
import flwr as fl
from clearml import Task
from ultralytics import YOLO
import torch
from flwr.common import parameters_to_ndarrays
import psutil
import copy  # For saving previous weights

# Initialize ClearML task
task = Task.init(project_name="FL-YOLOv9", task_name="Containerized FL Server")
logger = task.get_logger()

round_times = []
client_counts = []
map_scores = []
model_path = "server_model.pt"
VAL_YAML = "data.yaml"  # Centralized validation set
prev_parameters = None  # For storing previous round's parameters
prev_mAP = None         # For storing previous round's mAP@0.5

class TimedFedAvg(fl.server.strategy.FedAvg):
    def configure_fit(self, server_round, parameters, client_manager):
        connected_clients = len(client_manager.all())
        new_min_clients = min(connected_clients, 3)  # Minimum of 3 but increase if more join

        self.min_fit_clients = new_min_clients
        self.min_available_clients = new_min_clients

        print(f"[ROUND {server_round}] Adjusted min_fit_clients to {self.min_fit_clients} (Connected Clients: {connected_clients})")

        return super().configure_fit(server_round=server_round, parameters=parameters, client_manager=client_manager)


    def aggregate_fit(self, rnd, results, failures):
        global prev_parameters, prev_mAP

        start = time.time()
        parameters, metrics = super().aggregate_fit(rnd, results, failures)
        duration = time.time() - start
        logger.report_scalar("Round Duration", "duration", float(duration), int(rnd))

        client_count = len(results)
        client_counts.append(client_count)
        logger.report_scalar("Client Participation", "count", float(client_count), int(rnd))

        if parameters is None:
            print(f"[ROUND {rnd}] No parameters to aggregate (all clients failed)")
            return None, {}

        # Log model size and system usage
        size_mb = sum(len(p) for p in parameters.tensors) / 1e6
        logger.report_scalar("Model Size", "MB", float(size_mb), int(rnd))
        logger.report_scalar("System", "Server_CPU_Usage_%", psutil.cpu_percent(), int(rnd))
        logger.report_scalar("System", "Server_RAM_Usage_%", psutil.virtual_memory().percent, int(rnd))

        # Load YOLO model weights
        model = YOLO("best_final.pt")
        ndarrays = parameters_to_ndarrays(parameters)
        state_dict = model.model.state_dict()
        for k, v in zip(state_dict.keys(), ndarrays):
            state_dict[k] = torch.tensor(v)
        model.model.load_state_dict(state_dict)
        model.save(model_path)

        # Validation
        metrics = model.val(data=VAL_YAML, verbose=False)
        mAP50 = float(metrics.box.map50)
        mAP5095 = float(metrics.box.map)

        logger.report_scalar("Validation", "avg_mAP@0.5", mAP50, int(rnd))
        logger.report_scalar("Validation", "avg_mAP@0.5:0.95", mAP5095, int(rnd))

        # Data drift protection
        if rnd > 1 and prev_mAP is not None:
            drop_pct = (prev_mAP - mAP50) / prev_mAP
            if drop_pct > 0.10:
                print(f"[ROUND {rnd}] Detected mAP@0.5 drop of {drop_pct:.2%}, reverting to previous weights")
                logger.report_scalar("Drift", "Reverted", 1, int(rnd))
                return copy.deepcopy(prev_parameters), {"mAP@0.5": prev_mAP}

        prev_parameters = copy.deepcopy(parameters)
        prev_mAP = mAP50

        return parameters, {"mAP@0.5": mAP50}

if __name__ == "__main__":
    start_time = time.time()
    strategy = TimedFedAvg(
        fraction_fit=1.0,
        min_fit_clients=3,
        min_available_clients=3
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=5),
        strategy=strategy
    )

    total_time = time.time() - start_time
    logger.report_scalar("Summary", "Total FL Time (s)", float(total_time), 0)
    print(f"Total FL runtime: {total_time:.2f} seconds")
