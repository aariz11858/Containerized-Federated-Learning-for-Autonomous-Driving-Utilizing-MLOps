import flwr as fl
from ultralytics import YOLO
import os
import time
from clearml import Task
import torch
import gc
import psutil

# Load client-specific YAML file
CLIENT_ID = os.environ.get("CLIENT_ID", "client_1")
DATA_YAML = f"data.yaml"
MODEL_PATH = "best_final.pt"

# Initialize ClearML task
task = Task.init(project_name="FL-YOLOv9", task_name=f"Containerized {CLIENT_ID} FL Client")
logger = task.get_logger()

# Load model
model = YOLO(MODEL_PATH)

class YOLOClient(fl.client.NumPyClient):
    def get_parameters(self, config):
        return [v.cpu().numpy() for v in model.model.state_dict().values()]

    def set_parameters(self, parameters):
        state_dict = model.model.state_dict()
        new_state_dict = {}
        for (key, old_tensor), new_value in zip(state_dict.items(), parameters):
            new_state_dict[key] = torch.tensor(new_value)
        model.model.load_state_dict(new_state_dict)

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        round_num = int(config.get("server_round", 0))
        start = time.time()

        # Train
        model.train(data=DATA_YAML, epochs=1, imgsz=256, batch=4, verbose=False)
        duration = time.time() - start
        gc.collect()

        # Log training time
        logger.report_scalar("Training Time", "local_training_seconds", float(duration), round_num)

        # Log system metrics
        net_io = psutil.net_io_counters()
        logger.report_scalar("System", "Network_MB", net_io.bytes_sent / 1e6, round_num)
        logger.report_scalar("System", "CPU_Usage_%", psutil.cpu_percent(), round_num)
        logger.report_scalar("System", "RAM_Usage_%", psutil.virtual_memory().percent, round_num)

        return self.get_parameters(config), 100, {}

fl.client.start_numpy_client(server_address="3.14.66.81:8080", client=YOLOClient())
