from pydantic import BaseModel
import torch

class DiffusionArgs(BaseModel):
    # Data and Execution
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4096
    num_workers: int = 4

    # Model Hyperparameters (from your global constants)
    # MLPDiffusion Hyperparameters
    dim_token: int = 1024

    # Training Hyperparameters
    learning_rate: float = 1e-3 # LR
    weight_decay: float = 0.0   # WD
    num_epochs: int = 10000 
    scheduler_patience: int = 20
    scheduler_factor: float = 0.9
    early_stopping_patience: int = 500

    # MLflow
    mlflow_experiment_name: str = "Diffusion_Training_Experiment"
    load_ckp_from_run_id: str = ""
    load_from_checkpoint: bool = True
    max_checkpoints_to_keep: int = 5
    checkpoint_save_interval: int = 20