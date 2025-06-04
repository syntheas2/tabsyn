from pydantic import BaseModel
import torch

class SampleArgs(BaseModel):
    # Data and Execution
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4

    # Model Hyperparameters (from your global constants)
    # MLPDiffusion Hyperparameters
    dim_token: int = 1024
    # MLPDiffusion Hyperparameters
    vae_dim_token: int = 4

    # Training Hyperparameters
    learning_rate: float = 1e-3 # LR
    weight_decay: float = 0.0   # WD
    num_epochs: int = 10000 
    scheduler_patience: int = 20
    scheduler_factor: float = 0.9
    early_stopping_patience: int = 500

    # MLflow
    mlflow_experiment_name: str = "Diffusion_Sample"
    manual_bestmodel_subdir: str = "model_best"
    bestmodels_runid: str = "c436c533a384449bba9c64da746f53ca"