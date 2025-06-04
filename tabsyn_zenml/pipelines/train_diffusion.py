import __init__ # noqa: F401
from zenml import pipeline
import mlflow  # Import the mlflow library
from datetime import datetime # For the run description
import torch
from steps.load_data4diffusion import load_data4diffusion_step
from steps.train_diffusion import train_evaluate_diffusion
from pipelines.train_diffusion_args import DiffusionArgs


@pipeline
def train_diffusion_pipeline():
    args = DiffusionArgs()

    # --- Set the MLflow run name using a tag ---
    now = datetime.now() # Use current time for uniqueness
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.set_tag("mlflow.runName", f"{args.mlflow_experiment_name}_{timestamp_str}")
    
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    train_z = load_data4diffusion_step()

    # Step 2: Train and evaluate the VAE model
    return train_evaluate_diffusion(
        train_z=train_z,
        config=args
    )


if __name__ == "__main__":
    train_diffusion_pipeline.with_options(
        enable_cache=False  
    )()