import __init__ # noqa: F401
from zenml import pipeline
import mlflow  # Import the mlflow library
from datetime import datetime # For the run description
import torch
from steps.sample_load_data import sample_load_data_step
from steps.sample import sample_step
from pipelines.sample_args import SampleArgs


@pipeline
def sample_pipeline():
    args = SampleArgs()

    # --- Set the MLflow run name using a tag ---
    now = datetime.now() # Use current time for uniqueness
    timestamp_str = now.strftime("%Y-%m-%d_%H-%M-%S-%f")[:-3]
    mlflow.set_experiment(args.mlflow_experiment_name)
    mlflow.set_tag("mlflow.runName", f"{args.mlflow_experiment_name}_{timestamp_str}")
    
    if torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'

    train_z, pre_decoder, column_metadata, num_inverse, cat_inverse, model_diffusion_state_dict = sample_load_data_step(args)

    # Step 2: Train and evaluate the VAE model
    return sample_step(
        args=args,
        train_z=train_z,
        pre_decoder=pre_decoder,
        column_metadata=column_metadata,
        num_inverse=num_inverse,
        cat_inverse=cat_inverse,
        model_diffusion_state_dict=model_diffusion_state_dict
    )


if __name__ == "__main__":
    sample_pipeline.with_options(
        enable_cache=False  
    )()