import __init__ # Your local package structure if needed
from typing import Annotated, Dict, Any
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from zenml import step, get_step_context # Keep ZenML decorator and context
from zenml.logger import get_logger # ZenML's logger
from tqdm import tqdm
import time
import mlflow
import mlflow.pytorch
import tempfile # For temporary file handling
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model

# Assuming these are correctly importable from your project structure
from tabsyn.vae.main import compute_loss # Assuming this is your loss function
from utils_train import TabularDataset # Ensure this is findable
from pipelines.train_diffusion_args import DiffusionArgs # Your configuration class
from pythelpers.ml.mlflow import load_latest_checkpoint_from_mlflow, rotate_checkpoints, rotate_bestmodels

logger = get_logger(__name__) # Use ZenML's logger for the step

ReturnType = Annotated[Dict[str, Any], "model_dict"] # Define return type for clarity

@step(enable_cache=False) # ZenML step decorator
def train_evaluate_diffusion(
    train_z,
    config: DiffusionArgs, # Your configuration class
) -> ReturnType: # Returns path to the best model's state_dict (ZenML output)

    current_run = mlflow.active_run()
    if not current_run:
        # This can happen if autolog starts a run implicitly and we try to access it too soon
        # or if running outside a `with mlflow.start_run():` block when not relying on autolog to create it.
        # For robust explicit run management, you'd wrap the core logic in `with mlflow.start_run() as run:`.
        # However, ZenML + autolog usually handles run creation. If this becomes an issue, explicit run start is needed.
        logger.info("No active MLflow run found initially, will rely on autolog or subsequent calls to create/get it.")
        # Attempt to get it again, as autolog might initialize it.
        # If this is still None later when needed, it's an issue.
        # For now, we assume autolog handles it or it's available when `load_latest_checkpoint_from_mlflow` is called.
    
    run_id = current_run.info.run_id if current_run else None


    mlflow.pytorch.autolog(
        log_models=True, 
        checkpoint=True, 
        disable_for_unsupported_versions=True,
        registered_model_name=None
    )

    logger.info(f"Starting Diffusion training on device: {config.device}. MLflow autologging enabled.")

    # Log all config parameters manually
    mlflow.log_params(config.model_dump())

    device = config.device


    in_dim = train_z.shape[1] 

    # normalisierung um 0 -> für stabilere gradienten (mehr in readme)
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    train_data = train_z

    train_loader = DataLoader(
        train_data,
        batch_size = config.batch_size,
        shuffle = True,
        num_workers = config.num_workers,
    )

    num_epochs = config.num_epochs + 1

    denoise_fn = MLPDiffusion(in_dim, config.dim_token).to(device)
    logger.info(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    logger.info(f"the number of parameters {num_params}")

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    lr = config.learning_rate
    patience = 0
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=config.weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=config.scheduler_factor, patience=config.scheduler_patience)

    best_model_data_to_save = None
    best_loss = float('inf')  # Initialize best_loss if no checkpoint exists    
    start_epoch = 0
    if config.load_from_checkpoint:
        ckp_run_id = config.load_ckp_from_run_id
        if ckp_run_id:
            logger.info(f"Attempting to load checkpoint from MLflow run '{ckp_run_id}', subdir '{config.manual_checkpoint_subdir}'...")
            checkpoint_data = load_latest_checkpoint_from_mlflow(
                run_id=ckp_run_id,
                artifact_subdir=config.manual_checkpoint_subdir,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                device=config.device
            )
            if checkpoint_data:
                start_epoch = checkpoint_data.get('epoch', -1) + 1 # Resume from NEXT epoch
                best_loss = checkpoint_data.get('best_val_loss', float('inf'))
                patience = checkpoint_data.get('current_patience', 0)
                best_model_data_to_save = {
                    'epoch': start_epoch,
                    'model_state_dict': model.state_dict(),
                    'validation_loss': best_loss,
                }
                
                # Get the saved learning rate
                saved_lr = checkpoint_data.get('learning_rate', None)
                if saved_lr is not None:
                    logger.info(f"Loaded learning rate: {saved_lr}")
                    # Update the learning rate in the optimizer if needed
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = saved_lr
                else:
                    # Evaluate the loaded model to get its initial performance
                    model.eval()
                    batch_loss = 0.0
                    len_input = 0
                    with torch.no_grad():
                        for X_test_batch in train_loader:
                            X_test_batch = X_test_batch.float().to(device)
                            loss = model(X_test_batch)
                            loss = loss.mean()
                            batch_loss += loss.item() * len(X_test_batch)
                            len_input += len(X_test_batch)

                    best_loss = batch_loss/len_input  # Compute mean loss
                # Model, optimizer, scheduler states are loaded by the helper
                logger.info(f"Successfully loaded checkpoint. Resuming training from epoch {start_epoch}.")
            else:
                logger.info("No suitable checkpoint found. Starting training from scratch.")
        else:
            logger.warning("MLflow run ID not available, cannot attempt to load checkpoint.")

    old_best_loss = best_loss
    model.train()
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs):
        
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        batch_loss = 0.0
        len_input = 0
        for batch in pbar:
            inputs = batch.float().to(device)
            loss = model(inputs)
        
            loss = loss.mean()

            batch_loss += loss.item() * len(inputs)
            len_input += len(inputs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            pbar.set_postfix({"Loss": loss.item()})

        curr_loss = batch_loss/len_input
        scheduler.step(curr_loss)
        new_lr = optimizer.param_groups[0]['lr']

        logger.info(
            f"Epoch {epoch+1}: Train Loss: {curr_loss:.4f} | Learning Rate: {new_lr:.6f}"
        )

        mlflow.log_metrics({
            "train_loss_epoch": curr_loss,
            "learning_rate": new_lr
        }, step=epoch)       

        if curr_loss < best_loss:
            best_loss = curr_loss
            pbar.set_postfix({"Best Loss": best_loss})
            # Force refresh the progress bar to show the update
            pbar.update(0)
            patience = 0
            # Create a checkpoint dictionary with model state and learning rate
            best_model_data_to_save = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'validation_loss': curr_loss,
                # Add any other specific metadata you want for the "best model" artifact
            }
        else:
            patience += 1
            if patience == config.early_stopping_patience:
                logger.info('Early stopping')
                break

        save_this_epoch = (epoch % config.checkpoint_save_interval == 0) or \
                          (epoch == config.num_epochs - 1)
        if save_this_epoch and run_id:
            checkpoint_state: Dict[str, Any] = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_loss,
                'current_patience': patience,
                'config_dump': config.model_dump() # For reference
            }
            
            ckpt_filename = f"model_checkpoint_epoch_{epoch:04d}.pt" # Padded epoch for sorting
            with tempfile.TemporaryDirectory() as tmpdir:
                local_tmp_checkpoint_path = Path(tmpdir) / ckpt_filename
                torch.save(checkpoint_state, local_tmp_checkpoint_path)
                
                mlflow.log_artifact(
                    str(local_tmp_checkpoint_path), 
                    artifact_path=f"{config.manual_checkpoint_subdir}" 
                )
            # --- Checkpoint Rotation ---
            rotate_checkpoints(
                run_id=run_id,
                artifact_subdir=config.manual_checkpoint_subdir,
                max_checkpoints=config.max_checkpoints_to_keep
            )
            logger.info(f"Saved manual checkpoint to MLflow: {config.manual_checkpoint_subdir}")

            if old_best_loss > best_loss:
                bestmodel_filename = f"model_loss_{best_loss:.3f}.pt" # Padded epoch for sorting
                with tempfile.TemporaryDirectory() as tmpdir:
                    local_tmp_checkpoint_path = Path(tmpdir) / bestmodel_filename
                    torch.save(checkpoint_state, local_tmp_checkpoint_path)
                    
                    mlflow.log_artifact(
                        str(local_tmp_checkpoint_path), 
                        artifact_path=f"{config.manual_bestmodel_subdir}",
                        run_id=config.bestmodels_runid 
                    )
                            # --- Checkpoint Rotation ---
                rotate_bestmodels(
                    run_id=config.bestmodels_runid,
                    metric='loss',
                    artifact_subdir=config.manual_bestmodel_subdir,
                    max=config.max_checkpoints_to_keep
                )
                logger.info(f"Saved manual best model to MLflow: {config.manual_bestmodel_subdir}")
                old_best_loss = best_loss


    end_time = time.time()
    logger.info('Time: ', end_time - start_time)
    return best_model_data_to_save



    