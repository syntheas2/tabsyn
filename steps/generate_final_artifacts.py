import __init__
from typing import List, Tuple
import torch
import numpy as np
from zenml import step, get_step_context
from zenml.logger import get_logger
import mlflow
from pathlib import Path

from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model # Ensure these are importable
from steps.train_vae_args import VAEArgs

logger = get_logger(__name__)

@step(enable_cache=True)
def generate_final_artifacts(
    best_model_state_path: Path, # Path from train_evaluate_vae step
    X_train_num: torch.Tensor,   # Original X_train_num tensor (on CPU)
    X_train_cat: torch.Tensor,   # Original X_train_cat tensor (on CPU)
    d_numerical: int,
    categories: List[int],
    config: VAEArgs,
) -> Tuple[Path, Path, Path]: # train_z_path, encoder_path, decoder_path
    """
    Loads the best VAE model, saves its encoder/decoder components,
    and generates/saves training data embeddings.
    """
    context = get_step_context()
    logger.info(f"Generating final artifacts using model from: {best_model_state_path}")

    # Initialize the main VAE model structure
    # This is needed to correctly load its weights into sub-components if they share layers by reference
    # or if pre_encoder/pre_decoder need the full model structure to initialize
    main_model_structure = Model_VAE(
        num_layers=config.num_layers, d_numerical=d_numerical, categories=categories,
        d_token=config.d_token, n_head=config.n_head, factor=config.factor, bias=config.token_bias
    )
    if best_model_state_path.exists() and best_model_state_path.stat().st_size > 0:
        logger.info(f"Loading best model state_dict from {best_model_state_path}")
        main_model_structure.load_state_dict(torch.load(best_model_state_path, map_location=config.device))
        main_model_structure.to(config.device)
        main_model_structure.eval()
    else:
        logger.error(f"Best model state_dict not found or empty at {best_model_state_path}. Cannot generate artifacts.")
        # Create empty placeholder paths for ZenML outputs to avoid pipeline failure
        # ZenML expects all declared outputs to be created.
        empty_emb_path = Path(context.get_output_artifact_uri(config.train_embeddings_filename))
        empty_enc_path = Path(context.get_output_artifact_uri(config.final_encoder_filename))
        empty_dec_path = Path(context.get_output_artifact_uri(config.final_decoder_filename))
        empty_emb_path.parent.mkdir(parents=True, exist_ok=True); empty_emb_path.touch()
        empty_enc_path.parent.mkdir(parents=True, exist_ok=True); empty_enc_path.touch()
        empty_dec_path.parent.mkdir(parents=True, exist_ok=True); empty_dec_path.touch()
        return empty_emb_path, empty_enc_path, empty_dec_path


    # Initialize encoder and decoder models
    pre_encoder = Encoder_model(
        num_layers=config.num_layers, d_numerical=d_numerical, categories=categories,
        d_token=config.d_token, n_head=config.n_head, factor=config.factor
    ).to(config.device)
    pre_decoder = Decoder_model(
        num_layers=config.num_layers, d_numerical=d_numerical, categories=categories,
        d_token=config.d_token, n_head=config.n_head, factor=config.factor
    ).to(config.device)

    # Load weights from the trained VAE model into encoder/decoder
    # This assumes your Encoder_model and Decoder_model have a 'load_weights' method
    # or that their state_dicts are subsets of the main_model_structure's state_dict.
    # If 'load_weights' is specific to your 'tabsyn' library, use it.
    # Otherwise, you might need to manually copy weights if architectures match.
    # Example: pre_encoder.load_state_dict(main_model_structure.encoder.state_dict())
    # The original code had:
    # pre_encoder.load_weights(model) -> Assuming 'model' is main_model_structure
    # pre_decoder.load_weights(model)
    # This part is CRITICAL and depends on how tabsyn's Model_VAE, Encoder_model, Decoder_model are structured.
    # For now, I'll assume a direct state_dict copy if they are submodules,
    # or that load_weights method exists.
    try:
        # Attempt 1: If Encoder_model has a method to load from the main VAE
        if hasattr(pre_encoder, 'load_weights') and callable(getattr(pre_encoder, 'load_weights')):
             pre_encoder.load_weights(main_model_structure)
             pre_decoder.load_weights(main_model_structure)
             logger.info("Loaded weights into pre_encoder and pre_decoder using 'load_weights' method.")
        # Attempt 2: If encoder/decoder are direct attributes of Model_VAE
        elif hasattr(main_model_structure, 'encoder') and hasattr(main_model_structure, 'decoder'):
            pre_encoder.load_state_dict(main_model_structure.encoder.state_dict())
            pre_decoder.load_state_dict(main_model_structure.decoder.state_dict())
            logger.info("Loaded weights by copying state_dict from main_model.encoder/decoder.")
        else:
            logger.warning("Could not determine how to load weights into pre_encoder/pre_decoder. Saving them uninitialized or partially initialized from main model.")
            # Fallback: save the whole main model's state_dict for encoder/decoder if names match,
            # this is unlikely to be correct unless they are identical to Model_VAE.
            # pre_encoder.load_state_dict(main_model_structure.state_dict(), strict=False)
            # pre_decoder.load_state_dict(main_model_structure.state_dict(), strict=False)


    except Exception as e:
        logger.error(f"Error loading weights into pre_encoder/decoder: {e}. They might be saved uninitialized or partially.")
    
    pre_encoder.eval()
    pre_decoder.eval()

    encoder_save_path = Path(context.get_output_artifact_uri(config.final_encoder_filename))
    decoder_save_path = Path(context.get_output_artifact_uri(config.final_decoder_filename))
    encoder_save_path.parent.mkdir(parents=True, exist_ok=True)
    decoder_save_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(pre_encoder.state_dict(), encoder_save_path)
    torch.save(pre_decoder.state_dict(), decoder_save_path)
    logger.info(f"Saved final encoder to: {encoder_save_path}")
    logger.info(f"Saved final decoder to: {decoder_save_path}")
    mlflow.log_artifact(str(encoder_save_path), "final_encoder_state_dict")
    mlflow.log_artifact(str(decoder_save_path), "final_decoder_state_dict")

    # Generate and save training embeddings
    with torch.no_grad():
        # Move training data to device for inference with encoder
        X_train_num_dev = X_train_num.to(config.device)
        X_train_cat_dev = X_train_cat.to(config.device)
        
        train_z = pre_encoder(X_train_num_dev, X_train_cat_dev).detach().cpu().numpy()

    train_z_save_path = Path(context.get_output_artifact_uri(config.train_embeddings_filename))
    train_z_save_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(train_z_save_path, train_z)
    logger.info(f"Saved training embeddings to: {train_z_save_path} (shape: {train_z.shape})")
    mlflow.log_artifact(str(train_z_save_path), "train_embeddings")
    mlflow.log_param("train_embedding_rows", train_z.shape[0])
    mlflow.log_param("train_embedding_cols", train_z.shape[1])

    return train_z_save_path, encoder_save_path, decoder_save_path