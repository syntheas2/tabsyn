import __init__ # noqa: F401
from typing import Tuple, Annotated
from zenml import step
from typing import Annotated, List
import torch
# at first try train tabsyn
import numpy as np
from pipelines.sample_args import SampleArgs
from zenml.logger import get_logger # ZenML's logger
import time
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample
import pandas as pd

logger = get_logger(__name__) # Use ZenML's logger for the step

@step
def sample_step(args: SampleArgs, train_z, pre_decoder, column_metadata, num_inverse, cat_inverse, model_diffusion_state_dict) -> Annotated[pd.DataFrame, "syn_df"]:
    device = args.device
    token_dim = args.vae_dim_token

    in_dim = train_z.shape[1] 
    num_samples = train_z.shape[0]

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    model.load_state_dict(model_diffusion_state_dict)

    '''
        Generating samples    
    '''
    start_time = time.time()

    
    sample_dim = in_dim
    x_next = sample(model.denoise_fn_D, num_samples, sample_dim, device=device)
    x_next = x_next * 2 + mean.to(device)

    # Später im Sampling-Prozess
    syn_data = x_next.float().cpu().numpy()
    syn_num, syn_cat_df, syn_target = split_num_cat_target(syn_data, pre_decoder, token_dim, column_metadata, num_inverse, cat_inverse, device)


    # Wiederherstellung der ursprünglichen Datenstruktur
    target_column = column_metadata.get('target_column')
    numerical_columns = column_metadata['numerical_columns'] 
    syn_df = recover_data(syn_num, syn_cat_df, syn_target, numerical_columns, target_column)

    
    end_time = time.time()
    logger.info('Time: ', end_time - start_time)

    return syn_df