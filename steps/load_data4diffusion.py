import __init__ # noqa: F401
from scipy.sparse import csr_matrix
from typing import Tuple, Annotated
import pandas as pd
from zenml import step
from typing import Annotated, List
from zenml.client import Client
import torch
from utils import get_args
# at first try train tabsyn
from pathlib import Path
import numpy as np
from tabsyn.vae.main import main as main_fn_vae
from tabsyn.vae.main import transform_preprocessed_data



@step
def load_data4diffusion_step() -> Annotated[torch.Tensor, "train_z_latents"]:
    artifact = Client().get_artifact_version(
        "2522336c-aba2-4085-abae-a61f4739e187")
    train_z = artifact.load()
    train_z = torch.Tensor(train_z).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z