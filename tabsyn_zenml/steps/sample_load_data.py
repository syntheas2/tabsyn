import __init__ # noqa: F401
from typing import Tuple, Annotated, Any
from zenml import step
from typing import Annotated, List
from zenml.client import Client
import torch
# at first try train tabsyn
import numpy as np
from pythelpers.ml.mlflow import load_best_model
from pipelines.sample_args import SampleArgs



@step
def sample_load_data_step(args: SampleArgs) -> Tuple[
    Annotated[Any, "train_z"],
    Annotated[Any, "pre_decoder"],
    Annotated[Any, "column_metadata"],
    Annotated[Any, "num_inverse"],
    Annotated[Any, "cat_inverse"],
    Annotated[Any, "model_diffusion_state_dict"],
]:
    artifact = Client().get_artifact_version("fed88f42-789c-4138-a795-dcf42c590d31")
    column_metadata = artifact.load()

    artifact = Client().get_artifact_version(
    "2522336c-aba2-4085-abae-a61f4739e187")
    train_z = artifact.load()
    train_z = torch.Tensor(train_z).float()
    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    artifact3 = Client().get_artifact_version(
    "ae69d283-ceab-421d-a587-989f5ee58c69")
    pre_decoder = artifact3.load()

    artifact = Client().get_artifact_version("fc2f6b53-7ec1-4b69-8525-4b476ab7fb25")
    num_inverse = artifact.load()

    artifact = Client().get_artifact_version("81e5d2ed-4de9-4d2e-b725-a3c9b8518e5c")
    cat_inverse = artifact.load()

    model_diffusion_state_dict = load_best_model(
            run_id = args.bestmodels_runid,
            artifact_subdir = args.manual_bestmodel_subdir,
            device= args.device,
        )['model_state_dict']


    return train_z, pre_decoder, column_metadata, num_inverse, cat_inverse, model_diffusion_state_dict