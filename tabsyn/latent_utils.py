import os
import json
import numpy as np
import pandas as pd
import torch
from utils_train import preprocess, load_preprocessed_csv
from tabsyn.vae.model import Decoder_model 
from pathlib import Path

script_dir = Path(__file__).parent
data_outdir = str(script_dir / '../output')
data_indir = str(script_dir / '../input')

def get_input_train(args):
    dataname = args.dataname

    curr_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_path = Path(data_indir) / dataname

    # todo:
    # with open(f'{dataset_path}/info.json', 'r') as f:
    #     info = json.load(f)

    ckpt_dir = args.ckpt_dir
    embedding_save_path = args.ckpt_dir_vae + f'train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]
    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)

    return train_z, curr_dir, dataset_path, ckpt_dir


def get_input_generate(args):
    _, _, categories, d_numerical, num_inverse, cat_inverse, original_columns, column_metadata = load_preprocessed_csv(args.datapath, args.datatestpath, inverse = True)

    embedding_save_path = f'{args.ckpt_dir_vae}/train_z.npy'
    train_z = torch.tensor(np.load(embedding_save_path)).float()

    train_z = train_z[:, 1:, :]

    B, num_tokens, token_dim = train_z.size()
    in_dim = num_tokens * token_dim
    
    train_z = train_z.view(B, in_dim)
    pre_decoder = Decoder_model(2, d_numerical, categories, 4, n_head = 1, factor = 32)

    pre_decoder.load_state_dict(torch.load(args.vaedecoderpath))


    return train_z, pre_decoder, token_dim, num_inverse, cat_inverse, original_columns, column_metadata


 
@torch.no_grad()
def split_num_cat_target(syn_data, pre_decoder, token_dim, column_metadata, num_inverse, cat_inverse, device=None):
    """
    Split synthesized data using the pre-trained decoder model.
    
    Args:
        syn_data: The synthesized data from the model
        pre_decoder: The pre-trained decoder model
        token_dim: Dimension of each token
        column_metadata: Metadata about columns
        num_inverse: Function to inverse transform numerical data
        cat_inverse: Function to inverse transform categorical data
        device: Optional device for torch tensors
        
    Returns:
        tuple: (syn_num, syn_cat, syn_target) - numerical, categorical and target data
    """
    import torch
    import numpy as np
    
    # Bestimmen, ob wir eine Zielspalte haben
    target_column = column_metadata.get('target_column')
    has_target = target_column is not None
    
    # Dimensionen erfassen
    numerical_columns = column_metadata['numerical_columns']
    categorical_groups = column_metadata['categorical_groups']
    
    # Umformen der synthetischen Daten für den Pre-Decoder
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    syn_data_tensor = torch.tensor(syn_data, device=device)
    
    # Reshape für den Decoder - von flacher Repräsentation zu tokenisierter Form
    syn_data_reshaped = syn_data_tensor.reshape(syn_data_tensor.shape[0], -1, token_dim)
    
    # Pre-Decoder anwenden, um numerische und kategorische Features zu erhalten
    norm_input = pre_decoder(syn_data_reshaped)
    x_hat_num, x_hat_cat = norm_input
    
    # Kategorische Vorhersagen in Indizes konvertieren
    syn_cat = []
    for pred in x_hat_cat:
        syn_cat.append(pred.argmax(dim=-1))
    
    # Zu NumPy konvertieren
    syn_num = x_hat_num.cpu().numpy()
    syn_cat = torch.stack(syn_cat).t().cpu().numpy() if syn_cat else np.array([])
    
    # Inverse Transformationen anwenden
    syn_num = num_inverse(syn_num)
    
    # Hier müssen wir cat_inverse anpassen, da es nun mit Indizes arbeitet, nicht mit One-Hot
    if hasattr(cat_inverse, "__call__"):
        syn_cat = cat_inverse(syn_cat)
    
    # Ziel extrahieren, falls vorhanden
    if has_target and target_column in numerical_columns:
        # Annahme: Ziel ist die erste numerische Spalte
        syn_target = syn_num[:, :1]  # Erste Spalte als Ziel
        syn_num = syn_num[:, 1:]     # Rest als Features
    else:
        # Keine Zielspalte oder Ziel ist kategorisch
        syn_target = np.zeros((syn_num.shape[0], 1))
    
    return syn_num, syn_cat, syn_target

def recover_data(syn_num, syn_cat_df, syn_target, numerical_columns, target_column=None):
    """
    Reconstructs a DataFrame from synthesized numerical, categorical, and target data.
    
    Args:
        syn_num: Numerical data
        syn_cat_df: Categorical data (as DataFrame)
        syn_target: Target data
        numerical_columns: List of numerical column names
        target_column: Name of target column if any
        
    Returns:
        DataFrame: Recovered synthetic data
    """
    import pandas as pd
    import numpy as np
    
    # Create DataFrames for each component
    num_df = pd.DataFrame(syn_num, columns=[col for col in numerical_columns if col != target_column])
    
    # Target DataFrame (if applicable)
    if target_column:
        # Kopie erstellen um Warnung zu vermeiden
        target_values = syn_target.copy()
        
        # Runden
        target_values = np.round(target_values)
        
        # Auf Bereich begrenzen falls angegeben
        min_val, max_val = (0, 4) # impact 0 bis 4
        target_values = np.clip(target_values, min_val, max_val)
            
        # Zu Integer konvertieren
        target_values = target_values.astype(int).astype(str)
            
        target_df = pd.DataFrame(target_values, columns=[target_column])
        # Combine numerical and target
        result_df = pd.concat([target_df, num_df], axis=1)
    else:
        result_df = num_df

    # Konvertiere alle Spalten in syn_cat_df zu boolean 
    # (da alle Spalten in syn_cat_df One-Hot-Encoding-Spalten sind)
    for col in syn_cat_df.columns:
        syn_cat_df[col] = (syn_cat_df[col] > 0.5).astype(bool)
    
    # Combine with categorical data
    result_df = pd.concat([result_df, syn_cat_df], axis=1)
    
    return result_df
    

def process_invalid_id(syn_cat, min_cat, max_cat):
    syn_cat = np.clip(syn_cat, min_cat, max_cat)

    return syn_cat

