import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings

import os
from tqdm import tqdm
import json
import time
import pandas as pd

from tabsyn.vae.model import Model_VAE, Encoder_model, Decoder_model
from utils_train import preprocess, TabularDataset

warnings.filterwarnings('ignore')


LR = 1e-3
WD = 0
D_TOKEN = 4
TOKEN_BIAS = True

N_HEAD = 1
FACTOR = 32
NUM_LAYERS = 2


def compute_loss(X_num, X_cat, Recon_X_num, Recon_X_cat, mu_z, logvar_z):
    ce_loss_fn = nn.CrossEntropyLoss()
    mse_loss = (X_num - Recon_X_num).pow(2).mean()
    ce_loss = 0
    acc = 0
    total_num = 0

    for idx, x_cat in enumerate(Recon_X_cat):
        if x_cat is not None:
            ce_loss += ce_loss_fn(x_cat, X_cat[:, idx])
            x_hat = x_cat.argmax(dim = -1)
        acc += (x_hat == X_cat[:,idx]).float().sum()
        total_num += x_hat.shape[0]
    
    ce_loss /= (idx + 1)
    acc /= total_num
    # loss = mse_loss + ce_loss

    temp = 1 + logvar_z - mu_z.pow(2) - logvar_z.exp()

    loss_kld = -0.5 * torch.mean(temp.mean(-1).mean())
    return mse_loss, ce_loss, loss_kld, acc


def transform_preprocessed_data(train_df, test_df):
    """
    Load preprocessed CSV files and format them to match the output of the preprocess function.
    
    Args:
        task_type (str): Type of task ('binclass', 'multiclass', or 'regression')
        
    Returns:
        tuple: (X_num, X_cat, categories, d_numerical) where:
            - X_num is a tuple of (train_numerical_features, test_numerical_features)
            - X_cat is a tuple of (train_categorical_features, test_categorical_features)
            - categories is a list of sizes for each categorical variable
            - d_numerical is the number of numerical features
    """
    # Remove excluded columns only (keep the target 'impact')
    if 'combined_tks' in train_df.columns:
        train_df = train_df.drop(columns=['combined_tks', 'id'])
        test_df = test_df.drop(columns=['combined_tks', 'id'])
    
     # Identify numerical and categorical columns
    all_columns = train_df.columns.tolist()
    
    # Group categorical columns by their prefix to identify the categorical features
    category_prefixes = ['category_', 'sub_category1_', 'sub_category2_', 'ticket_type_', 'business_service_']
    
    # Dictionary to store grouped categorical columns
    categorical_groups = {}
    
    # Group the categorical columns by their prefix
    for prefix in category_prefixes:
        cols = [col for col in all_columns if col.startswith(prefix)]
        if cols:
            prefix_clean = prefix.rstrip('_')  # Remove trailing underscore
            categorical_groups[prefix_clean] = sorted(cols)
    
    # All other columns except combined_tks are considered numerical
    num_cols = [col for col in all_columns if not any(col.startswith(prefix) for prefix in category_prefixes) 
                and col != 'combined_tks']
    
    # Extract numerical features
    X_train_num = train_df[num_cols].values
    X_test_num = test_df[num_cols].values
    
    # Process categorical features - convert one-hot encoding to indices
    X_train_cat_indices = []
    X_test_cat_indices = []
    categories = []
    
    for group_name, group_cols in categorical_groups.items():
        # Extract one-hot encoded columns for this group
        train_group = train_df[group_cols].values
        test_group = test_df[group_cols].values
        
        # Convert one-hot encoding to indices (argmax)
        # If a row is all zeros, we'll use the last category as default
        train_indices = np.argmax(train_group, axis=1)
        test_indices = np.argmax(test_group, axis=1)
        
        # Handle case where all values in a row are 0 (no category selected)
        # Set to a special index (num_categories)
        all_zeros_train = (train_group.sum(axis=1) == 0)
        all_zeros_test = (test_group.sum(axis=1) == 0)
        
        if np.any(all_zeros_train) or np.any(all_zeros_test):
            # Add an extra category for "none selected"
            num_categories = len(group_cols) + 1
            train_indices[all_zeros_train] = len(group_cols)
            test_indices[all_zeros_test] = len(group_cols)
        else:
            num_categories = len(group_cols)
        
        X_train_cat_indices.append(train_indices)
        X_test_cat_indices.append(test_indices)
        categories.append(num_categories)
    
    # Stack all categorical features into a single matrix
    X_train_cat = np.column_stack(X_train_cat_indices) if X_train_cat_indices else np.empty((len(X_train_num), 0), dtype=np.int64)
    X_test_cat = np.column_stack(X_test_cat_indices) if X_test_cat_indices else np.empty((len(X_test_num), 0), dtype=np.int64)
    
    # Number of numerical features
    d_numerical = X_train_num.shape[1]
    
    # Return in the same format as preprocess
    return (X_train_num, X_test_num), (X_train_cat, X_test_cat), categories, d_numerical


def main(args):
    max_beta = args.max_beta
    min_beta = args.min_beta
    lambd = args.lambd

    device =  args.device

    ckpt_dir = args.ckpt_dir #f'{curr_dir}/ckpt/{dataname}' 
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    model_save_path = f'{ckpt_dir}/model.pt'
    encoder_save_path = f'{ckpt_dir}/encoder.pt'
    decoder_save_path = f'{ckpt_dir}/decoder.pt'

    X_num, X_cat, categories, d_numerical = load_preprocessed_csv(args.datapath, args.datatestpath) #preprocess(data_dir, task_type = info['task_type'])

    X_train_num, X_test_num = X_num
    X_train_cat, X_test_cat = X_cat

    X_train_num, X_test_num = torch.tensor(X_train_num).float(), torch.tensor(X_test_num).float()
    X_train_cat, X_test_cat =  torch.tensor(X_train_cat), torch.tensor(X_test_cat)


    train_data = TabularDataset(X_train_num.float(), X_train_cat)

    X_test_num = X_test_num.float().to(device)
    X_test_cat = X_test_cat.to(device)

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    model = Model_VAE(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR, bias = True)
    model = model.to(device)

    # Add this right after model initialization in the main function
    if os.path.exists(model_save_path):
        print(f"Loading existing model checkpoint from {model_save_path}")
        model.load_state_dict(torch.load(model_save_path))
        # You might also want to evaluate the loaded model to get its initial performance
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)
            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()
            best_train_loss = val_loss  # Initialize best_train_loss with the loaded model's performance
        print(f"Loaded model with validation loss: {best_train_loss}")
    else:
        best_train_loss = float('inf')  # Initialize best_train_loss if no checkpoint exists

    pre_encoder = Encoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)
    pre_decoder = Decoder_model(NUM_LAYERS, d_numerical, categories, D_TOKEN, n_head = N_HEAD, factor = FACTOR).to(device)

    pre_encoder.eval()
    pre_decoder.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.95, patience=10, verbose=True)

    num_epochs = 1
    best_train_loss = float('inf')

    current_lr = optimizer.param_groups[0]['lr']
    patience = 0

    beta = max_beta
    start_time = time.time()
    for epoch in range(num_epochs):
        pbar = tqdm(train_loader, total=len(train_loader))
        pbar.set_description(f"Epoch {epoch+1}/{num_epochs}")

        curr_loss_multi = 0.0
        curr_loss_gauss = 0.0
        curr_loss_kl = 0.0

        curr_count = 0

        for batch_num, batch_cat in pbar:
            model.train()
            optimizer.zero_grad()

            batch_num = batch_num.to(device)
            batch_cat = batch_cat.to(device)

            Recon_X_num, Recon_X_cat, mu_z, std_z = model(batch_num, batch_cat)
        
            loss_mse, loss_ce, loss_kld, train_acc = compute_loss(batch_num, batch_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)

            loss = loss_mse + loss_ce + beta * loss_kld
            loss.backward()
            optimizer.step()

            batch_length = batch_num.shape[0]
            curr_count += batch_length
            curr_loss_multi += loss_ce.item() * batch_length
            curr_loss_gauss += loss_mse.item() * batch_length
            curr_loss_kl    += loss_kld.item() * batch_length

        num_loss = curr_loss_gauss / curr_count
        cat_loss = curr_loss_multi / curr_count
        kl_loss = curr_loss_kl / curr_count
        

        '''
            Evaluation
        '''
        model.eval()
        with torch.no_grad():
            Recon_X_num, Recon_X_cat, mu_z, std_z = model(X_test_num, X_test_cat)

            val_mse_loss, val_ce_loss, val_kl_loss, val_acc = compute_loss(X_test_num, X_test_cat, Recon_X_num, Recon_X_cat, mu_z, std_z)
            val_loss = val_mse_loss.item() * 0 + val_ce_loss.item()    

            scheduler.step(val_loss)
            new_lr = optimizer.param_groups[0]['lr']

            if new_lr != current_lr:
                current_lr = new_lr
                print(f"Learning rate updated: {current_lr}")
                
            train_loss = val_loss
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                patience = 0
                torch.save(model.state_dict(), model_save_path)
            else:
                patience += 1
                if patience == 10:
                    if beta > min_beta:
                        beta = beta * lambd


        # print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Train ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, train_acc.item()))
        print('epoch: {}, beta = {:.6f}, Train MSE: {:.6f}, Train CE:{:.6f}, Train KL:{:.6f}, Val MSE:{:.6f}, Val CE:{:.6f}, Train ACC:{:6f}, Val ACC:{:6f}'.format(epoch, beta, num_loss, cat_loss, kl_loss, val_mse_loss.item(), val_ce_loss.item(), train_acc.item(), val_acc.item() ))

    end_time = time.time()
    print('Training time: {:.4f} mins'.format((end_time - start_time)/60))
    
    # Saving latent embeddings
    with torch.no_grad():
        pre_encoder.load_weights(model)
        pre_decoder.load_weights(model)

        torch.save(pre_encoder.state_dict(), encoder_save_path)
        torch.save(pre_decoder.state_dict(), decoder_save_path)

        X_train_num = X_train_num.to(device)
        X_train_cat = X_train_cat.to(device)

        print('Successfully load and save the model!')

        train_z = pre_encoder(X_train_num, X_train_cat).detach().cpu().numpy()

        np.save(f'{ckpt_dir}/train_z.npy', train_z)

        print('Successfully save pretrained embeddings in disk!')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Variational Autoencoder')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--max_beta', type=float, default=1e-2, help='Initial Beta.')
    parser.add_argument('--min_beta', type=float, default=1e-5, help='Minimum Beta.')
    parser.add_argument('--lambd', type=float, default=0.7, help='Decay of Beta.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'