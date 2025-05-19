import torch

import argparse
import warnings
import time

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample
from utils import load_df
import pandas as pd

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path
    num_samples = args.num_samples
    max_iterations = args.max_iterations
    # New parameters for conditional sampling
    conditions = args.conditions    

    train_z, pre_decoder, token_dim, num_inverse, cat_inverse, df_columns, column_metadata = get_input_generate(args)
    in_dim = train_z.shape[1] 
    
    mean = train_z.mean(0)
    
    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)
    
    model.load_state_dict(torch.load(f'{args.ckpt_dir}/model.pt')['model_state_dict'])

    '''
        Generating samples    
    '''
    start_time = time.time()

    sample_dim = in_dim
    # Dictionary to track samples for each condition
    condition_samples = {i: [] for i in range(len(conditions))}
    
    # Track how many samples we've collected for each condition
    samples_collected = {i: 0 for i in range(len(conditions))}
    
    iterations = 0
    all_conditions_met = False
    
    while not all_conditions_met and iterations < max_iterations:
        # Generate a batch of samples
        x_next = sample(model.denoise_fn_D, num_samples, sample_dim, device=device)
        x_next = x_next * 2 + mean.to(device)
        
        # Convert to numpy and process
        syn_data = x_next.float().cpu().numpy()
        syn_num, syn_cat_df, syn_target = split_num_cat_target(
            syn_data, pre_decoder, token_dim, column_metadata, num_inverse, cat_inverse, device
        )
        
        # Reconstruct the dataframe
        target_column = column_metadata.get('target_column')
        numerical_columns = column_metadata['numerical_columns']
        syn_df = recover_data(syn_num, syn_cat_df, syn_target, numerical_columns, target_column)
        
        # For each condition, find matching samples
        for i, condition_dict in enumerate(conditions):
            # Skip if we already have enough samples for this condition
            if samples_collected[i] >= num_samples:
                continue
                
            # Find rows that match this condition
            condition_mask = pd.Series(True, index=syn_df.index)
            for column_name, value in condition_dict.items():
                if column_name in syn_df.columns:
                    condition_mask &= (syn_df[column_name] == value)
                else:
                    print(f"Warning: Column '{column_name}' not found in generated data")
                    condition_mask &= False  # No rows can match if column doesn't exist
            
            matching_rows = syn_df[condition_mask]
            
            if len(matching_rows) > 0:
                # How many more samples do we need for this condition?
                samples_needed = num_samples - samples_collected[i]
                samples_to_add = matching_rows.head(samples_needed)
                
                # Add metadata to track which condition these samples satisfy
                samples_to_add = samples_to_add.copy()
                # Optionally add a column to track which condition these samples satisfy
                condition_str = ', '.join([f"{k}={v}" for k, v in condition_dict.items()])
                samples_to_add['condition_source'] = condition_str
                
                condition_samples[i].append(samples_to_add)
                samples_collected[i] += len(samples_to_add)
                
                print(f"Iteration {iterations+1}: Added {len(samples_to_add)} samples for condition: {condition_dict}")
        
        # Check if we have enough samples for all conditions
        all_conditions_met = all(
            count >= num_samples for count in samples_collected.values()
        )
        
        iterations += 1
        
        # Print progress summary
        print(f"Iteration {iterations}/{max_iterations} - Progress:")
        for i, condition_dict in enumerate(conditions):
            condition_str = ', '.join([f"{k}={v}" for k, v in condition_dict.items()])
            print(f"  Condition {i+1} ({condition_str}): {samples_collected[i]}/{num_samples}")
    
    # Combine all samples into a single DataFrame
    all_samples = []
    for i, samples_list in condition_samples.items():
        if samples_list:
            condition_df = pd.concat(samples_list)
            all_samples.append(condition_df)
            
            # Print summary for this condition
            condition_dict = conditions[i]
            condition_str = ', '.join([f"{k}={v}" for k, v in condition_dict.items()])
            print(f"Collected {len(condition_df)} samples for condition: {condition_str}")
    
    if all_samples:
        combined_df = pd.concat(all_samples)
        print(f"Total samples in combined DataFrame: {len(combined_df)}")
        print('Saving sampled data to {}'.format(save_path))
        combined_df.to_csv(save_path, index = False)
    else:
        print("No samples were collected for any condition")
    
    end_time = time.time()
    print('Time:', end_time - start_time)
    

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'