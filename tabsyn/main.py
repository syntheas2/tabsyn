# import __init__
import os
import torch

from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import argparse
import warnings
import time

from tqdm import tqdm
from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_train

warnings.filterwarnings('ignore')


def main(args): 
    device = args.device

    train_z, _, _, ckpt_path = get_input_train(args)
    model_save_path = f'{ckpt_path}/model.pt'

    print(ckpt_path)

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    in_dim = train_z.shape[1] 

    # normalisierung um 0 -> für stabilere gradienten (mehr in readme)
    mean, std = train_z.mean(0), train_z.std(0)
    train_z = (train_z - mean) / 2
    train_data = train_z

    batch_size = 4096
    train_loader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle = True,
        num_workers = 4,
    )

    num_epochs = 10000 + 1

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    print(denoise_fn)

    num_params = sum(p.numel() for p in denoise_fn.parameters())
    print("the number of parameters", num_params)

    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0)

    if os.path.exists(model_save_path):
        print(f"Loading existing model checkpoint from {model_save_path}")

        # Load the checkpoint
        checkpoint = torch.load(model_save_path)

        # Load the model state
        model.load_state_dict(checkpoint['model_state_dict'])

        # Load optimizer state if it exists and if you have an optimizer defined
        if 'optimizer_state_dict' in checkpoint and 'optimizer' in locals():
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Get the saved learning rate
        saved_lr = checkpoint.get('learning_rate', None)
        if saved_lr is not None:
            print(f"Loaded learning rate: {saved_lr}")
            # Update the learning rate in the optimizer if needed
            for param_group in optimizer.param_groups:
                param_group['lr'] = saved_lr

        # Get the saved best loss
        if 'best_loss' in checkpoint:
            best_loss = checkpoint['best_loss']
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

        print(f"Loaded model with validation loss: {best_loss}")
    else:
        best_loss = float('inf')  # Initialize best_loss if no checkpoint exists



    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=20, verbose=True)

    model.train()

    patience = 0
    current_lr = optimizer.param_groups[0]['lr']
    
    start_time = time.time()
    for epoch in range(num_epochs):
        
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

        if new_lr != current_lr:
            current_lr = new_lr
            print(f"Learning rate updated: {current_lr}")

        if curr_loss < best_loss:
            best_loss = curr_loss
            pbar.set_postfix({"Best Loss": best_loss})
            # Force refresh the progress bar to show the update
            pbar.update(0)
            patience = 0
            # Create a checkpoint dictionary with model state and learning rate
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),  # Save the optimizer state
                'best_loss': best_loss,
                'learning_rate': optimizer.param_groups[0]['lr']  # Save the current learning rate
            }

            # Save the complete checkpoint
            torch.save(checkpoint, model_save_path)

        else:
            patience += 1
            if patience == 500:
                print('Early stopping')
                break

        if epoch % 1000 == 0:
            torch.save(model.state_dict(), f'{ckpt_path}/model_{epoch}.pt')

    end_time = time.time()
    print('Time: ', end_time - start_time)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training of TabSyn')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'