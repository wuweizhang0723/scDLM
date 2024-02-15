import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader


def load_data2(
    batch_size=256,
    num_workers=12,
    persistent_workers=False,
    RC_mode=False, #########
):
    """Load data from file."""
    X_train = np.load('./data/X_train.npy')
    Y_train = np.load('./data/Y_train.npy')
    train_dataset = TensorDataset(torch.tensor(X_train).float(), torch.tensor(Y_train).long())
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    X_val = np.load('./data/X_val.npy')
    Y_val = np.load('./data/Y_val.npy')
    val_dataset = TensorDataset(torch.tensor(X_val).float(), torch.tensor(Y_val).long())
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    X_test = np.load('./data/X_test.npy')
    Y_test = np.load('./data/Y_test.npy')
    test_dataset = TensorDataset(torch.tensor(X_test).float(), torch.tensor(Y_test).long())
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        persistent_workers=persistent_workers
    )

    return train_loader, val_loader, test_loader