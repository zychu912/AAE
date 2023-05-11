"""
Author: Zhiyuan Chu
Date: March 27, 2023
File Name: dataset.py
"""

import anndata
import numpy as np
import os
import scipy
import torch
import pytorch_lightning as pl
import scanpy as sc
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import DataLoader, Dataset
from typing import Optional

def load_file(name: str, data_path: str = ''):
    """
    Load an h5ad gzip file.
    """
    if not os.path.exists(data_path + name + '.h5ad.gz'):
        raise ValueError(f"File {name} not found.")
    
    adata = sc.read_h5ad(data_path + name + '.h5ad.gz')
    if not scipy.sparse.issparse(adata.X):
        adata.X = scipy.sparse.csr_matrix(adata.X)
    adata.var_names_make_unique()
    return adata

def preprocessing_adata(adata, n_top_genes: int = 3000):
    sc.pp.filter_genes(adata, min_counts=3)
    adata.layers["counts"] = adata.X.copy()  # preserve counts
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.raw = adata  # freeze the state in `.raw`
    sc.pp.highly_variable_genes(adata, flavor='seurat_v3', n_top_genes=n_top_genes, subset=True, layer='counts')
    
class scDataset(Dataset):
    '''
    Single cell dataset for PyTorch dataloader.
    '''
    def __init__(self, adata: anndata.AnnData):
        self.adata = adata
        self.data = torch.from_numpy(adata.X.todense()).float() # convert to PyTorch tensor
        self.labels = torch.from_numpy(np.array(adata.obs['doublet_label'].values.map({ 'doublet': 0, 'singlet': 1, 'generated': 2}))).long() # convert to PyTorch tensor of long type

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ids):
        out = []
        for i in ids:
            x = self.data[i]
            y = self.labels[i]
            out.append((x, y))
        return out
    
class KFoldAdata(pl.LightningDataModule):
    """
    PyTorch Lightning data module for k-fold cross validation with anndata objects.
    """
    def __init__(self, scdata, seed=1, k=5, batch_size=64, num_workers=1, pin_memory=False):
        """
        Initializes the KFoldAdata class.
        """
        super(KFoldAdata, self).__init__()
        self.save_hyperparameters()
        self.data_train = None
        self.data_test = None
        self.prepare(scdata)
    
    def prepare(self, scdata):
        """
        Performs k-fold cross-validation on the input data and stores the splits.
        """
        if not self.data_train and not self.data_test:
            self.scdata = scdata
            kf = KFold(n_splits=self.hparams.k, shuffle=True, random_state=self.hparams.seed)
            self.all_splits = [k for k in kf.split(self.scdata)]
            
    def train_dataloader(self, split_id):
        """
        Returns a PyTorch DataLoader for the training data in the specified fold.
        """
        assert 0 <= split_id <= self.hparams.k, "Inappropriate fold number"
        train_index = self.all_splits[split_id][0]
        self.data_train = self.scdata[train_index]
        # Print data info
        print('Training data info:')
        print(self.scdata.adata[train_index].obs['doublet_label'].value_counts())
        print('\n')
        return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, 
                          shuffle=True)
    
    def test_dataloader(self, split_id):
        """
        Returns a PyTorch DataLoader for the test data in the specified fold.
        """
        assert 0 <= split_id <= self.hparams.k, "Inappropriate fold number"
        test_index = self.all_splits[split_id][1]
        self.data_test = self.scdata[test_index]
        # Print data info
        print("test data info:")
        print(self.scdata.adata[test_index].obs['doublet_label'].value_counts())
        print("\n")
        return DataLoader(dataset=self.data_test, batch_size=self.hparams.batch_size, 
                          num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory,
                          shuffle=False)
    
# class KFoldAdata(pl.LightningDataModule):
#     """
#     PyTorch Lightning data module for k-fold cross validation with anndata objects.
#     """
#     def __init__(self, seed=1, k=5, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, pin_memory=False):
#         """
#         Initializes the KFoldAdata class.
#         """
#         super(KFoldAdata, self).__init__()
#         self.save_hyperparameters()
#         self.data_train = None
#         self.data_val = None
#         self.data_test = None
#         self.train_val_adata = None
#         self.test_adata = None
#         # remember to test pin_memory=True

#     def prepare(self, adata):
#         """
#         Performs k-fold cross-validation on the input data and stores the splits.
#         """
#         if not self.data_train and not self.data_val and not self.data_test:
#             self.train_val_adata, self.test_adata = train_test_split(adata, test_size=0.2, random_state=self.hparams.seed)
#             self.train_val_data = scDataset(self.train_val_adata)
#             self.test_data = scDataset(self.test_adata)
#             kf = KFold(n_splits=self.hparams.k, shuffle=True, random_state=self.hparams.seed)
#             self.all_splits = [k for k in kf.split(self.train_val_adata)]
    
#     def train_dataloader(self, split_id):
#         """
#         Returns a PyTorch DataLoader for the training data in the specified fold.
#         """
#         assert 0 <= split_id <= self.hparams.k, "Inappropriate fold number"
#         train_index = self.all_splits[split_id][0]
#         self.data_train = scDataset(self.train_val_adata[train_index])
#         train_label = self.train_val_adata.obs['doublet_label'][train_index]
#         # Print data info
#         print('Training data info:')
#         print(train_label.value_counts())
#         print('\n')
#         return DataLoader(dataset=self.data_train, batch_size=self.hparams.batch_size, 
#                           num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, 
#                           shuffle=True)

#     def val_dataloader(self, split_id):
#         """
#         Returns a PyTorch DataLoader for the validation data in the specified fold.
#         """
#         assert 0 <= split_id <= self.hparams.k, "Inappropriate fold number"
#         val_index = self.all_splits[split_id][1]
#         self.data_val = scDataset(self.train_val_adata[val_index])
#         val_label = self.train_val_adata.obs['doublet_label'][val_index]
#         # Print data info
#         print("val data info:")
#         print(val_label.value_counts())
#         print("\n")
#         return DataLoader(dataset=self.data_val, batch_size=self.hparams.batch_size, 
#                           num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, 
#                           shuffle=False)
    
#     def test_dataloader(self):
#         """
#         Returns a PyTorch DataLoader for the test data in the specified fold.
#         """
#         self.data_test = scDataset(self.test_adata)
#         test_label = self.test_adata.obs['doublet_label']
#         # Print data info
#         print("test data info:")
#         print(test_label.value_counts())
#         print("\n")
#         return DataLoader(dataset=self.data_test, batch_size=self.hparams.batch_size, 
#                           num_workers=self.hparams.num_workers, pin_memory=self.hparams.pin_memory, 
#                           shuffle=False)