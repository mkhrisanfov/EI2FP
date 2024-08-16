# %%
import logging
import os

import numpy as np
import pandas as pd

# import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from sklearn.model_selection import train_test_split
from torchinfo import summary
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit import Chem
import IsoSpecPy as iso
import matplotlib.pyplot as plt
from functions import *

from torch.utils.tensorboard import SummaryWriter

# %load_ext tensorboard
torch.backends.cudnn.benchmark = True
# %%
FPS_MASK = [1,    4,   13,   15,   33,   36,   64,   80,  114,  119,  128,
            147,  175,  225,  250,  283,  293,  294,  301,  314,  322,  356,
            361,  362,  378,  389,  420,  512,  540,  561,  579,  591,  636,
            642,  650,  656,  659,  694,  695,  698,  710,  725,  726,  730,
            739,  784,  794,  807,  831,  841,  849,  875,  881,  887,  893,
            904,  923,  926,  935,  946, 1017, 1019]
MACCS_MASK = [42,  57,  62,  65,  66,  72,  74,  75,  77,  78,  79,  80,  81,
              83,  85,  86,  87,  88,  89,  90,  91,  92,  93,  94,  95,  96,
              97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
              110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122,
              123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135,
              136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146, 147, 148,
              149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161,
              162, 163, 164, 165]
# %%


class DEEPEI(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_first = nn.Sequential(
            nn.Linear(2000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000),
            nn.ReLU(),
            nn.Linear(1000, 500),
            nn.ReLU(),
        )

        self.fc_last = nn.Sequential(
            nn.Linear(500, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc_first(x)
        return self.fc_last(x).squeeze()


class EI_MIX_Dataset(Dataset):

    def __init__(self, smis, spectra):
        spectra = np.vstack(spectra) / 1000
        self.spectra = torch.FloatTensor(spectra)
        self.maccs = torch.FloatTensor(
            np.vstack(process_map(get_maccs, smis,
                      max_workers=4, chunksize=1000))
        )
        self.fps = torch.clip(torch.FloatTensor(
            np.vstack(process_map(get_fp, smis, max_workers=4, chunksize=1000))
        ), 0, 1)

    def __getitem__(self, index):
        return (self.maccs[index], self.fps[index], self.spectra[index])

    def __len__(self):
        return len(self.spectra)

# %%


def train(device, model, optim, crit, epoch_end, train_dl, val_dl, test_dl, name, fps_num=None, maccs_num=None):
    torch.cuda.empty_cache()
    for epoch in range(epoch_end):
        model.train()
        for maccs, fps, spectra in train_dl:
            optim.zero_grad()
            pred = model(spectra.to(device, non_blocking=True))
            if fps_num:
                loss = crit(pred, fps[:, fps_num].to(
                    device, non_blocking=True)).mean()
            elif maccs_num:
                loss = crit(pred, maccs[:, maccs_num].to(
                    device, non_blocking=True)).mean()
            loss.backward()
            optim.step()
    torch.save(model.state_dict(), f"../Models/DEEPEI_T/{name}_.pth")

    model.eval()
    with torch.no_grad():
        all_preds = []
        for _, _, spectra in test_dl:
            preds = model(spectra.to(device, non_blocking=True))
            all_preds.extend(preds.cpu())
    return all_preds


# %%
if __name__ == "__main__":

    lr = 1e-3
    batch_size = 32

    trn_ds, val_ds, tst_ds = get_train_test_datasets("../Data/In/mainlib.ms")
    trn_dl = DataLoader(
        trn_ds,
        batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size, pin_memory=True, num_workers=4, persistent_workers=True
    )
    tst_dl = DataLoader(
        tst_ds, batch_size, pin_memory=True, num_workers=4, persistent_workers=True
    )

    device = torch.device("cuda")
    crit = nn.BCELoss()
    fps_preds = []
    for fp_num in tqdm(FPS_MASK):
        model = DEEPEI().to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        name = f"DEEPEI_FP_{fp_num}"
        fps_preds.append(train(device, model, optim, crit, 8,
                         trn_dl, val_dl, tst_dl, name, fps_num=fp_num))
    fps_preds = np.hstack(fps_preds)
    np.savetxt("../Data/Out/TST_DEEPEI_fp_preds.txt", fps_preds)

    maccs_preds = []
    for maccs_num in tqdm(MACCS_MASK):
        model = DEEPEI().to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        name = f"DEEPEI_MACCS_{maccs_num}"
        maccs_preds.append(train(device, model, optim, crit, 8,
                           trn_dl, val_dl, tst_dl, name, maccs_num=maccs_num))
    maccs_preds = np.hstack(maccs_preds)
    np.savetxt("../Data/Out/TST_DEEPEI_maccs_preds.txt", maccs_preds)
