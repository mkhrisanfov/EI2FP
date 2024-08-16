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
seed = 47


class Lite(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc_first = nn.Sequential(
            nn.Linear(750, 2048),
            nn.SiLU(),
            nn.Linear(2048, 1024),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.BatchNorm1d(1024),
        )

        self.fc_maccs = nn.Sequential(
            nn.Linear(1024, 167),
            nn.Sigmoid(),
        )
        self.fc_fps = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc_first(x)
        maccs = self.fc_maccs(x)
        fps = self.fc_fps(x)
        return (maccs, fps)


class Lite_Dataset(Dataset):

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


def train(device, model, optim, crit, epoch_end, train_dl, val_dl, name):
    torch.cuda.empty_cache()
    b_trn = 10
    b_tst = 10
    # print("Start")
    writer = SummaryWriter(log_dir=f"../Logs/{name}", flush_secs=15)
    for epoch in tqdm(range(epoch_end)):
        model.train()
        train_loss = []
        for maccs, fps, spectra in train_dl:
            optim.zero_grad()
            pred_maccs, pred_fps = model(spectra.to(device, non_blocking=True))
            loss1 = crit(pred_maccs, maccs.to(
                device, non_blocking=True)).mean()
            loss2 = crit(pred_fps, fps.to(device, non_blocking=True)).mean()
            loss = (167*loss1+1024*loss2)/(167+1024)
            loss.backward()
            optim.step()
            train_loss.append(loss.detach())
        train_loss = torch.stack([x.cpu() for x in train_loss]).mean()
        b_trn = min(b_trn, train_loss)

        model.eval()
        val_loss = []
        with torch.no_grad():
            for maccs, fps, spectra in val_dl:
                pred_maccs, pred_fps = model(
                    spectra.to(device, non_blocking=True))
                loss1 = crit(pred_maccs, maccs.to(device, non_blocking=True))
                loss2 = crit(pred_fps, fps.to(device, non_blocking=True))
                loss = (167*loss1+1024*loss2)/(167+1024)
                val_loss.append(loss.detach().mean())
        val_loss = torch.stack([x.cpu() for x in val_loss]).mean()
        b_tst = min(b_tst, val_loss)

        logging.info(
            f"[{epoch+1}/{epoch_end}]:  trn: {train_loss:.4e}\ttst: {val_loss:.4e}"
        )
        writer.add_scalar("loss/trn", train_loss, epoch)
        writer.add_scalar("loss/tst", val_loss, epoch)
        # writer.add_scalar("loss/val",val_loss,epoch)
        if val_loss <= b_tst:
            torch.save(model.state_dict(), f"../Models/{name}_model.pth")

    torch.save(model.state_dict(), f"../Models/{name}_model_final.pth")
    return b_trn, b_tst


# %%
if __name__ == "__main__":

    lr = 1e-3
    batch_size = 512

    name = f"Lite_1.24_TST_{seed}_{lr:.2e}_{batch_size}"

    device = torch.device("cuda")
    model = Lite().to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCELoss()

    logging.basicConfig(
        filename=f"../Logs/{name}.log", encoding="utf-8", level=logging.DEBUG
    )
    logging.info(f"Model: {model}")
    logging.info(f"Params: {optim}")

    trn_ds, val_ds, tst_ds = get_train_test_datasets("../Data/In/mainlib.ms")
    np.savetxt(f"../Data/Out/TST_{seed}_maccs.txt",
               tst_ds.maccs.numpy()[:, MACCS_MASK])
    np.savetxt(f"../Data/Out/TST_{seed}_fps.txt",
               tst_ds.fps.numpy()[:, FPS_MASK])
    np.savetxt(f"../Data/Out/VAL_{seed}_maccs.txt",
               val_ds.maccs.numpy()[:, MACCS_MASK])
    np.savetxt(f"../Data/Out/VAL_{seed}_fps.txt",
               val_ds.fps.numpy()[:, FPS_MASK])
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

    b_train, b_test = train(device, model, optim, crit,
                            100, trn_dl, val_dl, name)
