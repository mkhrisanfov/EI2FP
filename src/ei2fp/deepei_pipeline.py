import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ei2fp.datasets import get_train_val_test_datasets
from ei2fp import BASE_DIR, FPS_MASK, MACCS_MASK
from ei2fp.models import DEEPEI
from ei2fp.datasets import EI2FPDataset


def train(
    device,
    model,
    optim,
    crit,
    epoch_end,
    train_dl,
    val_dl,
    test_dl,
    name,
    fps_num=None,
    maccs_num=None,
):
    torch.cuda.empty_cache()
    for epoch in range(epoch_end):
        model.train()
        for maccs, fps, spectra in train_dl:
            optim.zero_grad()
            pred = model(spectra.to(device, non_blocking=True))
            if fps_num:
                loss = crit(pred, fps[:, fps_num].to(device, non_blocking=True)).mean()
            elif maccs_num:
                loss = crit(
                    pred, maccs[:, maccs_num].to(device, non_blocking=True)
                ).mean()
            loss.backward()  # pyright: ignore[reportPossiblyUnboundVariable]
            optim.step()
    torch.save(model.state_dict(), BASE_DIR / f"models/deepei/{name}_.pth")

    model.eval()
    with torch.no_grad():
        all_preds = []
        for _, _, spectra in test_dl:
            preds = model(spectra.to(device, non_blocking=True))
            all_preds.extend(preds.cpu())
    return all_preds


if __name__ == "__main__":

    lr = 1e-3
    batch_size = 32

    trn_ds, val_ds, tst_ds = get_train_val_test_datasets(
        BASE_DIR / "data/input/input_lib.ms", EI2FPDataset, spectra_len=2000
    )
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
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    crit = nn.BCELoss()
    fps_preds = []
    for fp_num in tqdm(FPS_MASK):
        model = DEEPEI().to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        name = f"DEEPEI_FP_{fp_num}"
        fps_preds.append(
            train(
                device,
                model,
                optim,
                crit,
                8,
                trn_dl,
                val_dl,
                tst_dl,
                name,
                fps_num=fp_num,
            )
        )
        break
    fps_preds = np.hstack(fps_preds)
    np.savetxt(BASE_DIR / "data/output/TST_DEEPEI_fp_preds.txt", fps_preds)

    maccs_preds = []
    for maccs_num in tqdm(MACCS_MASK):
        model = DEEPEI().to(device)
        optim = torch.optim.AdamW(model.parameters(), lr=lr)
        name = f"DEEPEI_MACCS_{maccs_num}"
        maccs_preds.append(
            train(
                device,
                model,
                optim,
                crit,
                8,
                trn_dl,
                val_dl,
                tst_dl,
                name,
                maccs_num=maccs_num,
            )
        )
        break
    maccs_preds = np.hstack(maccs_preds)
    np.savetxt(BASE_DIR / "data/output/TST_DEEPEI_maccs_preds.txt", maccs_preds)
    print("Finished")
