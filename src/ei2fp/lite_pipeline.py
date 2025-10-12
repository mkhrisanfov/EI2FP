import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ei2fp import BASE_DIR, FPS_MASK, MACCS_MASK
from ei2fp.datasets import EI2FPDataset, get_train_val_test_datasets
from ei2fp.models import EI2FPLite


def train(device, model, optim, crit, epoch_end, train_dl, val_dl, name):
    torch.cuda.empty_cache()
    b_trn = 10
    b_tst = 10
    # print("Start")
    writer = SummaryWriter(log_dir=BASE_DIR / f"logs/{name}", flush_secs=15)
    for epoch in tqdm(range(epoch_end)):
        model.train()
        train_loss = []
        for maccs, fps, spectra in train_dl:
            optim.zero_grad()
            pred_maccs, pred_fps = model(spectra.to(device, non_blocking=True))
            loss1 = crit(pred_maccs, maccs.to(device, non_blocking=True)).mean()
            loss2 = crit(pred_fps, fps.to(device, non_blocking=True)).mean()
            loss = (167 * loss1 + 1024 * loss2) / (167 + 1024)
            loss.backward()
            optim.step()
            train_loss.append(loss.detach())
        train_loss = torch.stack([x.cpu() for x in train_loss]).mean()
        b_trn = min(b_trn, train_loss)

        model.eval()
        val_loss = []
        with torch.no_grad():
            for maccs, fps, spectra in val_dl:
                pred_maccs, pred_fps = model(spectra.to(device, non_blocking=True))
                loss1 = crit(pred_maccs, maccs.to(device, non_blocking=True))
                loss2 = crit(pred_fps, fps.to(device, non_blocking=True))
                loss = (167 * loss1 + 1024 * loss2) / (167 + 1024)
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
            torch.save(model.state_dict(), BASE_DIR / f"models/{name}_model.pth")

    torch.save(model.state_dict(), BASE_DIR / f"models/{name}_model_final.pth")
    return b_trn, b_tst


if __name__ == "__main__":
    seed = 47

    lr = 1e-3
    batch_size = 512

    name = f"Lite_{seed}_{lr:.2e}_{batch_size}"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    model = EI2FPLite().to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCELoss()

    trn_ds, val_ds, tst_ds = get_train_val_test_datasets(
        BASE_DIR / "data/input/input_lib.ms", EI2FPDataset
    )
    np.savetxt(
        BASE_DIR / f"data/output/TST_{seed}_maccs.txt",
        tst_ds.maccs.numpy()[:, MACCS_MASK],
    )
    np.savetxt(
        BASE_DIR / f"data/output/TST_{seed}_fps.txt", tst_ds.fps.numpy()[:, FPS_MASK]
    )
    np.savetxt(
        BASE_DIR / f"data/output/VAL_{seed}_maccs.txt",
        val_ds.maccs.numpy()[:, MACCS_MASK],
    )
    np.savetxt(
        BASE_DIR / f"data/output/VAL_{seed}_fps.txt", val_ds.fps.numpy()[:, FPS_MASK]
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
    print("Starte training")

    b_train, b_test = train(device, model, optim, crit, 100, trn_dl, val_dl, name)
    print("Finished training")
