from pathlib import Path

import torch
import torch.nn as nn
import typer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing_extensions import Annotated

from ei2fp import BASE_DIR
from ei2fp.datasets import EI2FPDataset, get_train_test_datasets
from ei2fp.models import EI2FPFull

app = typer.Typer()


def train(model, optim, crit, epochs, train_dl, test_dl, name):
    torch.cuda.empty_cache()
    pbar = tqdm(range(0, epochs), position=0)
    b_trn = 10
    b_tst = 10
    writer = SummaryWriter(log_dir=BASE_DIR / f"logs/{name}", flush_secs=15)
    for epoch in pbar:
        train_loss = model.train_fn(optim, crit, train_dl)
        b_trn = min(b_trn, train_loss)

        test_loss = model.eval_fn(crit, test_dl)
        b_tst = min(b_tst, test_loss)

        pbar.set_postfix_str(
            f"{train_loss:.3e}/{b_trn:.3e}  {test_loss:.3e}/{b_tst:.3e}"
        )
        writer.add_scalar("loss/trn", train_loss, epoch)
        writer.add_scalar("loss/tst", test_loss, epoch)
        if test_loss <= b_tst:
            torch.save(model.state_dict(), BASE_DIR / f"models/{name}.pth")
    pbar.close()
    return b_trn, b_tst


@app.command()
def main(
    input_filename: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ],
    batch_size: Annotated[int, typer.Argument()] = 512,
):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        typer.echo("Using CUDA")
    else:
        device = torch.device("cpu")
        typer.echo("Using CPU")

    seed = 42
    lr = 1e-3
    batch_size = 512

    name = f"EI2FPFull_{seed}_{lr:.2e}_{batch_size}"
    model = EI2FPFull().to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.BCELoss()

    trn_ds, tst_ds = get_train_test_datasets(input_filename, EI2FPDataset)
    trn_dl = DataLoader(
        trn_ds,
        batch_size,
        pin_memory=True,
        shuffle=True,
        num_workers=4,
        persistent_workers=True,
    )
    tst_dl = DataLoader(
        tst_ds,
        batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=4,
        persistent_workers=True,
    )
    typer.echo("Start training")
    b_train, b_test = train(model, optim, crit, 200, trn_dl, tst_dl, name)
    typer.echo("Training finished")
    typer.echo(f"Training loss:   {b_train:.3e}")
    typer.echo(f"Validation loss: {b_test:.3e}")


if __name__ == "__main__":
    app()
