from pathlib import Path

import torch
import typer
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing_extensions import Annotated

from ei2fp import BASE_DIR
from ei2fp.datasets import get_inference_dataset
from ei2fp.models import EI2FPFull

app = typer.Typer()


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
    output_filename: Annotated[
        Path,
        typer.Argument(
            exists=False,
            resolve_path=True,
        ),
    ],
    id_field: Annotated[
        str,
        typer.Argument(),
    ] = "name",
    model_weights: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            resolve_path=True,
        ),
    ] = BASE_DIR
    / "models/EI2FPFull.pth",
    batch_size: Annotated[int, typer.Argument()] = 64,
    use_cuda: Annotated[bool, typer.Option("--use_cuda")] = False,
):
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
        typer.echo("Using CUDA")
    else:
        device = torch.device("cpu")
        typer.echo("Using CPU")

    typer.echo("Loading dataset")
    names, fw_dataset = get_inference_dataset(input_filename, id_field=id_field)
    typer.echo(f"Molecules loaded: {len(names)}")
    typer.echo(f"Valid molecules: {len(fw_dataset)}")
    fw_dataloader = DataLoader(
        fw_dataset,
        batch_size,
        num_workers=4 if use_cuda else False,
        pin_memory=True if use_cuda else False,
        shuffle=False,
    )

    typer.echo("Loading EI2FP-Full model")
    model = EI2FPFull().to(device)
    model.load_state_dict(
        torch.load(
            model_weights,
            map_location=device,
            weights_only=True,
        )
    )

    typer.echo("Begin prediction")
    pred_maccs, pred_fps = model.predict(fw_dataloader)
    pred_maccs = (pred_maccs > 0.5).int().numpy()
    pred_fps = (pred_fps > 0.5).int().numpy()

    typer.echo("Writing output file")
    with open(output_filename, "w") as fout:
        for i, name in enumerate(tqdm(names)):
            fout.write(f"name: {name.rstrip('\n')}\n")
            fout.write(f"maccs: [{';'.join(map(str, pred_maccs[i]))}]\n")
            fout.write(f"ecfp4: [{';'.join(map(str, pred_fps[i]))}]\n")
            fout.write("\n")


if __name__ == "__main__":
    app()
