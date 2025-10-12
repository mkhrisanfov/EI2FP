import torch
import torch.nn as nn


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


class EI2FPBase(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def train_fn(self, optim, loss_fn, train_dl):
        self.train()
        epoch_train_loss = 0
        for maccs, fps, spectra in train_dl:
            optim.zero_grad()
            pred_maccs, pred_fps = self.forward(
                spectra.to(next(self.parameters()).device, non_blocking=True)
            )
            loss1 = loss_fn(
                pred_maccs, maccs.to(next(self.parameters()).device, non_blocking=True)
            ).sum()
            loss2 = loss_fn(
                pred_fps, fps.to(next(self.parameters()).device, non_blocking=True)
            ).sum()
            loss = (167 * loss1 + 1024 * loss2) / (167 + 1024)
            loss.backward()
            optim.step()
            epoch_train_loss += loss.detach().cpu()
        return epoch_train_loss / len(train_dl.dataset)

    def eval_fn(self, loss_fn, eval_dl):
        self.eval()
        epoch_eval_loss = 0
        with torch.no_grad():
            for maccs, fps, spectra in eval_dl:
                pred_maccs, pred_fps = self.forward(
                    spectra.to(next(self.parameters()).device, non_blocking=True)
                )
                loss1 = loss_fn(
                    pred_maccs,
                    maccs.to(next(self.parameters()).device, non_blocking=True),
                ).sum()
                loss2 = loss_fn(
                    pred_fps, fps.to(next(self.parameters()).device, non_blocking=True)
                ).sum()
                loss = (167 * loss1 + 1024 * loss2) / (167 + 1024)
                epoch_eval_loss += loss.detach().cpu()
        return epoch_eval_loss / len(eval_dl.dataset)

    def predict(self, forward_dl):
        self.eval()
        all_maccs, all_fps = [], []
        with torch.no_grad():
            for spectra in forward_dl:
                pred_maccs, pred_fps = self.forward(
                    spectra.to(next(self.parameters()).device, non_blocking=True)
                )
                all_maccs.append(pred_maccs)
                all_fps.append(pred_fps)
        all_maccs = torch.cat(all_maccs, 0).cpu()
        all_fps = torch.cat(all_fps, 0).cpu()
        return all_maccs, all_fps


class EI2FPFull(EI2FPBase):
    def __init__(self):
        super().__init__()

        self.fc_first = nn.Sequential(
            nn.Linear(750, 4096),
            nn.SiLU(),
            nn.Linear(4096, 4096),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 2048),
            nn.SiLU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 2048),
            nn.SiLU(),
            nn.BatchNorm1d(2048),
        )

        self.fc_maccs = nn.Sequential(
            nn.Linear(2048, 167),
            nn.Sigmoid(),
        )
        self.fc_fps = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.fc_first(x)
        maccs = self.fc_maccs(x)
        fps = self.fc_fps(x)
        return (maccs, fps)


class EI2FPLite(EI2FPBase):
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
