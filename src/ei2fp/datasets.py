import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm.auto import tqdm

from math import ceil
import re
from rdkit import Chem, rdBase

from ei2fp.functions import get_fps, get_maccs
from ei2fp import MAX_MZ
from sklearn.model_selection import train_test_split


class EI2FPDataset(Dataset):

    def __init__(self, spectra, molecules=None):
        self.spectra = torch.FloatTensor(np.vstack(spectra) / 1000)
        if molecules is not None:
            print("Generating MACCS")
            self.maccs = torch.FloatTensor(get_maccs(molecules))
            print("Generating Morgan fingerprints")
            self.fps = torch.clip(torch.FloatTensor(get_fps(molecules, radius=3)), 0, 1)

    def __getitem__(self, index):
        if not hasattr(self, "maccs"):
            return self.spectra[index]
        else:
            return (self.maccs[index], self.fps[index], self.spectra[index])

    def __len__(self):
        return len(self.spectra)


def get_train_test_datasets(file_name, dataset, seed=42):
    compounds = read_msp(file_name)
    smis = [x.get("smiles", None) for x in compounds]
    inchis = [x.get("inchi", None) for x in compounds]
    spectra = [x.get("ms", None) for x in compounds]

    num_smis = np.count_nonzero(smis)
    num_inchis = np.count_nonzero(inchis)

    print("Generating Molecules")
    rdBase.DisableLog("rdApp.*")
    if num_smis == 0 and num_inchis == 0:
        raise ValueError("0 valid identifier strings found")
    inchi_mols = list(map(Chem.MolFromInchi, tqdm(inchis)))
    smiles_mols = list(map(Chem.MolFromSmiles, tqdm(smis)))
    if np.count_nonzero(inchi_mols) >= np.count_nonzero(smiles_mols):
        mols = inchi_mols
    else:
        mols = smiles_mols
    rdBase.EnableLog("rdApp.*")
    print(f"Valid molecules: {len(mols)}")

    smis = np.array(smis)
    spectra = np.vstack(spectra)
    print("Generating molecules")
    inchikeys = [
        Chem.MolToInchiKey(x).split("-")[0] if x is not None else None
        for x in tqdm(mols)
    ]
    unique_inchikeys = set(inchikeys)
    if None in unique_inchikeys:
        unique_inchikeys.remove(None)
    trn_inchikeys, tst_inchikeys = train_test_split(
        sorted(list(unique_inchikeys)),  # pyright: ignore[reportArgumentType]
        test_size=0.2,
        random_state=seed,
    )

    trn_inchikeys = set(trn_inchikeys)
    tst_inchikeys = set(tst_inchikeys)
    trn_mask, tst_mask = [], []
    trn_mols, tst_mols = [], []
    for i, val in enumerate(inchikeys):
        if val in trn_inchikeys:
            trn_mask.append(i)
            trn_mols.append(mols[i])
        elif val in tst_inchikeys:
            tst_mask.append(i)
            tst_mols.append(mols[i])

    trn_spectra = spectra[trn_mask]
    tst_spectra = spectra[tst_mask]

    print("Lengths")
    print("TRN", len(trn_mols), len(trn_spectra))
    print("TST", len(tst_mols), len(tst_spectra))

    trn_ds = dataset(trn_spectra, trn_mols)
    tst_ds = dataset(tst_spectra, tst_mols)

    return (trn_ds, tst_ds)


def get_inference_dataset(input_filename, id_field="name"):
    compounds = read_msp(input_filename)
    spectra = [comp["ms"] for comp in compounds]
    names = [comp[id_field] for comp in compounds]
    dataset = EI2FPDataset(spectra)
    return names, dataset


def read_msp(filename):
    compounds = []
    compound = {"ms": np.zeros(MAX_MZ)}
    pattern = re.compile(r"^(?P<key>[\d\w\s]+):\s+(?P<val>.+)$")
    ms_pattern = re.compile(r"^(?P<mz>[\d\.]+)\s+(?P<int>\d+)$")
    with open(filename, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            match = re.match(pattern, line)
            ms_match = re.match(ms_pattern, line)
            if match:
                compound[match["key"]] = match[  # pyright: ignore[reportArgumentType]
                    "val"
                ]
            elif int(compound.get("num peaks", 0)) > 0 and ms_match:
                mz = int(ceil(float(ms_match["mz"]) - 0.65))
                if mz < MAX_MZ:
                    compound["ms"][mz] = float(ms_match["int"])
            elif line == "\n":
                compound["ms"] = (
                    compound["ms"]
                    / np.clip(compound["ms"].max(), a_min=1e-8, a_max=None)
                    * 999
                )
                compounds.append(compound)
                compound = {"ms": np.zeros(MAX_MZ)}
    return compounds


def get_train_val_test_datasets(file_name, dataset, seed=42, spectra_len=MAX_MZ):
    # compounds = []
    names = []
    smis = []
    spectra = []
    with open(file_name, "r") as f:  # "../data/input/input_lib.ms"
        for line in tqdm(f):
            name, smiles, spectrum, _ = line.split("|")
            spectrum = list(map(float, spectrum.split()))
            dense_spectrum = np.zeros(spectra_len)
            spectrum = np.array(spectrum).reshape(-1, 2)
            dense_spectrum[spectrum[:, 0].astype(int)] = spectrum[:, 1]
            names.append(name.strip())
            smis.append(smiles.strip())
            spectra.append(dense_spectrum)

    smis = np.array(smis)
    spectra = np.vstack(spectra)
    unique_smis, unique_index, unique_inverse = np.unique(
        smis, return_index=True, return_inverse=True
    )
    trn_val, tst = train_test_split(
        np.arange(len(unique_smis)), test_size=0.1, random_state=seed
    )
    trn, val = train_test_split(trn_val, test_size=0.15, random_state=seed)
    unique_trn_smis = set(unique_smis[trn])
    unique_val_smis = set(unique_smis[val])
    unique_tst_smis = set(unique_smis[tst])

    trn_idx, val_idx, tst_idx = [], [], []
    for i, smi in enumerate(smis):
        if smi in unique_trn_smis:
            trn_idx.append(i)
        elif smi in unique_val_smis:
            val_idx.append(i)
        elif smi in unique_tst_smis:
            tst_idx.append(i)

    trn_smis = smis[trn_idx]
    trn_spectra = spectra[trn_idx]

    val_smis = smis[val_idx]
    val_spectra = spectra[val_idx]

    tst_smis = smis[tst_idx]
    tst_spectra = spectra[tst_idx]

    print("Lengths")
    print("TRN", len(trn_smis), len(trn_spectra))
    print("VAL", len(val_smis), len(val_spectra))
    print("TST", len(tst_smis), len(tst_spectra))

    print("Intersections")
    print("TRN and VAL", np.intersect1d(trn_smis, val_smis))
    print("TRN and TST", np.intersect1d(trn_smis, tst_smis))
    print("TST and VAL", np.intersect1d(tst_smis, val_smis))
    trn_ds = dataset(trn_smis, trn_spectra)
    val_ds = dataset(val_smis, val_spectra)
    tst_ds = dataset(tst_smis, tst_spectra)

    return (trn_ds, val_ds, tst_ds)
