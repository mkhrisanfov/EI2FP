import numpy as np
from rdkit.Chem import AllChem, MACCSkeys
from rdkit import DataStructs
from rdkit import Chem
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split


def get_fp(smiles, radius: int = 2):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    fp = AllChem.GetHashedMorganFingerprint(mol, radius=radius, nBits=1024)
    fp_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(fp, fp_arr)
    return fp_arr


def get_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    maccs = MACCSkeys.GenMACCSKeys(mol)
    maccs_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return maccs_arr


def get_train_test_datasets(file_name, dataset, seed=42):
    # compounds = []
    names = []
    smis = []
    spectra = []
    with open(file_name, "r") as f:  # "../Data/mainlib.ms"
        for line in tqdm(f, total=261219):
            name, smiles, spectrum, _ = line.split("|")
            spectrum = list(map(float, spectrum.split()))
            dense_spectrum = np.zeros(750)
            spectrum = np.array(spectrum).reshape(-1, 2)
            dense_spectrum[spectrum[:, 0].astype(int)] = spectrum[:, 1]
            names.append(name.strip())
            smis.append(smiles.strip())
            spectra.append(dense_spectrum)

    smis = np.array(smis)
    spectra = np.vstack(spectra)
    unique_smis, unique_index, unique_inverse = np.unique(
        smis, return_index=True, return_inverse=True)
    trn_val, tst = train_test_split(
        np.arange(len(unique_smis)), test_size=0.1, random_state=seed)
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
