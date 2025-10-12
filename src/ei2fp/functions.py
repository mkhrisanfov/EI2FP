import re
from math import ceil

import numpy as np
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors
from tqdm.auto import tqdm
from tqdm.contrib.concurrent import thread_map

from ei2fp import MAX_MZ


def get_fps(molecules, radius: int = 3):
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=1024
    )
    morgan_fingerprints = np.array(
        thread_map(
            morgan_generator.GetCountFingerprintAsNumPy,
            molecules,
            chunksize=500,
            max_workers=8,
        )
    )
    return morgan_fingerprints


def get_maccs_base(molecule):
    maccs = rdMolDescriptors.GetMACCSKeysFingerprint(molecule)
    maccs_arr = np.zeros(1)
    DataStructs.ConvertToNumpyArray(maccs, maccs_arr)
    return maccs_arr


def get_maccs(molecules):
    maccs = np.vstack(
        thread_map(
            get_maccs_base,
            molecules,
            chunksize=500,
            max_workers=8,
        )
    )
    return maccs


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
