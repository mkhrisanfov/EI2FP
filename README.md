## EI2FP: efficient prediction of molecular fingerprints from electron ionization mass spectra

### Overview

The objective of this project is to provide a lightweight and performant neural network model capable of predicting molecular fingerprints (ECFP6 and MACCS keys) from Electron Ionization Mass Spectra (EI-MS). The repository contains code for both the Full and Lite architectures of neural networks, notebook with statistics and figures presented in the original publication. These models address the limitations of the DeepEI approach, which uses separate neural networks for each fingerprint, by employing a single multi-output neural network.

### Original publication

Khrisanfov, M. D., Matyushin, D. D., Samokhin, A. S., & Buryak, A. K. (2024). EI2FP: Efficient Prediction of Molecular Fingerprints from Electron Ionization Mass Spectra. , 15(4), 78-185, [https://doi.org/10.5478/MSL.2024.15.4.178](https://doi.org/10.5478/MSL.2024.15.4.178)

The original pipelines for the article are available in the respective files:

- `./src/ei2fp/deepei_pipeline.py` for DeepEI models;
- `./src/ei2fp/full_pipeline.py` for EI2FP Full model;
- `./src/ei2fp/lite_pipeline.py` for EI2FP Lite model.

The code since have been refactored to allow for easier installation and usage.

### Installation

The project requires **Python 3.12**. You can get one from the [official site](https://www.python.org/downloads/).
Clone the repository:

```
git clone https://github.com/mkhrisanfov/EI2FP
```

Change folder to `EI2FP`:

```
cd EI2FP
```

Create a virtual environment (or install globally, for advanced users, skip to installing dependencies):

```
python -m venv .venv
```

**Activate virtual environment** for Linux:

```bash
source .venv/bin/activate
```

or **activate virtual environment** for Windows (cmd):

```cmd
.venv\Scripts\activate
```

or **activate virtual environment** for Windows (PowerShell):

```powershell
.venv\Scripts\Activate.ps1
```

Install dependencies and `ei2fp` package from [pyproject.toml](./pyproject.toml):

```
pip install -e .
```

Load the weights for EI2Fp model from the huggingface ([mkhrisanfov/EI2FP](https://huggingface.co/mkhrisanfov/EI2FP)) and place them into the `models/` folder:

```
wget -P ./models https://huggingface.co/mkhrisanfov/EI2FP/resolve/main/EI2FPFull.pth
```

### Inference

**Activate virtual environment (see above).**

Run the inference script with .MSP input file containing compounds with ids in the `name:` field and electron ionization mass spectra:

```
ei2fp-predict input_file.msp output_file.msp ./models/EI2FPFull.pth
```

The output will be an .MSP file with original identifiers, as well as **MACCS** keys (field `maccs:`) and **ECFP6** fingerprints (field `ecfp6:`) inside the "[]" brackets separated by semicolons, predicted from electron ionization mass spectra.

Any other id field name can be specified insted of `name:`.
There are two optional arguments that can be used: batch size and `--use_cuda` flag. Full syntax:

```
ei2fp-predict <input file name> <output file name> <model weights> <id field name> <batch size, default is 64> <--use_cuda to use CUDA>
```

### Training and Optimization

The model is implemented to be trained using .MSP files (large mass spectral databases like MassBank, NIST are advised) with the fixed set of hyperparameters using a 80-20% training-validation split:

```
ei2fp-train database.msp
```

The weights for each run will be placed into `models/` folder. Logs, including Tensorboard scalars for train and test at each epoch available at `logs/`.

### Troubleshooting

**If the commands above are not working** (virtual environment is not active or package was installed improperly). Replace the shortcut commands with the full ones, starting from `EI2FP/` folder.

For Linux:

- `ei2fp-predict` -> `.venv/bin/python ./src/ei2fp/predict.py`
- `ei2fp-train` -> `.venv/bin/python ./src/ei2fp/train.py`

For example:

```
.venv/bin/python ./src/ei2fp/predict.py input_file.msp output_file.msp ./models/EI2FPFull.pth
```

For Windows:

- `ei2fp-predict` -> `.venv\Scripts\python.exe src\ei2fp\predict.py`
- `ei2fp-train` -> `.venv\Scripts\python.exe src\ei2fp\train.py`

For example:

```
.venv\Scripts\python.exe src\ei2fp\predict.py input_file.msp output_file.msp models\EI2FPFull.pth
```
