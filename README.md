
# Transformers for Infant Brain EEG Modelling
This code repository includes a pre-trained Transformer-based model for processing a longitudinal EEG dataset from South African infants. 

This repository contains adapted code from the original paper *EEGPT: Pretrained Transformer for Universal and Reliable Representation of EEG Signals*
by Guagnyu Wang, Wenchao Liu, Yuhong He, Cong Xu, Lin Ma, Haifeng Li 
The original code repository can be found at [EEGPT repository](https://github.com/BINE022/EEGPT/tree/main).

The code is adapted and extended by Tziyona Cohen, University of Cape Town (UCT)




## Run Locally

Clone the project

```bash
  git clone https://github.com/tziyonasgit/deepEEG.git
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Proceed to usage section on exact implementation details.


## Datasets
The Khula dataset used in this project contains confidential patient data and is only available upon request from the Khula Study.

## Checkpoints

You can download the EEGPT pretrained model which is done on a mixed dataset (58-channels, 256Hz, 4s time length EEG) using patch size 64 -> [EEG_large](https://figshare.com/s/e37df4f8a907a866df4b)

Our pretrained model with the Khula dataset can be found here:
- [large variant](https://drive.google.com/file/d/1d7Ox07lFgiuaoYltFI1wNcJh_VJtEOLP/view?usp=drive_link)
- [tiny2 variant](https://drive.google.com/file/d/1hUhNUaIU5y3FAzpldCwGu81yZvJLYEuo/view?usp=drive_link)



## Usage
When running the following commands for each program, all parameters are kept at their default settings unless manually changed or specified upon running:

### Finetune: classification
```bash
  cd /EEGPT/downstream/classification
  python run_class_finetuning.py
```

### Finetune: regression
```bash
  cd /EEGPT/downstream/regression
  python run_reg_finetuning.py
```

### Pretrain
```bash
  cd /EEGPT/pretrain
  python run_pretraining.py
```
## Data preprocessing

The ```setup``` folder in this repository contains the necessary files.

Assuming you have access to the khula dataset, the following files handle preprocessing:

- Multi-age Classification:
```bash
  python make_khula.py
```

- Binary age Classification:
```bash
  python make_khula_binary.py
```

- Regression:
```bash
  python make_khula_reg.py
```

- Manual feature engineering for linear regression model:
```bash
  python make_khula_linreg.py
```

- Synthetic data:
```bash
  python make_synthetic.py
```
## Baseline model
To run the linear regression model, do the following:
```bash
    cd Baselines
    python linear_regression.py
```