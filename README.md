# RareTopo

## Introduction

**RareTopo** is a progressive parameter-efficient adaptation framework for adapting a pretrained protein diffusion model to hard-to-sample protein topologies.

RareTopo is built upon [TopoDiff](https://github.com/meneshail/TopoDiff) and progressively applies **Low-Rank Adaptation (LoRA)** to the **structure diffusion** and **latent diffusion** modules.

Using β-solenoid proteins as a representative hard-to-sample topology, RareTopo enables targeted generation of β-solenoid-like protein backbones while retaining broader protein backbone generation capability.

```text
Pretrained TopoDiff
        |
        v
Structure Diffusion LoRA
        |
        v
Latent Diffusion LoRA
        |
        v
      RareTopo
```

---

# Quick Start

The following steps provide the fastest way to install RareTopo, download the released model weights, and generate protein backbones.

## 1. Clone the repository

```bash
git clone https://github.com/The-Protein-Architects/RareTopo.git
cd RareTopo
```

## 2. Create the environment

All required dependencies are provided in the root `environment.yml`.

Create and activate the RareTopo environment:

```bash
conda env create -f environment.yml
conda activate raretopo_env
```

Install TopoDiff:

```bash
pip install -e ./TopoDiff
```

Set the Python path:

```bash
export PYTHONPATH="$PWD/TopoDiff:$PYTHONPATH"
```

The provided environment reproduces the software setup used for RareTopo, including Python 3.8, PyTorch 2.0, and CUDA 11.7.

## 3. Download the RareTopo model weights

The final RareTopo model weights are publicly available through GitHub Releases.

**Release:**

[**RareTopo v1.0.0 – Final Model Weights**](https://github.com/The-Protein-Architects/RareTopo/releases/tag/v1.0.0)

**Checkpoint:**

```text
raretopo_weights.ckpt
```

Create the checkpoint directory:

```bash
mkdir -p ./data/weights/raretopo
```

Download the checkpoint:

```bash
wget -O ./data/weights/raretopo/raretopo_weights.ckpt \
https://github.com/The-Protein-Architects/RareTopo/releases/download/v1.0.0/raretopo_weights.ckpt
```

Alternatively, download `raretopo_weights.ckpt` manually from the GitHub Release page and place it at:

```text
RareTopo/
└── data/
    └── weights/
        └── raretopo/
            └── raretopo_weights.ckpt
```

## 4. Run a small test

Set the GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Create the output directory:

```bash
mkdir -p ./sampling_result
```

Run a small sampling test:

```bash
python ./TopoDiff/run_sampling.py \
  -v custom \
  --ckpt ./data/weights/raretopo/raretopo_weights.ckpt \
  -o ./sampling_result \
  -s 100 \
  -e 120 \
  -i 10 \
  -n 3 \
  --seed 42 \
  --gpu 0 \
  --lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0 \
  --lora_target backbone \
  --latent_lora \
  --latent_lora_rank 4 \
  --latent_lora_alpha 8 \
  --latent_lora_dropout 0 \
  --latent_lora_target backbone
```

This test generates three protein backbones at each of the following lengths:

```text
100, 110, 120
```

Generated structures are saved in:

```text
./sampling_result/
```

---

# Installation

RareTopo is implemented on top of TopoDiff.

The repository contains an `environment.yml` file defining the complete Conda environment required for RareTopo.

Clone the repository:

```bash
git clone https://github.com/The-Protein-Architects/RareTopo.git
cd RareTopo
```

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate raretopo_env
```

Install TopoDiff:

```bash
pip install -e ./TopoDiff
```

Before running training or sampling commands, set:

```bash
export PYTHONPATH="$PWD/TopoDiff:$PYTHONPATH"
```

The main repository structure is:

```text
RareTopo/
├── TopoDiff/
├── notebook/
├── environment.yml
├── my_conditional_guidance_lora.py
├── LICENSE
└── README.md
```

The released RareTopo model weights are distributed separately through **GitHub Releases** and are not stored directly in the source-code repository.

---

# Model Weights

## RareTopo checkpoint

The final RareTopo model weights used in the associated study are publicly available through GitHub Releases.

### Download location

[**RareTopo v1.0.0 – Final Model Weights**](https://github.com/The-Protein-Architects/RareTopo/releases/tag/v1.0.0)

Released checkpoint:

```text
raretopo_weights.ckpt
```

### Local checkpoint location

After downloading, place the checkpoint at:

```text
RareTopo/
└── data/
    └── weights/
        └── raretopo/
            └── raretopo_weights.ckpt
```

The checkpoint path used in the sampling commands is:

```text
./data/weights/raretopo/raretopo_weights.ckpt
```

### Automatic download

From the RareTopo project root, run:

```bash
mkdir -p ./data/weights/raretopo

wget -O ./data/weights/raretopo/raretopo_weights.ckpt \
https://github.com/The-Protein-Architects/RareTopo/releases/download/v1.0.0/raretopo_weights.ckpt
```

After downloading, the expected directory structure is:

```text
RareTopo/
└── data/
    └── weights/
        └── raretopo/
            └── raretopo_weights.ckpt
```

## Original TopoDiff checkpoint

The original pretrained TopoDiff checkpoint is required when **training RareTopo from the pretrained TopoDiff model**.

It can be obtained from the original [TopoDiff repository](https://github.com/meneshail/TopoDiff).

For training, place the TopoDiff checkpoint at:

```text
data/weights/v1_1_2/model.ckpt
```

The recommended complete checkpoint structure is:

```text
RareTopo/
└── data/
    └── weights/
        ├── v1_1_2/
        │   └── model.ckpt
        └── raretopo/
            └── raretopo_weights.ckpt
```

The original TopoDiff checkpoint is not required when directly sampling with the released `raretopo_weights.ckpt`.

---

# Usage

## Unconditional Sampling

To reproduce the unconditional generation experiment reported in the study, first set:

```bash
export PYTHONPATH="$PWD/TopoDiff:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
```

Create the output directory:

```bash
mkdir -p ./sampling_result
```

Run:

```bash
python ./TopoDiff/run_sampling.py \
  -v custom \
  --ckpt ./data/weights/raretopo/raretopo_weights.ckpt \
  -o ./sampling_result \
  -s 80 \
  -e 240 \
  -i 20 \
  -n 200 \
  --seed 42 \
  --gpu 0 \
  --lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0 \
  --lora_target backbone \
  --latent_lora \
  --latent_lora_rank 4 \
  --latent_lora_alpha 8 \
  --latent_lora_dropout 0 \
  --latent_lora_target backbone
```

This generates 200 protein backbones at each of the following lengths:

```text
80, 100, 120, 140, 160, 180, 200, 220, 240
```

for a total of:

```text
9 lengths × 200 structures = 1,800 structures
```

Generated structures are saved in:

```text
./sampling_result/
```

---

## Small Test Run

For a quick installation and model check:

```bash
python ./TopoDiff/run_sampling.py \
  -v custom \
  --ckpt ./data/weights/raretopo/raretopo_weights.ckpt \
  -o ./sampling_result \
  -s 100 \
  -e 120 \
  -i 10 \
  -n 3 \
  --seed 42 \
  --gpu 0 \
  --lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0 \
  --lora_target backbone \
  --latent_lora \
  --latent_lora_rank 4 \
  --latent_lora_alpha 8 \
  --latent_lora_dropout 0 \
  --latent_lora_target backbone
```

This generates three protein backbones at each of the following lengths:

```text
100, 110, 120
```

When using the released RareTopo checkpoint, keep the LoRA configuration consistent with the parameters shown above.

---

# Training

RareTopo uses progressive **structure-to-latent LoRA adaptation**.

Before training, set:

```bash
export PYTHONPATH="$PWD/TopoDiff:$PYTHONPATH"
```

The pretrained TopoDiff checkpoint should be placed at:

```text
data/weights/v1_1_2/model.ckpt
```

## Stage 1: Structure Diffusion LoRA

Create the experiment directory:

```bash
mkdir -p ./experiments/structure_lora
```

Run structure diffusion adaptation:

```bash
python ./TopoDiff/run_training.py \
  -o ./experiments/structure_lora \
  --model structure \
  --stage 3 \
  --init_ckpt ./data/weights/v1_1_2/model.ckpt \
  --batch_size 4 \
  --n_epoch 15 \
  --gpu 0 \
  --lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target backbone
```

The structure diffusion module was adapted for **15 epochs**.

The checkpoint from epoch 15 was used for subsequent latent diffusion adaptation.

## Stage 2: Latent Diffusion LoRA

Run latent diffusion adaptation:

```bash
python ./TopoDiff/run_training.py \
  -o ./experiments/structure_lora \
  --model latent \
  --latent_epoch 15 \
  --init_ckpt ./data/weights/v1_1_2/model.ckpt \
  --n_epoch 100 \
  --latent_batch_size 256 \
  --latent_lr 1e-5 \
  --gpu 0 \
  --lora \
  --lora_rank 4 \
  --lora_alpha 8 \
  --lora_dropout 0.05 \
  --lora_target backbone
```

The latent diffusion module was adapted for **100 epochs** while retaining the adapted structure diffusion module.

The resulting checkpoint corresponds to the final **RareTopo** model.

The final model is publicly released as:

```text
raretopo_weights.ckpt
```

and is available from:

[**RareTopo v1.0.0 – Final Model Weights**](https://github.com/The-Protein-Architects/RareTopo/releases/tag/v1.0.0)

---

# Data Preparation

RareTopo follows the original TopoDiff preprocessing pipeline.

Protein structures can be preprocessed using:

```bash
topodiff-preprocess \
  --input_dir ./data/raw_pdbs/ \
  --output_dir ./data/processed_data/ \
  --n_worker 32
```

The adaptation dataset used in this study contains:

```text
10,996 β-solenoid structures
+
20,000 background structures
=
30,996 structures
```

The β-solenoid structures were collected from:

- Protein Data Bank (PDB)
- AlphaFold Protein Structure Database (AFDB)
- ESM Atlas

Detailed dataset construction, filtering, redundancy removal, and preprocessing procedures are described in the accompanying manuscript and Supplementary Methods.

---

# Reproducibility

The main configuration used for the reported RareTopo experiments is:

```text
RareTopo release:       v1.0.0
Checkpoint:             raretopo_weights.ckpt
Random seed:            42

Structure LoRA
  Rank:                  8
  Alpha:                 16
  Dropout (training):    0.05
  Training epochs:       15

Latent LoRA
  Rank:                  4
  Alpha:                 8
  Dropout (training):    0.05
  Training epochs:       100
```

For sampling with the released checkpoint:

```text
Structure LoRA dropout:  0
Latent LoRA dropout:     0
```

The released `raretopo_weights.ckpt` corresponds to the RareTopo model used in the associated study.

---

# Citation

The accompanying manuscript is:

**RareTopo: Progressive Adaptation of a Protein Diffusion Model to Hard-to-Sample Topologies**

Citation information will be updated upon publication.

RareTopo is developed based on [TopoDiff](https://github.com/meneshail/TopoDiff).

Please also cite the original TopoDiff work when using this repository.

---

# License

RareTopo-specific code is distributed under the license provided in this repository.

Code inherited or modified from TopoDiff remains subject to the original TopoDiff license.

---

# Contact

For questions, bug reports, or usage issues related to RareTopo, please open an issue in this repository.
