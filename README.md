# RareTopo

## Introduction

This repository contains the official implementation of **RareTopo**, a progressive parameter-efficient adaptation framework for adapting a pretrained protein diffusion model to hard-to-sample protein topologies.

RareTopo is built upon [TopoDiff](https://github.com/meneshail/TopoDiff) and sequentially applies **Low-Rank Adaptation (LoRA)** to the **structure diffusion** and **latent diffusion** modules.

Using β-solenoid proteins as a representative hard-to-sample topology, RareTopo improves target-topology generation while retaining broader protein backbone generation capability.

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

## Installation

```bash
git clone https://github.com/The-Protein-Architects/RareTopo.git
cd RareTopo

conda env create -n raretopo_env -f TopoDiff/env.yml
conda activate raretopo_env

pip install -e ./TopoDiff
```

The main repository structure is:

```text
RareTopo/
├── TopoDiff/
├── notebook/
├── my_conditional_guidance_lora.py
├── LICENSE
└── README.md
```

Model checkpoints and training datasets are not stored directly in this repository.

---

## Model Weights

The original TopoDiff checkpoint should be obtained from the [TopoDiff repository](https://github.com/meneshail/TopoDiff).

The final RareTopo checkpoint used in this study corresponds to:

```text
stage3_epoch15_latent_lora_epoch100.ckpt
```

For public release, the checkpoint is renamed as:

```text
raretopo.ckpt
```

and should be placed at:

```text
data/weights/raretopo/raretopo.ckpt
```

The pretrained TopoDiff checkpoint used for training should be placed at:

```text
data/weights/v1_1_2/model.ckpt
```

Recommended local checkpoint structure:

```text
RareTopo/
└── data/
    └── weights/
        ├── v1_1_2/
        │   └── model.ckpt
        └── raretopo/
            └── raretopo.ckpt
```

The RareTopo checkpoint will be made publicly available separately upon public release.

---

## Usage

### Unconditional Sampling

Run the following commands from the RareTopo project root:

```bash
cd RareTopo

export PYTHONPATH="$PWD/TopoDiff:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0

mkdir -p ./sampling_result

python ./TopoDiff/run_sampling.py \
  -v custom \
  --ckpt ./data/weights/raretopo/raretopo.ckpt \
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

for a total of 1,800 generated structures.

For a small test run, for example three structures at lengths 100, 110, and 120:

```bash
python ./TopoDiff/run_sampling.py \
  -v custom \
  --ckpt ./data/weights/raretopo/raretopo.ckpt \
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

When using the released RareTopo checkpoint, keep the LoRA configuration unchanged.

---

## Training

RareTopo uses progressive **structure-to-latent LoRA adaptation**.

Before training, run:

```bash
export PYTHONPATH="$PWD/TopoDiff:$PYTHONPATH"
```

### Stage 1: Structure Diffusion LoRA

```bash
mkdir -p ./experiments/structure_lora

python ./TopoDiff/run_training.py \
  -o ./experiments/structure_lora \
  --model structure \
  --stage 3 \
  --init_ckpt ./data/weights/v1_1_2/model.ckpt \
  --batch_size 4 \
  --n_epoch 25 \
  --gpu 0 \
  --lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target backbone
```

The structure diffusion model was trained for 25 epochs, and the checkpoint from **epoch 15** was selected for subsequent latent diffusion adaptation.

### Stage 2: Latent Diffusion LoRA

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

After latent diffusion adaptation, the resulting checkpoint is the final **RareTopo** model.

---

## Data Preparation

RareTopo follows the original TopoDiff preprocessing pipeline.

```bash
topodiff-preprocess \
  --input_dir ./data/raw_pdbs/ \
  --output_dir ./data/processed_data/ \
  --n_worker 32
```

The adaptation dataset contains:

```text
10,996 β-solenoid structures
+
20,000 background structures
=
30,996 structures
```

The β-solenoid structures were collected from PDB, AlphaFold Protein Structure Database, and ESM Atlas.

Detailed dataset construction, filtering, and preprocessing procedures are described in the accompanying manuscript and Supplementary Methods.

---

## Citation

The accompanying manuscript is currently in preparation:

**RareTopo: Progressive Adaptation of a Protein Diffusion Model to Hard-to-Sample Topologies**

Citation information will be updated upon publication.

RareTopo is developed based on [TopoDiff](https://github.com/meneshail/TopoDiff). Please also cite the original TopoDiff work when using this repository.

---

## License

RareTopo-specific code is distributed under the license provided in this repository.

Code inherited or modified from TopoDiff remains subject to the original TopoDiff license.

---

## Contact

For questions or issues related to RareTopo, please open an issue in this repository.