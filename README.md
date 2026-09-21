# RareTopo

## Introduction

**RareTopo** is a progressive parameter-efficient adaptation framework for adapting pretrained protein diffusion models to hard-to-sample protein topologies.

RareTopo is built upon [TopoDiff](https://github.com/meneshail/TopoDiff) and progressively applies **Low-Rank Adaptation (LoRA)** to the structure diffusion and latent diffusion modules.

Using β-solenoid proteins as a representative hard-to-sample topology, RareTopo enables targeted protein backbone generation while retaining broader backbone generation capability.

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

Clone the repository:

```bash
git clone https://github.com/The-Protein-Architects/RareTopo.git
cd RareTopo
```

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

All required dependencies are specified in the root `environment.yml`.

---

## Model Weights

The final RareTopo model weights are available from the GitHub Release:

[**RareTopo v1.0.0 – Final Model Weights**](https://github.com/The-Protein-Architects/RareTopo/releases/tag/v1.0.0)

Released checkpoint:

```text
raretopo_weights.ckpt
```

Download the checkpoint:

```bash
mkdir -p ./data/weights/raretopo

wget -O ./data/weights/raretopo/raretopo_weights.ckpt \
https://github.com/The-Protein-Architects/RareTopo/releases/download/v1.0.0/raretopo_weights.ckpt
```

The expected checkpoint location is:

```text
data/weights/raretopo/raretopo_weights.ckpt
```

---

## Usage

### Unconditional Sampling

Set the GPU:

```bash
export CUDA_VISIBLE_DEVICES=0
```

Create an output directory:

```bash
mkdir -p ./sampling_result
```

Run RareTopo:

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

Main sampling parameters:

```text
-s    minimum backbone length
-e    maximum backbone length
-i    length interval
-n    number of generated backbones per length
```

The command above generates 200 protein backbones at each length from 80 to 240 aa with an interval of 20 aa, producing a total of 1,800 structures.

Generated structures are saved in:

```text
./sampling_result/
```

When using the released RareTopo checkpoint, keep the LoRA configuration consistent with the parameters shown above.

---

<details>
<summary><b>Training RareTopo</b></summary>

RareTopo uses progressive **structure-to-latent LoRA adaptation**.

The original pretrained TopoDiff checkpoint is required for training and can be obtained from the [TopoDiff repository](https://github.com/meneshail/TopoDiff).

Place the pretrained TopoDiff checkpoint at:

```text
data/weights/v1_1_2/model.ckpt
```

Before training, set:

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
  --n_epoch 15 \
  --gpu 0 \
  --lora \
  --lora_rank 8 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --lora_target backbone
```

The structure diffusion module was adapted for **15 epochs**.

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

The latent diffusion module was adapted for **100 epochs**.

The resulting checkpoint corresponds to the final **RareTopo** model.

</details>

---

<details>
<summary><b>Data Preparation</b></summary>

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

Detailed dataset construction, filtering, redundancy removal, and preprocessing procedures are described in the associated manuscript and Supplementary Methods.

</details>

---

## Citation

The associated manuscript is:

**RareTopo: Progressive Adaptation of a Protein Diffusion Model to Hard-to-Sample Topologies**

Citation information will be updated upon publication.

RareTopo is developed based on [TopoDiff](https://github.com/meneshail/TopoDiff). Please also cite the original TopoDiff work when using this repository.

---

## License

RareTopo-specific code is distributed under the license provided in this repository.

Code inherited or modified from TopoDiff remains subject to the original TopoDiff license.

---

## Contact

For questions, bug reports, or usage issues related to RareTopo, please open an issue in this repository.
