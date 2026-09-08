#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import json
import pickle

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pack fine-tuned stage3 + own latent + rebuilt embedding labels from training json."
    )

    parser.add_argument(
        "--official_ckpt",
        required=True,
        help="Official packed TopoDiff checkpoint, e.g. ../data/weights/v1_1_1/model.ckpt"
    )

    parser.add_argument(
        "--stage3_ckpt",
        required=True,
        help="Fine-tuned stage3 checkpoint, e.g. ../my_experiment_lora/save/ckpt/epoch_25.pkl"
    )

    parser.add_argument(
        "--latent_ckpt",
        required=True,
        help="Fine-tuned latent checkpoint, e.g. ../my_experiment_lora/latent/save/model_1/epoch_25/ckpt/epoch_400.pkl"
    )

    parser.add_argument(
        "--embedding",
        required=True,
        help="Stage3 embedding file, e.g. ../my_experiment_lora/save/embedding/epoch_25.pkl"
    )

    parser.add_argument(
        "--train_json",
        required=True,
        help="Training json used to generate the embedding, e.g. ../my_train_data/my_train_json.json"
    )

    parser.add_argument(
        "-o",
        "--out",
        required=True,
        help="Output packed checkpoint path."
    )

    return parser.parse_args()


def load_torch(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_json(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu()
    return torch.as_tensor(x)


def unwrap_stage3_ckpt(ckpt):
    """
    Handle different save formats.
    A LoRA-trained checkpoint is sometimes a full dict, sometimes the main
    weights live under main_ckpt/model/state_dict.
    """
    if isinstance(ckpt, dict):
        if "main_ckpt" in ckpt:
            return ckpt["main_ckpt"]
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def unwrap_latent_ckpt(ckpt):
    """
    Handle latent checkpoint save formats.
    """
    if isinstance(ckpt, dict):
        if "latent_ckpt" in ckpt:
            return ckpt["latent_ckpt"]
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def get_length_from_record(record):
    length = record.get("trimmed_length", None)

    if length is not None:
        return int(length)

    seq = record.get("seq_pdb", None)
    if isinstance(seq, str) and len(seq) > 0:
        return len(seq)

    seq = record.get("seq_cath", None)
    if isinstance(seq, str) and len(seq) > 0:
        return len(seq)

    raise ValueError(f"Cannot get length from record: keys={list(record.keys())}")


def get_topology_from_record(record, target_dim=9):
    topology = record.get("topology", None)

    if not isinstance(topology, list) or len(topology) == 0:
        raise ValueError(f"Invalid topology in record: {topology}")

    topology = [int(x) for x in topology]

    if len(topology) < target_dim:
        topology = topology + [1] * (target_dim - len(topology))
    elif len(topology) > target_dim:
        topology = topology[:target_dim]

    return topology


def build_label_length(json_data, official_label_length=None):
    lengths = []

    for _, record in json_data.items():
        lengths.append(get_length_from_record(record))

    length_tensor = torch.tensor(lengths, dtype=torch.long)

    # try to match the official label_length dimension format
    if isinstance(official_label_length, torch.Tensor):
        if official_label_length.ndim == 1:
            return length_tensor
        elif official_label_length.ndim >= 2:
            # e.g. official is [N, 1] or [N, k]
            tail_shape = official_label_length.shape[1:]
            view_shape = (len(lengths),) + tuple(1 for _ in tail_shape)
            length_tensor = length_tensor.view(view_shape)
            length_tensor = length_tensor.expand((len(lengths),) + tuple(tail_shape)).clone()
            return length_tensor

    # default to [N, 1], because the sampler applies mean(dim=-1)
    return length_tensor.view(-1, 1)


def build_label_topology(json_data, official_label_topology=None):
    target_dim = 9

    if isinstance(official_label_topology, torch.Tensor):
        if official_label_topology.ndim >= 2:
            target_dim = official_label_topology.shape[1]

    topology_list = []

    for _, record in json_data.items():
        topology_list.append(
            get_topology_from_record(record, target_dim=target_dim)
        )

    topology_tensor = torch.tensor(topology_list, dtype=torch.long)

    return topology_tensor


def main():
    args = parse_args()

    print("========== Loading files ==========")
    print(f"official_ckpt: {args.official_ckpt}")
    print(f"stage3_ckpt  : {args.stage3_ckpt}")
    print(f"latent_ckpt  : {args.latent_ckpt}")
    print(f"embedding    : {args.embedding}")
    print(f"train_json   : {args.train_json}")
    print(f"out          : {args.out}")

    official_ckpt = load_torch(args.official_ckpt)
    stage3_ckpt = unwrap_stage3_ckpt(load_torch(args.stage3_ckpt))
    latent_ckpt = unwrap_latent_ckpt(load_torch(args.latent_ckpt))
    embedding = load_pickle(args.embedding)
    json_data = load_json(args.train_json)

    required_keys = [
        "embedding_dict",
        "sc_head_ckpt",
        "novelty_head_ckpt",
        "alpha_head_ckpt",
        "beta_head_ckpt",
        "coil_head_ckpt",
    ]

    missing_required_keys = [
        key for key in required_keys
        if key not in official_ckpt
    ]

    if missing_required_keys:
        raise ValueError(
            "official_ckpt must be a packed official sampling checkpoint, "
            f"but these keys are missing: {missing_required_keys}"
        )

    if "latent_mu" not in embedding:
        raise ValueError(
            f"embedding file does not contain latent_mu: {args.embedding}"
        )

    latent_mu = to_tensor(embedding["latent_mu"]).float()
    n_latent = latent_mu.shape[0]
    n_json = len(json_data)

    print("\n========== Count check ==========")
    print(f"latent_mu count: {n_latent}")
    print(f"json count     : {n_json}")

    if n_latent != n_json:
        raise ValueError(
            "Count mismatch: latent_mu count != train_json count.\n"
            f"latent_mu count = {n_latent}\n"
            f"json count      = {n_json}\n\n"
            "This means the embedding file and train_json are not the same dataset or are not in the same order.\n"
            "Please use the json that truly corresponds to the training run, or regenerate the embedding."
        )

    official_embedding_dict = official_ckpt["embedding_dict"]
    embedding_dict = dict(official_embedding_dict)

    official_label_length = official_embedding_dict.get("label_length", None)
    official_label_topology = official_embedding_dict.get("label_topology", None)

    # 1. replace with our own latent
    embedding_dict["label_latent"] = latent_mu

    # 2. rebuild length from our own training json
    embedding_dict["label_length"] = build_label_length(
        json_data=json_data,
        official_label_length=official_label_length,
    )

    # 3. rebuild topology from our own training json
    embedding_dict["label_topology"] = build_label_topology(
        json_data=json_data,
        official_label_topology=official_label_topology,
    )

    # 4. if the official checkpoint has label_name, replace it with our json key order
    if "label_name" in embedding_dict:
        embedding_dict["label_name"] = list(json_data.keys())

    # 5. check that all core fields have consistent counts
    print("\n========== New embedding_dict check ==========")
    for k in ["label_latent", "label_length", "label_topology", "label_name"]:
        if k in embedding_dict:
            v = embedding_dict[k]
            if isinstance(v, torch.Tensor):
                print(f"{k}: tensor shape={tuple(v.shape)} dtype={v.dtype}")
            elif isinstance(v, list):
                print(f"{k}: list len={len(v)}")
            else:
                print(f"{k}: {type(v)}")

    if embedding_dict["label_latent"].shape[0] != n_json:
        raise ValueError("label_latent size mismatch")

    if embedding_dict["label_length"].shape[0] != n_json:
        raise ValueError("label_length size mismatch")

    if embedding_dict["label_topology"].shape[0] != n_json:
        raise ValueError("label_topology size mismatch")

    if "label_name" in embedding_dict and len(embedding_dict["label_name"]) != n_json:
        raise ValueError("label_name size mismatch")

    # 6. assemble the packed checkpoint
    packed_ckpt = dict(official_ckpt)
    packed_ckpt["main_ckpt"] = stage3_ckpt
    packed_ckpt["latent_ckpt"] = latent_ckpt
    packed_ckpt["embedding_dict"] = embedding_dict

    packed_ckpt["pack_info"] = {
        "main_ckpt_source": os.path.abspath(args.stage3_ckpt),
        "latent_ckpt_source": os.path.abspath(args.latent_ckpt),
        "embedding_source": os.path.abspath(args.embedding),
        "train_json_source": os.path.abspath(args.train_json),
        "official_template_source": os.path.abspath(args.official_ckpt),
        "description": "fine-tuned LoRA stage3 + own latent + rebuilt label_length/label_topology from training json",
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(packed_ckpt, args.out)

    print("\n========== Done ==========")
    print(f"Saved packed checkpoint to: {args.out}")
    print("Packed keys:", sorted(packed_ckpt.keys()))


if __name__ == "__main__":
    main()