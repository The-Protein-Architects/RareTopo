#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
TOPODIFF_ROOT = os.path.join(PROJECT_ROOT, "TopoDiff")
if TOPODIFF_ROOT not in sys.path:
    sys.path.insert(0, TOPODIFF_ROOT)

import json
import random
import numpy as np
import torch

try:
    torch.backends.mha.set_fastpath_enabled(False)
    print("[torch] disabled mha fastpath for LoRA-wrapped Transformer layers")
except Exception as e:
    print("[torch] could not disable mha fastpath:", e)

from TopoDiff.config.config import model_config
from TopoDiff.model.diffusion import Diffusion
from TopoDiff.model.lora import apply_lora_to_linear
from TopoDiff.data.structure import StructureBuilder
from TopoDiff.data.encoder_transform import coords_to_dict
from myopenfold.np import protein

def parse_args():
    parser = argparse.ArgumentParser(
        description="Conditional (topology-guided) sampling with a fine-tuned LoRA TopoDiff model"
    )

    parser.add_argument("--target_name", type=str, default="4c5eC02", help="Target topology name (key in the cache json)")
    parser.add_argument("--cache_json", type=str, required=True, help="Path to the training cache json (e.g. my_train_json.json)")
    parser.add_argument("--pt_root", type=str, required=True, help="Root directory of the per-target .pt data files")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to the packed fine-tuned checkpoint")
    parser.add_argument("--model_preset", type=str, default="v1_1_2", help="Model preset name")
    parser.add_argument("--outdir", type=str, required=True, help="Output directory for generated PDB files")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device id")

    parser.add_argument("--length_start", type=int, default=100, help="Minimum sequence length to sample")
    parser.add_argument("--length_end", type=int, default=120, help="Maximum sequence length to sample")
    parser.add_argument("--length_interval", type=int, default=10, help="Step between sampled lengths")

    parser.add_argument("--num_gen_per_length", type=int, default=10, help="Number of samples per length")
    parser.add_argument("--timesteps", type=int, default=200, help="Number of diffusion timesteps")
    parser.add_argument("--latent_noise", type=float, default=0.0, help="Std of Gaussian noise added to the latent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    parser.add_argument("--use_lora", dest="use_lora", action="store_true", default=True, help="Enable LoRA (default: enabled)")
    parser.add_argument("--no_lora", dest="use_lora", action="store_false", help="Disable LoRA")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.0, help="LoRA dropout")
    parser.add_argument("--lora_target", type=str, default="backbone", help="Comma-separated module name keywords to apply LoRA")

    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_batched_tensor(x, device):
    if not torch.is_tensor(x):
        x = torch.tensor(x)
    x = x.to(device)
    if x.dim() == 0:
        x = x.unsqueeze(0)
    return x


def build_encoder_feat_from_pt(sample_dict, device):
    if all(k in sample_dict for k in ["encoder_feats", "encoder_coords", "encoder_mask", "encoder_adj_mat"]):
        feat = {}
        for k in ["encoder_feats", "encoder_coords", "encoder_mask", "encoder_adj_mat"]:
            t = to_batched_tensor(sample_dict[k], device)

            if k == "encoder_feats" and t.dim() == 2:
                t = t.unsqueeze(0)
            elif k == "encoder_coords" and t.dim() == 2:
                t = t.unsqueeze(0)
            elif k == "encoder_adj_mat" and t.dim() == 2:
                t = t.unsqueeze(0)
            elif k == "encoder_mask" and t.dim() == 1:
                t = t.unsqueeze(0)

            feat[k] = t
        return feat

    if "coord_gt" not in sample_dict or "coord_gt_mask" not in sample_dict:
        raise KeyError("The current .pt has neither encoder_* nor coord_gt/coord_gt_mask; cannot build encoder input.")

    coord_gt = sample_dict["coord_gt"]
    coord_gt_mask = sample_dict["coord_gt_mask"]

    if not torch.is_tensor(coord_gt):
        coord_gt = torch.tensor(coord_gt)
    if not torch.is_tensor(coord_gt_mask):
        coord_gt_mask = torch.tensor(coord_gt_mask)

    coords = coord_gt[:, 1]
    coords_mask = coord_gt_mask[:, 1].bool()
    coords = coords[coords_mask]

    enc = coords_to_dict(coords)
    feat = {
        "encoder_feats": enc["encoder_feats"].unsqueeze(0).to(device),
        "encoder_coords": enc["encoder_coords"].unsqueeze(0).to(device),
        "encoder_adj_mat": enc["encoder_adj_mat"].unsqueeze(0).to(device),
        "encoder_mask": torch.ones((1, enc["encoder_feats"].shape[0]), dtype=torch.bool, device=device),
    }
    return feat


def detect_topo_embedder_type_from_state_dict(state_dict):
    keys = set(state_dict.keys())

    has_projection = any(
        ("node_topo_projection" in k) or ("edge_topo_projection" in k)
        for k in keys
    )
    has_shift_scale = any(
        ("node_topo_shift" in k) or ("node_topo_scale" in k) or
        ("edge_topo_shift" in k) or ("edge_topo_scale" in k)
        for k in keys
    )

    if has_shift_scale and not has_projection:
        return "continuous_v2"
    elif has_projection and not has_shift_scale:
        return "continuous"
    elif has_projection and has_shift_scale:
        raise ValueError("state_dict contains both topo_projection and topo_shift/scale; cannot auto-detect topo_embedder.type")
    else:
        raise ValueError("state_dict has neither topo_projection nor topo_shift/scale; cannot auto-detect topo_embedder.type")


def load_model(ckpt_path: str, model_preset: str, device: torch.device,
               use_lora: bool, lora_rank: int, lora_alpha: int,
               lora_dropout: float, lora_target: str):
    packed = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(packed, dict) and "main_ckpt" in packed:
        state_dict = packed["main_ckpt"]
    else:
        state_dict = packed

    topo_type = detect_topo_embedder_type_from_state_dict(state_dict)

    cfg = model_config(model_preset)
    cfg.Model.Embedder_v2.topo_embedder.type = topo_type
    cfg.Model.Diffuser.SO3.cache_dir = os.path.join(TOPODIFF_ROOT, "cache")

    print("[sys.path first]", sys.path[:3])
    print("[config] auto topo_embedder.type =", topo_type)
    print("[config] SO3 cache_dir =", cfg.Model.Diffuser.SO3.cache_dir)

    model = Diffusion(cfg.Model, log=False)

    if use_lora:
        target_keywords = [x.strip() for x in lora_target.split(",") if x.strip()]
        replaced = apply_lora_to_linear(
            model,
            target_keywords=target_keywords,
            r=lora_rank,
            alpha=lora_alpha,
            dropout=lora_dropout,
        )
        print(f"[LoRA] enabled=True, target={target_keywords}, replaced={len(replaced)}")

    model = model.to(device).eval()

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"[load] missing={len(missing)}, unexpected={len(unexpected)}")

    if missing:
        print("[missing_keys]")
        for k in missing:
            print(" ", k)

    if unexpected:
        print("[unexpected_keys]")
        for k in unexpected:
            print(" ", k)

    return model


def resolve_target_pt(cache_json: str, pt_root: str, target_name: str):
    with open(cache_json, "r", encoding="utf-8") as f:
        cache = json.load(f)

    if target_name not in cache:
        raise KeyError(f"target_name not found in cache_json: {target_name}")

    pt_path = os.path.join(pt_root, target_name[1:3], f"{target_name}.pt")
    if not os.path.exists(pt_path):
        raise FileNotFoundError(f".pt not found: {pt_path}")

    return pt_path


def get_final_coord4(pred):
    if "coord_hat" not in pred:
        raise KeyError("'coord_hat' not found in pred")

    x = pred["coord_hat"]
    if not torch.is_tensor(x):
        raise TypeError(f"pred['coord_hat'] is not a tensor, got {type(x)}")

    if x.dim() == 3:
        return x
    elif x.dim() == 4:
        return x[-1]
    elif x.dim() == 5:
        return x[0, -1]
    else:
        raise ValueError(f"Unsupported coord_hat dimension: {tuple(x.shape)}")


def save_pdb_from_coord4(coord4_single, out_path: str, label: str):
    if coord4_single.dim() != 3 or coord4_single.shape[-2:] != (4, 3):
        raise ValueError(f"Unexpected coord4_single shape, expected (L,4,3), got {tuple(coord4_single.shape)}")

    sb = StructureBuilder()
    coord37_record, coord37_mask = sb.coord14_to_coord37(coord4_single[None], trunc=True)

    prot_traj = sb.get_coord_traj(
        coord37_record,
        aa_mask=coord37_mask,
        label_override=label,
        default_res="G",
    )

    with open(out_path, "w") as f:
        f.write(protein.to_pdb(prot_traj[0]))


def main():
    args = parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.outdir, exist_ok=True)
    set_seed(args.seed)

    pt_path = resolve_target_pt(args.cache_json, args.pt_root, args.target_name)
    print(f"[target] name={args.target_name}")
    print(f"[target] pt_path={pt_path}")

    model = load_model(
        args.ckpt_path, args.model_preset, device,
        use_lora=args.use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target=args.lora_target,
    )

    sample = torch.load(pt_path, map_location="cpu", weights_only=False)
    print("[sample keys]", list(sample.keys())[:20])

    feat = build_encoder_feat_from_pt(sample, device)

    print("[feat summary]")
    for k, v in feat.items():
        print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")

    with torch.no_grad():
        enc = model.encode_topology(feat)
        z_base = enc["latent_mu"][0]

    lengths = list(range(args.length_start, args.length_end + 1, args.length_interval))
    print(f"[latent] shape={tuple(z_base.shape)}")
    print(f"[latent first 8 dims] {z_base[:8].detach().cpu().numpy()}")
    print(f"[sampling] lengths={lengths}, num_gen_per_length={args.num_gen_per_length}, noise={args.latent_noise}")

    first_debug = True

    with torch.no_grad():
        for num_res in lengths:
            length_dir = os.path.join(args.outdir, f"length_{num_res}")
            os.makedirs(length_dir, exist_ok=True)

            for rep in range(args.num_gen_per_length):
                if args.latent_noise > 0:
                    z = z_base + args.latent_noise * torch.randn_like(z_base)
                else:
                    z = z_base.clone()

                pred = model.sample_latent_conditional(
                    latent=z,
                    return_traj=True,
                    return_frame=False,
                    return_position=True,
                    reconstruct_position=True,
                    num_res=num_res,
                    timestep=args.timesteps,
                )

                coord4_final = get_final_coord4(pred)

                if first_debug:
                    print("[pred keys]", list(pred.keys()))
                    print("[coord_hat final shape]", tuple(coord4_final.shape))
                    print("[coord_hat first 10 values]", coord4_final.reshape(-1)[:10].detach().cpu().tolist())
                    first_debug = False

                out_name = f"{args.target_name}_len{num_res}_rep{rep:03d}.pdb"
                out_path = os.path.join(length_dir, out_name)

                save_pdb_from_coord4(
                    coord4_final,
                    out_path,
                    label=f"target={args.target_name}, len={num_res}, rep={rep}",
                )

            print(f"[done] length={num_res}, generated={args.num_gen_per_length}")

    print(f"All done. Output dir: {os.path.abspath(args.outdir)}")


if __name__ == "__main__":
    main()