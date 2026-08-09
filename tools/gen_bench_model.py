#!/usr/bin/env python3
"""Generate a synthetic Kimi K3 checkpoint with REAL-ish per-token compute.

Both engines read these exact bytes, so the token ids must match; only the clock
may differ. Weights are random because the comparison is of arithmetic throughput,
not of model quality.

MXFP4 geometry constraint: an expert matrix is packed at group size 32, so both
`routed_expert_hidden_size` and `moe_intermediate_size` must be multiples of 32.
"""

import json
import shutil
import struct
import sys
from pathlib import Path

import numpy as np

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/k3-bench")
H = int(sys.argv[2] if len(sys.argv) > 2 else 1024)

CFG = {
    "hidden_size": H,
    "num_hidden_layers": 13,
    "vocab_size": 1024,
    "rms_norm_eps": 1e-05,
    "tie_word_embeddings": False,
    "kda_num_heads": 8,
    "kda_head_dim": H // 8,
    "short_conv_kernel_size": 4,
    "gate_lower_bound": -5.0,
    "num_attention_heads": 8,
    "q_lora_rank": H // 2,
    "kv_lora_rank": H // 4,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "mla_use_output_gate": True,
    "num_experts": 8,
    "num_experts_per_token": 2,
    "num_shared_experts": 2,
    "routed_expert_hidden_size": H // 2,
    "moe_intermediate_size": H // 2,
    "routed_scaling_factor": 1.0,
    "moe_renormalize": True,
    "latent_moe_use_norm": True,
    "first_k_dense_replace": 1,
    "intermediate_size": 2 * H,
    "attn_res_block_size": 3,
    "situ_beta": 4.0,
    "situ_linear_beta": 25.0,
    "full_attn_layers": [4, 8, 12, 13],  # ONE-BASED
}

V, L = CFG["vocab_size"], CFG["num_hidden_layers"]
KH, KD = CFG["kda_num_heads"], CFG["kda_head_dim"]
P = KH * KD
NH, QL, KVL = CFG["num_attention_heads"], CFG["q_lora_rank"], CFG["kv_lora_rank"]
QN, QR, VH = CFG["qk_nope_head_dim"], CFG["qk_rope_head_dim"], CFG["v_head_dim"]
LAT, MI = CFG["routed_expert_hidden_size"], CFG["moe_intermediate_size"]
NE, NS, CK = (
    CFG["num_experts"],
    CFG["num_shared_experts"],
    CFG["short_conv_kernel_size"],
)
DI, FKD = CFG["intermediate_size"], CFG["first_k_dense_replace"]
MLA = {x - 1 for x in CFG["full_attn_layers"]}
assert LAT % 32 == 0 and MI % 32 == 0, (LAT, MI)

rng = np.random.default_rng(20260809)
T = {}


def f32(name, shape, vals):
    T[name] = (list(shape), "F32", np.asarray(vals, dtype="<f4").tobytes())


def rnd(*shape, s=0.02):
    return rng.normal(0.0, s, size=shape).astype("<f4")


E2M1_SMALL = np.array([0, 1, 2, 8, 9, 10], dtype=np.uint8)  # 0, .5, 1, -0, -.5, -1
SCALE_BYTE = 121  # E8M0: 2^(121-127)


def expert(name, rows, cols):
    assert cols % 32 == 0
    lo = rng.choice(E2M1_SMALL, size=rows * cols // 2)
    hi = rng.choice(E2M1_SMALL, size=rows * cols // 2)
    T[f"{name}.weight_packed"] = ([rows, cols // 2], "U8", (lo | (hi << 4)).tobytes())
    T[f"{name}.weight_scale"] = (
        [rows, cols // 32],
        "U8",
        np.full(rows * cols // 32, SCALE_BYTE, dtype=np.uint8).tobytes(),
    )


PRE = "language_model.model."
f32(PRE + "embed_tokens.weight", [V, H], rnd(V, H))
f32(PRE + "norm.weight", [H], np.ones(H))
f32(PRE + "output_attn_res_norm.weight", [H], np.ones(H))
f32(PRE + "output_attn_res_proj.weight", [1, H], rnd(H))
f32("language_model.lm_head.weight", [V, H], rnd(V, H))

for l in range(L):
    p = f"{PRE}layers.{l}."
    for n in (
        "input_layernorm",
        "post_attention_layernorm",
        "self_attention_res_norm",
        "mlp_res_norm",
    ):
        f32(f"{p}{n}.weight", [H], np.ones(H))
    for n in ("self_attention_res_proj", "mlp_res_proj"):
        f32(f"{p}{n}.weight", [1, H], rnd(H))

    if l in MLA:
        f32(f"{p}self_attn.q_a_proj.weight", [QL, H], rnd(QL, H))
        f32(f"{p}self_attn.q_a_layernorm.weight", [QL], np.ones(QL))
        f32(
            f"{p}self_attn.q_b_proj.weight",
            [NH * (QN + QR), QL],
            rnd(NH * (QN + QR), QL),
        )
        f32(f"{p}self_attn.kv_a_proj_with_mqa.weight", [KVL + QR, H], rnd(KVL + QR, H))
        f32(f"{p}self_attn.kv_a_layernorm.weight", [KVL], np.ones(KVL))
        f32(
            f"{p}self_attn.kv_b_proj.weight",
            [NH * (QN + VH), KVL],
            rnd(NH * (QN + VH), KVL),
        )
        f32(f"{p}self_attn.o_proj.weight", [H, NH * VH], rnd(H, NH * VH))
        f32(f"{p}self_attn.g_proj.weight", [NH * VH, H], rnd(NH * VH, H))
    else:
        for n in ("q_proj", "k_proj", "v_proj", "g_proj"):
            f32(f"{p}self_attn.{n}.weight", [P, H], rnd(P, H))
        f32(f"{p}self_attn.o_proj.weight", [H, P], rnd(H, P))
        for n in ("q_conv1d", "k_conv1d", "v_conv1d"):
            f32(f"{p}self_attn.{n}.weight", [P, 1, CK], rnd(P, CK))
        f32(f"{p}self_attn.f_a_proj.weight", [KD, H], rnd(KD, H))
        f32(f"{p}self_attn.f_b_proj.weight", [P, KD], rnd(P, KD))
        f32(f"{p}self_attn.b_proj.weight", [KH, H], rnd(KH, H))
        f32(f"{p}self_attn.A_log", [KD], rnd(KD, s=0.5))  # only the first KH are taken
        f32(f"{p}self_attn.dt_bias", [P], rnd(P, s=0.5))
        f32(f"{p}self_attn.o_norm.weight", [KD], np.ones(KD))

    if l < FKD:
        f32(f"{p}mlp.gate_proj.weight", [DI, H], rnd(DI, H))
        f32(f"{p}mlp.up_proj.weight", [DI, H], rnd(DI, H))
        f32(f"{p}mlp.down_proj.weight", [H, DI], rnd(H, DI))
    else:
        m, si = f"{p}block_sparse_moe.", MI * NS
        f32(f"{m}gate.weight", [NE, H], rnd(NE, H))
        f32(f"{m}gate.e_score_correction_bias", [NE], rnd(NE))
        f32(f"{m}routed_expert_down_proj.weight", [LAT, H], rnd(LAT, H))
        f32(f"{m}routed_expert_up_proj.weight", [H, LAT], rnd(H, LAT))
        f32(f"{m}routed_expert_norm.weight", [LAT], np.ones(LAT))
        f32(f"{m}shared_experts.gate_proj.weight", [si, H], rnd(si, H))
        f32(f"{m}shared_experts.up_proj.weight", [si, H], rnd(si, H))
        f32(f"{m}shared_experts.down_proj.weight", [H, si], rnd(H, si))
        for e in range(NE):
            expert(f"{m}experts.{e}.w1", MI, LAT)
            expert(f"{m}experts.{e}.w2", LAT, MI)
            expert(f"{m}experts.{e}.w3", MI, LAT)

shutil.rmtree(OUT, ignore_errors=True)
OUT.mkdir(parents=True)
(OUT / "config.json").write_text(
    json.dumps(CFG, indent=1)
)  # FLAT at root: both readers accept it
hdr, off = {}, 0
for name, (shape, dt, raw) in T.items():
    hdr[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(raw)]}
    off += len(raw)
hb = json.dumps(hdr, separators=(",", ":")).encode()
hb += b" " * ((-len(hb)) % 8)
with (OUT / "model-00001-of-00001.safetensors").open("wb") as f:
    f.write(struct.pack("<Q", len(hb)))
    f.write(hb)
    for _, (_, _, raw) in T.items():
        f.write(raw)
size = (OUT / "model-00001-of-00001.safetensors").stat().st_size
print(f"hidden={H} P={P} tensors={len(T)} size={size / 1e6:.1f} MB -> {OUT}")
