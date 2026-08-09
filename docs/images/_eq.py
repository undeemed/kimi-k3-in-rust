"""Render the blog's equations to white-background PNGs via matplotlib mathtext.

Each image is one equation (large) plus a short plain caption underneath. No LaTeX
install needed, mathtext covers everything here.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path(__file__).resolve().parent


def render(name: str, formula: str, caption: str) -> None:
    fig = plt.figure(figsize=(8.4, 2.0), dpi=200)
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.60, formula, fontsize=24, ha="center", va="center", color="#1f2937")
    fig.text(0.5, 0.16, caption, fontsize=11.5, ha="center", va="center", color="#6b7280")
    path = OUT / f"{name}.png"
    fig.savefig(path, facecolor="white", bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"  rendered {path.name}")


EQUATIONS = {
    # ---- the fit ledger
    "eq_naive_memory": (
        r"$\mathrm{bytes} = P \cdot b = 2.78\times10^{12} \cdot 2 = 5.56\ \mathrm{TB}$",
        "The naive requirement: every parameter resident at bf16",
    ),
    "eq_fit_ledger": (
        r"$5560\ \mathrm{GB} \to 1560 \to 113.49 \to 8.24\ \mathrm{GB}$",
        "Four reductions: MXFP4 experts, routing, then streaming the trunk",
    ),
    "eq_sparsity_ratio": (
        r"$\frac{\mathrm{active}}{\mathrm{total}} = \frac{104\times10^{9}}{2.78\times10^{12}} \approx 3.7\%$",
        "Only 16 of 896 experts fire per layer, so most of the model sleeps",
    ),
    "eq_resident_set": (
        r"$\mathrm{resident} = \mathrm{trunk} + \mathrm{embed} + \mathrm{lm\_head} = 56{,}743{,}648{,}000\ \mathrm{params}$",
        "The always-active set: 113.49 GB at bf16, everything else is streamable",
    ),
    # ---- formats and files
    "eq_st_layout": (
        r"$\mathrm{file} = [\,8\ \mathrm{B\ length}\,][\,\mathrm{JSON\ header}\,][\,\mathrm{tensor\ data}\,]$",
        "safetensors: one length, one header, then raw bytes at known offsets",
    ),
    "eq_layer_map": (
        r"$\mathrm{MLA} = \{4, 8, 12, \ldots, 92\} \cup \{93\}, \quad |\mathrm{MLA}| = 24$",
        "One-based layer indices, and 92 and 93 are both MLA by design",
    ),
    "eq_mxfp4": (
        r"$w_i = \mathrm{E2M1}[n_i] \cdot 2^{\,s_g - 127}, \quad g = \lfloor i/32 \rfloor$",
        "MXFP4: a 4-bit nibble scaled by one 8-bit exponent per 32 weights",
    ),
    "eq_bytes_per_weight": (
        r"$\frac{1}{2} + \frac{1}{32} = 0.53125\ \frac{\mathrm{bytes}}{\mathrm{weight}} \;\Rightarrow\; 17{,}547{,}264\ \mathrm{B}$",
        "Half a byte per weight plus the shared scale gives one expert exactly",
    ),
    "eq_nibble_pack": (
        r"$\mathrm{byte} = n_{\mathrm{even}} + 16\, n_{\mathrm{odd}}, \quad n \in [0,\,15]$",
        "The low nibble is the EVEN element, and reversing it is silently wrong",
    ),
    "eq_dequant_cost": (
        r"$132\ \mathrm{MB} \times 1472\ \mathrm{experts} = 194\ \frac{\mathrm{GB}}{\mathrm{token}}$",
        "What dequantizing would cost, which is why we multiply from the nibbles",
    ),
    # ---- kernels
    "eq_rmsnorm": (
        r"$y_i = \frac{x_i}{\sqrt{\frac{1}{n}\sum_j x_j^{2} + \varepsilon}} \cdot g_i$",
        "RMSNorm with epsilon inside the square root, accumulated in double",
    ),
    "eq_bf16_widen": (
        r"$\mathrm{f32}(w) = \mathrm{u32}(w) \ll 16$",
        "bf16 to fp32 is a shift, not a conversion, so widening is lossless",
    ),
    "eq_accum_order": (
        r"$s = (s_0 + s_1) + (s_2 + s_3), \quad s_k = \sum_{i \equiv k\ (4)} x_i w_i$",
        "A fixed reduction order, so portable, NEON and AVX2 agree bit for bit",
    ),
    # ---- KDA
    "eq_shortconv": (
        r"$y_t = \mathrm{SiLU}\!\left(\sum_{j=0}^{3} c_j\, x_{t-j}\right)$",
        "A depthwise causal convolution of width 4 with the activation fused in",
    ),
    "eq_l2norm": (
        r"$\hat{q} = \frac{q}{\sqrt{\sum_i q_i^{2} + \varepsilon}}$",
        "A sum of squares, not a mean, and applied to q and k only",
    ),
    "eq_kda_decay": (
        r"$g_h = g_{\min} \cdot \sigma\!\left(e^{A_h} z\right), \quad g_{\min} = -5$",
        "The decay gate, with A indexed per head and not per channel",
    ),
    "eq_kda_recurrence": (
        r"$S_t = \mathrm{diag}(g)\,S_{t-1} + \beta\,(v_t - S_{t-1}k_t)\,k_t^{\top}$",
        "The delta rule: decay, read, write the difference, then read again",
    ),
    "eq_kda_gate": (
        r"$o = \mathrm{RMSNorm}(o) \odot \sigma(W_g x), \quad y = W_o\, o$",
        "Norm first, then gate, then project, and that order is not interchangeable",
    ),
    "eq_prefetch": (
        r"$t_{\mathrm{layer}} = \max\left(t_{\mathrm{read}}(L{+}1),\ t_{\mathrm{compute}}(L)\right)$",
        "A fixed walk order means the next read can start before this layer finishes",
    ),
    "eq_kda_state": (
        r"$|S| = H D^{2} = 96 \cdot 128^{2} \;\Rightarrow\; 626.25\ \mathrm{MB\ total}$",
        "The recurrent state is the same size at 10 tokens and at 100,000",
    ),
    # ---- MLA
    "eq_mla_latent": (
        r"$c_t = W_{kv}\,x_t \in \mathbb{R}^{576}, \quad k_t, v_t = W_{up}\,c_t$",
        "One latent per position is cached, and k and v are rebuilt on use",
    ),
    "eq_mla_kv": (
        r"$\mathrm{KV} = 2.37\ \frac{\mathrm{MB}}{\mathrm{position}} \;\Rightarrow\; 19.4\ \mathrm{GB\ at\ 8k}$",
        "Twenty-four MLA layers, expanded k and v in fp32, per position",
    ),
    "eq_kv_compression": (
        r"$\frac{H(d_{qk}+d_v)}{d_{kv}+d_{rope}} = \frac{96 \cdot 320}{576} \approx 53\times$",
        "The released code caches expanded heads, the latent is 53x smaller",
    ),
    # ---- residuals and MoE
    "eq_attn_res": (
        r"$h = \sum_{b} \mathrm{softmax}_b\!\left(\hat{s}_b \cdot f\right) s_b$",
        "Each layer attends over the outputs of every preceding block",
    ),
    "eq_block_count": (
        r"$n_{\mathrm{blocks}} = \left\lfloor \frac{93}{12} \right\rfloor + 2 = 9$",
        "Blocks of twelve, so the residual stack never exceeds nine sources",
    ),
    "eq_router": (
        r"$\mathrm{select} = \mathrm{top}_{16}\left(\sigma(\ell_e) + b_e\right), \quad w_e = \sigma(\ell_e)$",
        "The bias steers selection only, the weights come from unbiased scores",
    ),
    "eq_situ_glu": (
        r"$y = \beta_1 \tanh\!\left(\frac{u}{\beta_1}\right) \cdot \beta_2 \tanh\!\left(\frac{v}{\beta_2}\right), \ |y| \leq 100$",
        "SiTU-GLU with beta1 = 4 and beta2 = 25, so the product is bounded",
    ),
    "eq_moe_latent": (
        r"$7168 \longrightarrow 3584 \longrightarrow 16\ \mathrm{experts} \longrightarrow \mathrm{norm} \longrightarrow 7168$",
        "Experts run in a narrow latent, and the norm is on the aggregate",
    ),
    # ---- streaming and cache
    "eq_align": (
        r"$\mathrm{offset} \equiv 0, \quad \mathrm{length} \equiv 0 \quad (\mathrm{mod}\ 4096)$",
        "O_DIRECT needs both ends aligned, which costs 4 KB per layer",
    ),
    "eq_ram_budget": (
        r"$\mathrm{need} = \mathrm{trunk} + \mathrm{model} + \mathrm{cache} + \mathrm{state} + \mathrm{buf} + \mathrm{KV}$",
        "Everything is summed before anything is allocated, then compared to free RAM",
    ),
    "eq_slot_count": (
        r"$n_{\mathrm{slot}} = \left\lfloor \frac{\mathrm{budget}}{17.56\ \mathrm{MB}} \right\rfloor$",
        "The expert cache holds whole experts, so the budget divides exactly",
    ),
    "eq_queue_depth": (
        r"$\mathrm{QD} = 16 \ \mathrm{vs} \ \mathrm{QD} = 1$",
        "Batched preads keep the device busy, serial gets leave it idle",
    ),
    "eq_compulsory": (
        r"$\mathrm{hit}_{\max} = 1 - \frac{10{,}010}{100{,}096} = 90.0\%$",
        "Every distinct expert must be read once, so no policy can beat this",
    ),
    "eq_working_set": (
        r"$10{,}010 \times 17.55\ \mathrm{MB} = 175.65\ \mathrm{GB}$",
        "The experts this trace touched at all, against a 1.45 TB pool",
    ),
    # ---- validation
    "eq_tolerance": (
        r"$\mathrm{budget} = 1.2\times10^{-7}\sqrt{d}\cdot 50 = 5.1\times10^{-4}$",
        "The rounding budget for 93 layers at hidden size 7168",
    ),
    # ---- performance
    "eq_traffic": (
        r"$\mathrm{bytes/token} = 108.81 + 25.83 = 134.64\ \mathrm{GB}$",
        "The trunk is re-read in full every token, the experts are only sampled",
    ),
    "eq_prefill": (
        r"$t_0 \propto n_{\mathrm{prompt}}, \qquad t_{n>0} \approx \mathrm{const}$",
        "Step zero pays for the whole prompt, every later step reads 25.83 GB",
    ),
    "eq_trunk_hit": (
        r"$\frac{(T-1)\,p}{T \cdot 93} = \frac{7 \cdot 90}{744} = 84.7\%$",
        "Token zero pays the pinning cost, so short runs understate the hit rate",
    ),
    "eq_retention": (
        r"$\mathrm{retained} = 1 - \frac{\mathrm{evictions}}{\mathrm{requests}}$",
        "The only expert-cache metric that agrees with the bytes actually read",
    ),
    "eq_gb_per_layer": (
        r"$\frac{108.81\ \mathrm{GB}}{93\ \mathrm{layers}} \approx 1.17\ \frac{\mathrm{GB}}{\mathrm{token}}$",
        "What one pinned layer removes from the guaranteed traffic",
    ),
    "eq_io_share": (
        r"$\mathrm{io} = \frac{t_{\mathrm{trunk}} + t_{\mathrm{expert}}}{t_{\mathrm{wall}}} \in [40.9\%,\ 60.6\%]$",
        "Both terms are whole-run totals, and this is an I/O problem at every budget",
    ),
    "eq_spread": (
        r"$\frac{\max - \min}{\mathrm{mean}} = \frac{20.14 - 14.67}{16.53} = 33.1\%$",
        "Three identical runs, so a smaller difference is not an effect",
    ),
    # ---- precision
    "eq_absmax": (
        r"$s = \frac{\max_i |w_i|}{2^{\,b-1}-1}, \quad \hat{w}_i = s \cdot \mathrm{round}\!\left(\frac{w_i}{s}\right)$",
        "Symmetric per-row quantization, the method behind the int8 and int4 study",
    ),
    "eq_stream_vs_quantize": (
        r"$\mathrm{streaming} \Rightarrow \Delta t \ \mathrm{recoverable}, \quad \mathrm{rounding} \Rightarrow \Delta w \ \mathrm{permanent}$",
        "Seconds are bought back with RAM, and rounding error is not, at any budget",
    ),
    "eq_context_ceiling": (
        r"$10^{6}\ \mathrm{positions} \times 2.37\ \mathrm{MB} \approx 2485\ \mathrm{GB}$",
        "The advertised million-token context is a memory fact, not an engine limit",
    ),
}

ok = fail = 0
for name, (formula, caption) in EQUATIONS.items():
    try:
        render(name, formula, caption)
        ok += 1
    except Exception as e:
        fail += 1
        print(f"  FAILED {name}: {type(e).__name__}: {e}")
print(f"done: {ok} rendered, {fail} failed")
