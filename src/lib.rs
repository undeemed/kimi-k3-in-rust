// SPDX-License-Identifier: Apache-2.0
//! Kimi K3 inference engine: a Rust port of <https://github.com/FareedKhan-dev/kimi-k3-in-c>
//! at commit `ff11dce`.
//!
//! OVERVIEW
//!
//! Kimi K3 is a 2.78-trillion-parameter mixture-of-experts model. This engine runs it on a
//! single CPU by treating memory as a dial rather than a floor:
//!
//! - the dense trunk (108.81 GB) is either held resident or streamed from disk in a fixed
//!   layer order, so the next read is always known in advance;
//! - the 1.45 TB of routed experts are never resident. They stream on demand and are
//!   multiplied straight out of their packed MXFP4 form, never widened to fp32.
//!
//! THE THREE WEIGHT FIGURES, since they are easy to confuse
//!
//! | bytes | what |
//! |---|---|
//! | 108.81 GB | the 93 per-layer trunk runs, at bf16. Streamable, and re-read IN FULL on every token. |
//! | 4.70 GB | embed and lm_head. Always resident; not part of the streamed trunk. |
//! | 113.49 GB | the two together: 56,743,648,000 always-active parameters at bf16. Doubles to ~227 GB if anything widens this to fp32, which is why nothing does. |
//! | 1.45 TB | the routed experts, at MXFP4. Never resident at any budget. |
//!
//! The practical consequence is that the model runs in 8 GB of RAM and in 224 GB, and
//! produces byte-identical output at both.
//!
//! ARCHITECTURE (all values verified against the released config.json)
//!
//! 93 layers: 69 Kimi Delta Attention (KDA) + 24 Gated MLA, plus one dense layer.
//! Hidden 7168, 96 heads, 896 routed experts with top-16 selection and 2 shared, latent
//! width 3584, SiTU-GLU activation, MXFP4 expert weights.
//!
//! THREE INVARIANTS THAT MUST HOLD
//!
//! Each is a place where a plausible-looking implementation produces a model that runs,
//! emits fluent text, and is wrong. Each is gated by a fixture in `fixtures/ops` chosen so
//! that getting it wrong changes the output, and each is restated at its point of use.
//!
//! 1. `A_log` is indexed PER HEAD, not per channel. The checkpoint ships `head_dim` floats
//!    but only the first `num_heads` are meaningful; the remainder are padding. Gated by
//!    the `kda_decay` fixture, whose `A_log` is a linspace, so a per-channel misindex moves
//!    every element.
//! 2. MLA uses NoPE, yet the 64 rope dimensions still exist and are still cached. Only the
//!    rotation is absent. Dropping the slots changes the head width. Gated by the `mla`
//!    fixture, which asserts the softmax scale is over the FULL head width
//!    `qk_nope + qk_rope`, not over `qk_nope` alone.
//! 3. The MoE routing bias steers SELECTION only. Combining weights come from the UNBIASED
//!    sigmoid scores. Gated by the `router` fixture, whose bias reorders the top-k on 5 of
//!    its 6 rows.
//!
//! NOT INVARIANTS OF THIS IMPLEMENTATION
//!
//! The UT-transform inverse `(I + Akk)^-1` and the retention of `Aqk`'s diagonal but not
//! `Akk`'s describe the chunked parallel form of the delta rule. This engine does not use
//! it: `ops::kda_step` runs the naive sequential O(T) recurrence, one position at a time.
//! Neither matrix is ever formed, so there is nothing to get right and no test could catch
//! getting it wrong. Anyone adding a chunked KDA path must reinstate them here together
//! with the fixtures that gate them.

// These three lints fight the port's reason for existing, so they are silenced once here
// rather than per item.
//
// - `needless_range_loop`: the numeric kernels are a statement-for-statement translation of
//   the C source, and the index loops are how they stay diffable against it. `for i in 0..n`
//   next to `for (i = 0; i < n; i++)` is the property that lets a reviewer check the two
//   line by line; iterator chains would break that and, where several arrays are indexed by
//   the same counter, would also reorder the reads.
// - `too_many_arguments`: the kernel signatures mirror the C API in `include/k3/k3.h`,
//   argument for argument. Bundling them into structs would make the two headers disagree.
// - `type_complexity`: the resolver callbacks in `bind` are the C `const void *` +
//   out-parameter idiom expressed in the type system; naming each one would add an alias
//   used exactly twice.
#![allow(
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::type_complexity
)]

pub mod bind;
pub mod cache;
pub mod cfg;
pub mod io_util;
pub mod load;
pub mod ops;
pub mod st;
pub mod tok;
pub mod trunk;

pub use cfg::Cfg;
pub use ops::{
    attn_res, decoder_layer, decoder_layer_inc, expert_drops, kda_decay, kda_layer, kda_scratch,
    kda_step, l2norm, layer_scratch, matmul, matmul_bf16, matmul_mxfp4, matmul_q8, mla, mla_cached,
    mla_scratch, mla_scratch_cached, mmw, moe, moe_prefill, moe_scratch, mxfp4_dequant,
    reset_expert_drops, rmsnorm, router, shortconv, situ_glu, Attn, ExpertQ, ExpertSrc, KdaW,
    LayerW, MlaW, MoeW, WMat,
};
