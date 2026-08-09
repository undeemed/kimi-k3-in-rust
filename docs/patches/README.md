# Patches to the C reference

Local patches applied to a clone of
[FareedKhan-dev/kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) to make a
comparison possible. **None of these are committed upstream**, and the clone here is left
clean; they live as files so the comparison is reproducible and the C tree's state is
deliberate rather than residue.

Apply with `git apply` from the root of a C clone at commit `ff11dce`.

## `c-macos-pread-cap.patch`

Without this the C build **cannot load the released checkpoint on macOS at all**, so
there is nothing to compare against.

`k3_st_read` hands a whole tensor to one `pread` and loops on a short return. macOS does
not return short for an oversized request: it rejects it with `EINVAL` for any size
`>= 2^31`. Measured on the real shard, `pread` of 2,147,483,647 bytes succeeds and
2,147,483,648 fails. `embed_tokens` and `lm_head` are `163840 x 7168` at bf16, which is
2,348,810,240 bytes each, so the first call fails at offset 0:

```text
k3_st: short read on language_model.model.embed_tokens.weight at +0
k3_bind: short read of language_model.model.embed_tokens.weight
```

The patch caps every request at `0x7ffff000`. The surrounding loops already handle a
short return, so nothing else changes.

Linux is unaffected: there `pread` caps at the same `0x7ffff000` and returns short rather
than failing, so the existing loop absorbs it. Every published run of the original is on
Linux, which is why this has not surfaced before. The Rust port is unaffected on both
platforms because `pread_full` goes through `FileExt::read_at`, which clamps the request
before the syscall.

Written up as finding 6 in the top-level [README](../../README.md#things-the-port-turned-up).
