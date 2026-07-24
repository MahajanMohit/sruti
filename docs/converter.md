# The on-device converter

Turns a Hugging Face safetensors checkpoint into a quantized GGUF, entirely on the
phone. This is the thing Sruti does that apps consuming a fixed catalogue cannot:
point it at a supported checkpoint directory and it produces a runnable model with
no desktop anywhere in the loop.

## How it works

Two stages.

**Stage 1 — safetensors to F16 GGUF.** Ours. Architecture detection, tensor name
mapping, the RoPE permutation, dtype conversion and vocabulary conversion all
happen here.

**Stage 2 — F16 to Q4_K_M.** llama.cpp's own `llama_model_quantize`. Reusing it
means per-tensor type selection matches desktop `llama-quantize` exactly rather
than being an approximation of it — including the fallbacks when a tensor's row
size is not divisible by the quantization block size. SmolLM2's 576-wide rows
trigger that on 180 of 272 tensors, and llama.cpp handles it correctly for free.

The F16 intermediate costs disk transiently — roughly two bytes per parameter —
and is deleted once stage 2 succeeds. `estimated_working_bytes()` accounts for
both files existing at once; preflight with it, because running out of space
part-way wastes everything already spent.

## Supported architectures

| Architecture | HF `architectures[0]` | Notes |
|---|---|---|
| Llama | `LlamaForCausalLM`, `MistralForCausalLM` | Q/K permutation applies |
| Qwen2 | `Qwen2ForCausalLM` | QKV biases |
| Qwen3 | `Qwen3ForCausalLM` | Q/K norms, explicit `head_dim` |
| Gemma 2 / 3 | `Gemma2ForCausalLM`, `Gemma3ForCausalLM` | four norms per block, `1 + w` norms |
| Phi-3 | `Phi3ForCausalLM` | fused `qkv_proj` and `gate_up_proj` |

Tokenizers must be byte-level BPE with a `tokenizer.json`. **SentencePiece-only
checkpoints, which ship a `tokenizer.model` protobuf instead, are rejected with a
clear message rather than converted incorrectly.** This currently excludes Gemma
checkpoints that ship no `tokenizer.json`.

## The traps

Every item here produces a model that loads cleanly and generates fluent-looking
text while being subtly wrong. None of them fail loudly. They are the reason the
converter is validated by byte-comparison against the reference rather than by
"it seemed to work".

**Q/K permutation.** Llama-family checkpoints interleave RoPE pairs as two
contiguous halves per head; GGUF expects them adjacent. Qwen and Gemma do not need
this. Applying it to the wrong family, or skipping it for the right one, corrupts
attention silently. Under GQA, Q is permuted by head count and K by *KV* head
count — using one for both corrupts K alone.

**`post_attention_layernorm` means different things.** For Llama and Qwen it maps
to `ffn_norm`. For Gemma, which has four norms per block, it maps to
`post_attention_norm` and `ffn_norm` comes from `pre_feedforward_layernorm`
instead. Same source name, different destination.

**Gemma norms are stored as `w` but implemented as `1 + w`.** Every norm weight
needs shifting, and only norm weights.

**GGUF reverses dimension order.** An HF `[out_features, in_features]` weight
becomes ggml `ne = {in, out}`.

**Tensor data is not aligned.** Safetensors data begins immediately after a
variable-length JSON header, so a tensor of floats routinely starts on an odd
address. Casting to `const float *` and dereferencing is undefined behaviour and
faults on some ARM configurations — every read goes through `memcpy`. UBSan caught
this one.

**Qwen2.5 declares a sliding window it does not use.** `sliding_window: 32768`
alongside `use_sliding_window: false`. Emitting the GGUF key regardless switches on
sliding-window attention the model was never trained with.

**Control tokens are not always flagged special.** Qwen marks `<|fim_prefix|>` and
five siblings as `special: false`. Trusting the flag alone lets those tokens be
rendered as visible text in generated output. Tokens shaped `<|...|>`, `<unused…>`,
or matching a small literal set are promoted to control tokens regardless.

**`add_bos` is a property of the post-processor, not a config flag.** HF encodes
"prepend BOS" as a `TemplateProcessing` post-processor whose single-sequence
template starts with a special token. Llama 3 has one; Qwen and SmolLM do not.
Assuming true inserts a spurious BOS into every prompt.

**Pre-tokenizer identity changes tokenization.** `tokenizer.ggml.pre` selects
llama.cpp's regex set. Llama 3 and Qwen2 differ by a single quantifier —
`\p{N}{1,3}` versus `\p{N}` — which changes how every number is split. Some
families, SmolLM among them, carry no regex at all and are identified by the
structure of their pre-tokenizer chain instead.

> The reference converter identifies pre-tokenizers by encoding a probe string and
> hashing the resulting token ids, which needs a full BPE implementation. Sruti
> matches on the regex where there is one and on the component structure where
> there is not. When neither matches it falls back to `"default"` **and says so** —
> a surfaced warning, not a silent guess.

## Validation

The only trustworthy check is byte-comparison against llama.cpp's own converter.

```bash
# build the host driver — the same code the app runs
cmake -S tools/convert_cli -B build/convert-cli -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build/convert-cli -j"$(nproc)"

# convert with ours
./build/convert-cli/sruti-convert <model-dir> ours-f16.gguf F16

# convert with the reference
python3 native/llama.cpp/convert_hf_to_gguf.py <model-dir> \
    --outfile ref-f16.gguf --outtype f16

# compare
python3 tools/compare_gguf.py ours-f16.gguf ref-f16.gguf

# and check it actually generates
./build/convert-cli/sruti-run ours-f16.gguf "The capital of France is" 24
```

### Results

| Model | Architecture | Tensors | Result |
|---|---|---|---|
| SmolLM2-135M-Instruct | Llama | 272 | **identical** — no differences at all |
| Qwen2.5-0.5B-Instruct | Qwen2 | 290 | 290/290 tensors bit-identical; 4 extra metadata keys, all correct values |

The Qwen extras are `rope.dimension_count`, `vocab_size`, `add_space_prefix` and
`eot_token_id`. Each carries a correct value that llama.cpp would otherwise derive
or default, so the output is a superset of the reference rather than a conflict.
`rope.dimension_count` is kept deliberately: Qwen3 and Gemma3 decouple `head_dim`
from `hidden_size / head_count`, where derivation would be wrong.

Behavioural checks, greedy decoding so runs are directly comparable:

- Tokenization matches the HF tokenizer exactly for both models
  (`[504, 3575, 282, 4649, 314]` and `[785, 6722, 315, 9625, 374]`).
- Generated text is character-identical between our GGUF and the reference GGUF.
- The full Q4_K_M path produces a working model: SmolLM2 at 258 MiB F16 becomes
  100.6 MiB, and answers "The capital of France is" with "Paris. Paris is a city
  that is known for its historical landmarks, art museums, and cultural
  institutions."

### Unit tests

```bash
./tools/run_native_tests.sh
```

Host-side, under ASan and UBSan, no device needed. The permutation is checked
against an independently written reference that walks explicit 4-D strides rather
than the row-copy loop the implementation uses — two implementations of the same
expression agreeing is worth considerably more than one agreeing with itself.
