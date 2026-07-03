#!/usr/bin/env python3
# Convert an EAGLE1 draft checkpoint (e.g. yuhuili/EAGLE-LLaMA3.1-Instruct-8B) to GGUF.
#
# The EAGLE1 checkpoint contains only: embed_tokens, fc, layers.0.* (no input_layernorm,
# no final norm, no lm_head). The lm_head is the base model's, applied to the raw layer
# output - so it is raw-copied (same quantized bytes) from the base model GGUF, together
# with all tokenizer metadata. The result loads in llama.cpp as architecture "eagle".
#
# Usage:
#   python scripts/fionda/convert_eagle1_to_gguf.py \
#       --eagle-dir models/EAGLE-LLaMA3.1-Instruct-8B \
#       --base-gguf models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf \
#       --outfile   models/eagle1-llama3.1-8b-instruct.gguf

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

import gguf

logger = logging.getLogger("convert_eagle1_to_gguf")


def permute(weights: np.ndarray, n_head: int, n_head_kv: int) -> np.ndarray:
    # same as LlamaModel.permute in convert_hf_to_gguf.py: HF interleaved rope layout ->
    # the consecutive-pair layout expected by LLAMA_ROPE_TYPE_NORM
    if n_head_kv is not None and n_head != n_head_kv:
        n_head = n_head_kv
    return (weights.reshape(n_head, 2, weights.shape[0] // n_head // 2, *weights.shape[1:])
            .swapaxes(1, 2)
            .reshape(weights.shape))


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an EAGLE1 draft checkpoint to GGUF")
    parser.add_argument("--eagle-dir", type=Path, required=True, help="dir with pytorch_model.bin + config.json")
    parser.add_argument("--base-gguf", type=Path, required=True, help="base (target) model GGUF - source of lm_head and tokenizer")
    parser.add_argument("--outfile",   type=Path, required=True)
    parser.add_argument("--outtype",   choices=["f16", "f32"], default="f16")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    with open(args.eagle_dir / "config.json") as f:
        config = json.load(f)

    assert config.get("rope_scaling") is None, "rope_scaling not supported - rope_freqs generation not implemented"

    n_embd    = config["hidden_size"]
    n_head    = config["num_attention_heads"]
    n_head_kv = config["num_key_value_heads"]

    sd = torch.load(args.eagle_dir / "pytorch_model.bin", map_location="cpu", weights_only=True)
    logger.info("eagle checkpoint keys: %s", list(sd.keys()))

    # tolerate both "layers.0." and "midlayer." naming
    def get(*names: str) -> torch.Tensor | None:
        for name in names:
            if name in sd:
                return sd[name]
        return None

    dtype = np.float16 if args.outtype == "f16" else np.float32
    ftype = gguf.LlamaFileType.MOSTLY_F16 if args.outtype == "f16" else gguf.LlamaFileType.ALL_F32

    def npy(t: torch.Tensor) -> np.ndarray:
        return t.to(torch.float32).numpy().astype(dtype)

    tensors: dict[str, np.ndarray] = {}

    tensors["token_embd.weight"] = npy(get("embed_tokens.weight"))
    tensors["fc.weight"]         = npy(get("fc.weight"))

    fc_bias = get("fc.bias")
    if fc_bias is not None:
        tensors["fc.bias"] = npy(fc_bias)

    tensors["blk.0.attn_q.weight"]      = permute(npy(get("layers.0.self_attn.q_proj.weight", "midlayer.self_attn.q_proj.weight")), n_head, n_head)
    tensors["blk.0.attn_k.weight"]      = permute(npy(get("layers.0.self_attn.k_proj.weight", "midlayer.self_attn.k_proj.weight")), n_head, n_head_kv)
    tensors["blk.0.attn_v.weight"]      = npy(get("layers.0.self_attn.v_proj.weight",         "midlayer.self_attn.v_proj.weight"))
    tensors["blk.0.attn_output.weight"] = npy(get("layers.0.self_attn.o_proj.weight",         "midlayer.self_attn.o_proj.weight"))
    tensors["blk.0.ffn_gate.weight"]    = npy(get("layers.0.mlp.gate_proj.weight",            "midlayer.mlp.gate_proj.weight"))
    tensors["blk.0.ffn_up.weight"]      = npy(get("layers.0.mlp.up_proj.weight",              "midlayer.mlp.up_proj.weight"))
    tensors["blk.0.ffn_down.weight"]    = npy(get("layers.0.mlp.down_proj.weight",            "midlayer.mlp.down_proj.weight"))
    tensors["blk.0.ffn_norm.weight"]    = npy(get("layers.0.post_attention_layernorm.weight", "midlayer.post_attention_layernorm.weight")).astype(np.float32)

    handled = {"embed_tokens.weight", "fc.weight", "fc.bias"}
    handled |= {k for k in sd.keys() if "input_layernorm" in k}  # dropped: EAGLE1 layer 0 has no attn_norm
    for k in sd.keys():
        if k not in handled and not any(part in k for part in ("self_attn", "mlp.", "post_attention_layernorm")):
            raise ValueError(f"unhandled checkpoint tensor: {k}")

    reader = gguf.GGUFReader(args.base_gguf)

    base_ctx_field = reader.get_field("llama.context_length")
    n_ctx = int(base_ctx_field.contents()) if base_ctx_field else config["max_position_embeddings"]

    writer = gguf.GGUFWriter(args.outfile, arch="eagle")

    writer.add_type(gguf.GGUFType.MODEL)
    writer.add_name("EAGLE1 draft " + str(config.get("_name_or_path", args.eagle_dir.name)))
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_file_type(ftype)

    writer.add_context_length(n_ctx)
    writer.add_embedding_length(n_embd)
    writer.add_block_count(config["num_hidden_layers"])
    writer.add_feed_forward_length(config["intermediate_size"])
    writer.add_head_count(n_head)
    writer.add_head_count_kv(n_head_kv)
    writer.add_rope_freq_base(config["rope_theta"])
    writer.add_rope_dimension_count(n_embd // n_head)
    writer.add_layer_norm_rms_eps(config["rms_norm_eps"])
    writer.add_vocab_size(config["vocab_size"])

    # tokenizer: copied verbatim from the base model GGUF
    n_tok_kv = 0
    for field in reader.fields.values():
        if not field.name.startswith("tokenizer."):
            continue
        val_type = field.types[0]
        sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
        writer.add_key_value(field.name, field.contents(), val_type, sub_type=sub_type)
        n_tok_kv += 1
    logger.info("copied %d tokenizer KVs from %s", n_tok_kv, args.base_gguf)
    assert n_tok_kv > 0, "no tokenizer metadata found in base GGUF"

    for name, data in tensors.items():
        logger.info("writing %-28s %s %s", name, data.shape, data.dtype)
        writer.add_tensor(name, data)

    # lm_head: raw copy from the base model (same quantization)
    output_w = next((t for t in reader.tensors if t.name == "output.weight"), None)
    assert output_w is not None, "base GGUF has no output.weight (tied embeddings?) - not supported"
    logger.info("writing %-28s %s %s (raw copy from base)", output_w.name, output_w.shape, output_w.tensor_type.name)
    writer.add_tensor(output_w.name, output_w.data, raw_shape=output_w.data.shape, raw_dtype=output_w.tensor_type)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()

    logger.info("wrote %s", args.outfile)


if __name__ == "__main__":
    main()
