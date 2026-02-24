# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
"""
Qwen3-TTS 4-bit pre-quantized model loader (GTX 1080 compatible).

Loads smdesai/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit.
- Download size: ~486MB (60% smaller than original)
- No bitsandbytes dependency required

Usage:
  load_4bit_dequant(): Dequantize to fp32 at load time (~2.4GB VRAM, fast inference)
"""
import json
import os

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModel, AutoProcessor

from ..core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
from ..core.models.modeling_qwen3_tts import Qwen3TTSTalkerRotaryEmbedding, Qwen3TTSRotaryEmbedding
from .qwen3_tts_model import Qwen3TTSModel
from .qwen3_tts_tokenizer import Qwen3TTSTokenizer

REPO_4BIT_DEFAULT = "Wookidooki/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit"
REPO_ORIG_DEFAULT = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
GROUP_SIZE_DEFAULT = 64

# Precomputed shift table (shared across all Linear4bit instances)
_SHIFTS = None


def _get_shifts(device):
    global _SHIFTS
    if _SHIFTS is None or _SHIFTS.device != device:
        _SHIFTS = torch.arange(8, dtype=torch.int64, device=device) * 4
    return _SHIFTS


def dequantize_4bit_affine(weight_uint32, scales, biases, group_size=64):
    """Dequantize custom affine 4-bit uint32-packed weights to float32.

    The 4-bit format packs 8 x 4-bit values per uint32, LSB-first.
    Each group of `group_size` elements shares one scale and one bias:
        val = int4_val * scale + bias
    """
    out_features = weight_uint32.shape[0]
    w = weight_uint32.to(torch.int64)

    shifts = _get_shifts(w.device)
    w_int4 = (w.unsqueeze(-1) >> shifts.view(1, 1, 8)) & 0xF
    w_int4 = w_int4.reshape(out_features, -1).float()

    in_features = w_int4.shape[1]
    num_groups = in_features // group_size

    w_grouped = w_int4.reshape(out_features, num_groups, group_size)
    w_dequant = w_grouped * scales.float().unsqueeze(-1) + biases.float().unsqueeze(-1)
    return w_dequant.reshape(out_features, in_features)


def _get_cache_path(repo_4bit: str, compute_dtype: torch.dtype) -> str:
    cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "qwen_tts")
    os.makedirs(cache_dir, exist_ok=True)
    slug = repo_4bit.replace("/", "--")
    dtype_tag = str(compute_dtype).split(".")[-1]  # e.g. "float32" or "bfloat16"
    return os.path.join(cache_dir, f"{slug}_{dtype_tag}.safetensors")


def load_4bit_dequant(
    repo_4bit=REPO_4BIT_DEFAULT,
    repo_orig=REPO_ORIG_DEFAULT,
    group_size=GROUP_SIZE_DEFAULT,
    compute_dtype=torch.bfloat16,
):
    """Load a pre-quantized 4-bit model by dequantizing to bf16.

    On first run, dequantizes on GPU and caches the bf16 weights to disk.
    On subsequent runs, loads the cached bf16 weights directly to GPU.

    Args:
        repo_4bit: HuggingFace repo id for the 4-bit quantized model.
        repo_orig: HuggingFace repo id for the original model (used for speech_tokenizer).
        group_size: Quantization group size (default 64).
        compute_dtype: Target dtype for dequantized weights (default torch.bfloat16).

    Returns:
        Qwen3TTSModel: Ready-to-use TTS model instance.
    """
    # Auto-detect: GPUs before Ampere (SM < 8.0) lack native bf16 — use fp32
    if torch.cuda.get_device_capability()[0] < 8:
        compute_dtype = torch.float32

    AutoConfig.register("qwen3_tts", Qwen3TTSConfig)
    AutoModel.register(Qwen3TTSConfig, Qwen3TTSForConditionalGeneration)

    # 1) Config -> empty model
    config_path = hf_hub_download(repo_4bit, "config.json")
    config = Qwen3TTSConfig.from_pretrained(os.path.dirname(config_path))

    with torch.device("meta"):
        model = Qwen3TTSForConditionalGeneration._from_config(config, dtype=compute_dtype)

    # 2) Load state_dict: from disk cache if available, else dequantize from 4-bit
    cache_path = _get_cache_path(repo_4bit, compute_dtype)
    if os.path.exists(cache_path):
        print(f"[qwen_tts] Loading cached weights: {cache_path}")
        state_dict = load_file(cache_path, device="cuda")
    else:
        print("[qwen_tts] First run: dequantizing 4-bit weights on GPU (will cache for next time)...")
        st_path = hf_hub_download(repo_4bit, "model.safetensors")

        state_dict = {}
        with safe_open(st_path, framework="pt", device="cuda") as f:
            all_keys = set(f.keys())
            processed = set()

            for key in sorted(all_keys):
                if key in processed:
                    continue

                if key.endswith(".scales"):
                    continue
                if key.endswith(".biases"):
                    base = key[:-7]
                    if f"{base}.scales" in all_keys and f"{base}.weight" in all_keys:
                        continue

                if key.endswith(".weight"):
                    base = key[:-7]
                    scales_key = f"{base}.scales"
                    biases_key = f"{base}.biases"

                    if scales_key in all_keys and biases_key in all_keys:
                        w = f.get_tensor(key)
                        s = f.get_tensor(scales_key)
                        b = f.get_tensor(biases_key)
                        state_dict[key] = dequantize_4bit_affine(w, s, b, group_size).to(compute_dtype)
                        processed.update([key, scales_key, biases_key])

                        bias_key = f"{base}.bias"
                        if bias_key in all_keys:
                            state_dict[bias_key] = f.get_tensor(bias_key).to(compute_dtype)
                            processed.add(bias_key)
                        continue

                state_dict[key] = f.get_tensor(key).to(compute_dtype)
                processed.add(key)

        print(f"[qwen_tts] Saving dequantized cache to {cache_path} (~1.9GB, one-time)...")
        save_file({k: v.contiguous().cpu() for k, v in state_dict.items()}, cache_path)
        print("[qwen_tts] Cache saved.")

    # 3) Load state_dict -> GPU (state_dict is already on CUDA)
    model.load_state_dict(state_dict, strict=False, assign=True)

    # Recompute rotary embeddings (persistent=False, not in state_dict)
    for module in model.modules():
        if isinstance(module, (Qwen3TTSTalkerRotaryEmbedding, Qwen3TTSRotaryEmbedding)):
            inv_freq, module.attention_scaling = module.rope_init_fn(module.config, "cuda")
            module.inv_freq = inv_freq
            module.original_inv_freq = inv_freq

    # Materialize any other remaining meta tensors
    for module in model.modules():
        for key, buf in list(module._buffers.items()):
            if buf is not None and buf.is_meta:
                module._buffers[key] = torch.zeros(buf.shape, dtype=buf.dtype, device="cuda")
        for key, param in list(module._parameters.items()):
            if param is not None and param.is_meta:
                module._parameters[key] = torch.nn.Parameter(
                    torch.zeros(param.shape, dtype=compute_dtype, device="cuda")
                )

    model = model.cuda()

    # 4) Speech tokenizer (from original repo - quantization-independent)
    orig_dir = snapshot_download(repo_orig, allow_patterns=["speech_tokenizer/*"])
    st_dir = os.path.join(orig_dir, "speech_tokenizer")
    speech_tokenizer = Qwen3TTSTokenizer.from_pretrained(st_dir)
    speech_tokenizer.model = speech_tokenizer.model.to(device="cuda", dtype=compute_dtype)
    speech_tokenizer.device = torch.device("cuda")
    model.load_speech_tokenizer(speech_tokenizer)

    # 5) Generation config
    gen_config_path = hf_hub_download(repo_4bit, "generation_config.json")
    with open(gen_config_path, "r", encoding="utf-8") as gf:
        generate_config = json.load(gf)
    model.load_generate_config(generate_config)

    # 6) Processor
    AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor)
    processor = AutoProcessor.from_pretrained(repo_4bit, fix_mistral_regex=True)

    return Qwen3TTSModel(model=model, processor=processor, generate_defaults=generate_config)


