import copy
import json
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM

from transmla.convert_pretrained import convert_model, main, make_export_config, source_config_from_dict
from transmla.lora_qkv import LoraQKV
from transmla.modeling_transmla import TransMLAAttention, TransMLAForCausalLM
from transmla.partial_rope import PartialRope
from transmla.qwen3_conversion import QKNormPartialRope, compress_qwen3_attention
from transmla.utils import evaluate_ppl, get_qkv_calibrate_outputs


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(123)
    torch.set_num_threads(2)


def source_model(family):
    # Qwen3's query width need not equal hidden_size (true for Qwen3-4B).
    cls = Qwen3Config if family == "qwen3" else Qwen2Config
    config = cls(vocab_size=41, hidden_size=24 if family == "qwen3" else 32,
                 intermediate_size=48, num_hidden_layers=2, num_attention_heads=4,
                 num_key_value_heads=2, head_dim=8, max_position_embeddings=128,
                 attention_dropout=0.0, pad_token_id=0, eos_token_id=1,
                 rms_norm_eps=1e-6 if family == "qwen3" else 1e-5,
                 tie_word_embeddings=family == "qwen3", use_sliding_window=False)
    config.head_dim = 8
    config._attn_implementation = "sdpa"
    model = (Qwen3ForCausalLM if family == "qwen3" else Qwen2ForCausalLM)(config).eval()
    if family == "qwen3":
        with torch.no_grad():
            for layer in model.model.layers:
                layer.self_attn.q_norm.weight.copy_(torch.linspace(0.4, 1.7, 8))
                layer.self_attn.k_norm.weight.copy_(torch.linspace(1.6, 0.5, 8))
    else:
        with torch.no_grad():
            for layer in model.model.layers:
                for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj):
                    projection.bias.copy_(torch.linspace(-0.03, 0.04, projection.bias.numel()))
    return model


def batches():
    return [{"input_ids": torch.randint(2, 41, (2, 12)), "attention_mask": torch.ones(2, 12, dtype=torch.long)} for _ in range(2)]


@pytest.mark.parametrize("family", ["qwen3", "mimo"])
@torch.no_grad()
def test_rorope_without_positional_removal_preserves_source(family):
    model = source_model(family)
    data = batches()
    expected = model(**data[0], use_cache=False).logits
    acts = get_qkv_calibrate_outputs(model, data)
    for index, layer in enumerate(model.model.layers):
        cls = QKNormPartialRope if family == "qwen3" else PartialRope
        # Keep RoPE in every KV head: only the equivalent basis rotation remains.
        layer.self_attn = cls(layer.self_attn, acts["key"][index], freqfold=1,
                              collapse=1, rope_head=model.config.num_key_value_heads).eval()
    actual = model(**data[0], use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("family", ["qwen3", "mimo"])
@torch.no_grad()
def test_full_rank_mla_matches_partial_rope_reference(family):
    model = source_model(family)
    data = batches()
    acts = get_qkv_calibrate_outputs(model, data)
    for index, layer in enumerate(model.model.layers):
        cls = QKNormPartialRope if family == "qwen3" else PartialRope
        layer.self_attn = cls(layer.self_attn, acts["key"][index], freqfold=2, collapse=2).eval()
    expected = model(**data[0], use_cache=False).logits
    acts = get_qkv_calibrate_outputs(model, data)
    # 2 * 2 KV heads * 8 dimensions - 4 RoPE dimensions: no PCA truncation.
    config = make_export_config(model.config, kv_lora_rank=28, qk_mqa_dim=4, source_model_type=family)
    for index, layer in enumerate(model.model.layers):
        if family == "qwen3":
            layer.self_attn = compress_qwen3_attention(layer.self_attn, acts["key"][index], acts["value"][index], config)
        else:
            layer.self_attn = LoraQKV(layer.self_attn, acts["query"][index], acts["key"][index], acts["value"][index],
                                      kv_lora_rank=28, qk_mqa_dim=4, collapse=2, balance_kv_ratio=1.0).eval()
    actual = model(**data[0], use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("family", ["qwen3", "mimo"])
@torch.no_grad()
def test_portable_export_cache_padding_and_generation(family, tmp_path):
    source = source_model(family)
    original = {k: v.clone() for k, v in source.state_dict().items() if ".self_attn." not in k}
    norms = copy.deepcopy(source.model.layers[0].self_attn.q_norm) if family == "qwen3" else None
    model = convert_model(source, batches(), kv_lora_rank=12, qk_mqa_dim=4, freqfold=2, source_model_type=family)
    for name, value in original.items():
        torch.testing.assert_close(model.state_dict()[name], value, atol=0, rtol=0)
    if norms:
        torch.testing.assert_close(model.model.layers[0].self_attn.q_norm.weight, norms.weight, atol=0, rtol=0)
    assert all(p.grad is None and not p.requires_grad for p in model.parameters())
    tokens = torch.randint(2, 41, (1, 9))
    expected = model(tokens, use_cache=False).logits
    prefill = model(tokens[:, :4], use_cache=True)
    middle = model(tokens[:, 4:7], past_key_values=prefill.past_key_values, use_cache=True)
    final = model(tokens[:, 7:], past_key_values=middle.past_key_values, use_cache=True)
    actual = torch.cat((prefill.logits, middle.logits, final.logits), 1)
    torch.testing.assert_close(actual, expected, atol=3e-6, rtol=3e-5)
    for layer in final.past_key_values.layers:
        assert layer.keys.shape == (1, 1, 9, 12)
        assert layer.values.shape == (1, 1, 9, 4)
        assert layer.keys.numel() + layer.values.numel() == 9 * (12 + 4)
    padded = torch.cat((torch.zeros(1, 3, dtype=torch.long), tokens), 1)
    mask = (padded != 0).long()
    positions = (mask.cumsum(-1) - 1).clamp(min=0)
    padded_result = model(padded, attention_mask=mask, position_ids=positions, use_cache=False).logits[:, 3:]
    torch.testing.assert_close(padded_result, expected, atol=3e-6, rtol=3e-5)
    gen_kwargs = dict(max_new_tokens=3, do_sample=False, pad_token_id=0, eos_token_id=None)
    assert torch.equal(model.generate(tokens, use_cache=True, **gen_kwargs), model.generate(tokens, use_cache=False, **gen_kwargs))
    model.save_pretrained(tmp_path)
    reloaded, info = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True,
                                                         local_files_only=True, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    torch.testing.assert_close(reloaded(tokens, use_cache=False).logits, expected, atol=3e-6, rtol=3e-5)
    reloaded.config._attn_implementation = "eager"
    torch.testing.assert_close(reloaded(tokens, use_cache=False).logits, expected, atol=3e-6, rtol=3e-5)
    if family == "qwen3":
        assert reloaded.lm_head.weight.data_ptr() == reloaded.model.embed_tokens.weight.data_ptr()


@torch.no_grad()
def test_linear_absorbed_path_matches_upstream_with_query_compression():
    model = source_model("mimo")
    data = batches()
    acts = get_qkv_calibrate_outputs(model, data)
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = PartialRope(layer.self_attn, acts["key"][i], freqfold=2, collapse=2).eval()
    acts = get_qkv_calibrate_outputs(model, data)
    config = make_export_config(model.config, kv_lora_rank=12, qk_mqa_dim=4, q_lora_rank=8, source_model_type="mimo")
    for i, layer in enumerate(model.model.layers):
        layer.self_attn = LoraQKV(layer.self_attn, acts["query"][i], acts["key"][i], acts["value"][i],
                                  q_lora_rank=8, kv_lora_rank=12, qk_mqa_dim=4, collapse=2, balance_kv_ratio=1.0).eval()
    expected = model(**data[0], use_cache=False).logits
    for i, layer in enumerate(model.model.layers):
        attention = TransMLAAttention(config, i)
        attention.load_state_dict(layer.self_attn.state_dict(), strict=True)
        layer.self_attn = attention.eval()
    torch.testing.assert_close(model(**data[0], use_cache=False).logits, expected, atol=3e-6, rtol=3e-5)


@pytest.mark.parametrize("family", ["qwen3", "mimo"])
def test_local_cli_preserves_tokenizer_and_records_provenance(family, tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    source = tmp_path / "source"
    source_model(family).save_pretrained(source)
    if family == "mimo":
        config = json.loads((source / "config.json").read_text())
        config["model_type"] = "mimo"
        (source / "config.json").write_text(json.dumps(config))
    backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2, "hello": 3, "world": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
    tokenizer.chat_template = "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    tokenizer.save_pretrained(source)
    data = tmp_path / "calibration.jsonl"
    data.write_text("\n".join(json.dumps({"text": "hello world " * 20}) for _ in range(12)))
    output = tmp_path / "converted"
    args = ["--model-path", str(source), "--save-path", str(output), "--local-files-only",
            "--dtype", "fp32", "--calibration-file", str(data), "--cal-nsamples", "2",
            "--cal-batch-size", "1", "--cal-max-seqlen", "8", "--kv-lora-rank", "12",
            "--qk-mqa-dim", "4", "--freqfold", "2"]
    inspected = main(args + ["--inspect"])
    assert not output.exists()
    assert inspected["source"]["model_type"] == family
    report = main(args)
    assert report["training_tokens"] == report["optimizer_steps"] == 0
    assert report["calibration"]["tokens"] == 16
    assert report["calibration"]["sha256"]
    assert json.loads((output / "conversion_report.json").read_text()) == report
    from transformers import AutoTokenizer
    assert AutoTokenizer.from_pretrained(output).chat_template == tokenizer.chat_template
    restored, info = AutoModelForCausalLM.from_pretrained(output, trust_remote_code=True, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"]
    assert torch.isfinite(restored(torch.tensor([[3, 4, 3]]), use_cache=True).logits).all()
    with pytest.raises(SystemExit):
        main(args)  # do not overwrite a converted checkpoint


def test_qwen3_rejects_query_compression_and_invalid_geometry():
    source = source_model("qwen3")
    with pytest.raises(ValueError, match="query LoRA"):
        convert_model(source, batches(), kv_lora_rank=12, qk_mqa_dim=4, freqfold=2, q_lora_rank=8)
    with pytest.raises(ValueError, match="freqfold"):
        convert_model(source, batches(), kv_lora_rank=12, qk_mqa_dim=4, freqfold=3)
    with pytest.raises(ValueError, match="joint non-RoPE"):
        convert_model(source, batches(), kv_lora_rank=100, qk_mqa_dim=4, freqfold=2)
    with pytest.raises(ValueError, match="nonempty"):
        convert_model(source, [], kv_lora_rank=12, qk_mqa_dim=4, freqfold=2)


def test_calibration_hooks_are_removed_after_failed_forward():
    model = source_model("qwen3")
    def fail(*args, **kwargs):
        raise RuntimeError("intentional calibration failure")
    model.forward = fail
    with pytest.raises(RuntimeError, match="intentional"):
        get_qkv_calibrate_outputs(model, batches())
    assert all(not module._forward_hooks for module in model.modules())


@pytest.mark.parametrize("family", ["qwen3", "mimo"])
@torch.no_grad()
def test_bfloat16_conversion_export_and_cache(family, tmp_path):
    source = source_model(family).bfloat16()
    model = convert_model(source, batches(), kv_lora_rank=12, qk_mqa_dim=4, freqfold=2, source_model_type=family)
    tokens = torch.randint(2, 41, (1, 7))
    expected = model(tokens, use_cache=False).logits
    assert expected.dtype == torch.bfloat16 and torch.isfinite(expected).all()
    prefill = model(tokens[:, :4], use_cache=True)
    decoded = model(tokens[:, 4:], past_key_values=prefill.past_key_values, use_cache=True)
    torch.testing.assert_close(decoded.logits, expected[:, 4:], atol=3e-3, rtol=2e-2)
    model.save_pretrained(tmp_path)
    restored = AutoModelForCausalLM.from_pretrained(tmp_path, trust_remote_code=True, torch_dtype="auto")
    actual = restored(tokens, use_cache=False).logits
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=3e-3, rtol=2e-2)


@torch.no_grad()
def test_perplexity_counts_real_eos_when_pad_equals_eos():
    model = source_model("qwen3")
    ids = torch.tensor([[4, 1, 5, 1, 1], [3, 1, 6, 7, 1]])
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]])
    logits = model(ids, attention_mask=mask, use_cache=False).logits[:, :-1]
    losses = torch.nn.functional.cross_entropy(logits.flatten(0, 1), ids[:, 1:].flatten(), reduction="none").view(2, -1)
    valid = mask[:, 1:].bool() & mask[:, :-1].bool()
    expected = (losses[valid].mean()).exp().item()
    actual = evaluate_ppl(model, 1, [{"input_ids": ids, "attention_mask": mask}])
    assert actual == pytest.approx(expected, rel=1e-6)
