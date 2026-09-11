"""CPU regressions for full-weight GPU validation logic; these do not release jobs."""
from unittest.mock import patch

import pytest
import torch

from test_pretrained_conversion import source_model, batches
from transmla.convert_pretrained import convert_model
from transmla.experiments.validate import (
    FP32_ATOL, FP32_RTOL, check_bf16_runtime, check_fp32_model, check_model, export_reference, run_validation,
)


@pytest.fixture(autouse=True)
def deterministic():
    torch.manual_seed(19)
    torch.set_num_threads(2)


@pytest.mark.parametrize("family", ["qwen3", "mimo"])
@pytest.mark.parametrize("converted", [False, True])
def test_fp32_reference_checks_cache_and_padding(family, converted):
    model = source_model(family)
    if converted:
        model = convert_model(model, batches(), kv_lora_rank=12, qk_mqa_dim=4,
                              freqfold=2, source_model_type=family)
    report, reference = check_fp32_model(model, torch.randint(2, 41, (1, 64)), converted)
    assert reference.shape == (1, 64, 41)
    assert report["cached_max_abs_error"] < 1e-4
    assert report["padded_max_abs_error"] < 1e-4
    if converted:
        assert report["expanded_math_max_abs_error"] < 1e-4


def test_fp32_gate_rejects_a_cache_specific_error_below_the_old_bf16_tolerance():
    model = source_model("qwen3")
    forward = model.forward
    def corrupt(*args, **kwargs):
        output = forward(*args, **kwargs)
        if kwargs.get("past_key_values") is not None:
            output.logits = output.logits + 0.01
        return output
    prior = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    with patch.object(model, "forward", side_effect=corrupt), pytest.raises(AssertionError):
        check_fp32_model(model, torch.randint(2, 41, (1, 64)))
    assert (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32) == prior


def test_fp32_gate_cannot_silently_use_bf16():
    with pytest.raises(ValueError, match="FP32"):
        check_fp32_model(source_model("qwen3").bfloat16(), torch.randint(2, 41, (1, 64)))


def test_export_reference_matches_reload_backend_and_restores_runtime_backend():
    model = convert_model(source_model("qwen3"), batches(), kv_lora_rank=12,
                          qk_mqa_dim=4, freqfold=2, source_model_type="qwen3")
    ids = torch.randint(2, 41, (1, 64))
    with torch.inference_mode():
        model.config.mla_prefill_backend = "chunked"
        expected = model(ids, use_cache=False).logits
    model.config.mla_prefill_backend = "auto"
    result = export_reference(model, ids)
    assert result["prefill_backend"] == "chunked"
    torch.testing.assert_close(result["logits"], expected, atol=0, rtol=0)
    assert model.config.mla_prefill_backend == "auto"
    with patch.object(model, "forward", side_effect=RuntimeError("probe error")), pytest.raises(RuntimeError):
        export_reference(model, ids)
    assert model.config.mla_prefill_backend == "auto"


def test_bf16_checkpoint_roundtrip_uses_fp32_reference(tmp_path):
    model = convert_model(source_model("qwen3"), batches(), kv_lora_rank=12,
                          qk_mqa_dim=4, freqfold=2, source_model_type="qwen3")
    # Match HF's BF16 parameter loading while keeping RoPE buffers in FP32.
    for parameter in model.parameters():
        parameter.data = parameter.data.bfloat16()
    model.save_pretrained(tmp_path)
    ids = torch.randint(2, 41, (1, 64))
    before = export_reference(model, ids)
    assert before["dtype"] == "float32" and before["tf32"] is False
    assert before["logits"].dtype == torch.float32
    reloaded = type(model).from_pretrained(tmp_path, torch_dtype=torch.bfloat16).eval()
    assert reloaded.dtype == torch.bfloat16
    _, after = check_fp32_model(reloaded.float(), ids, converted=True)
    torch.testing.assert_close(after, before["logits"], atol=FP32_ATOL, rtol=FP32_RTOL)


def test_runtime_diagnostics_still_reject_nonfinite_cache_logits():
    model = source_model("qwen3")
    forward = model.forward
    def corrupt(*args, **kwargs):
        output = forward(*args, **kwargs)
        if kwargs.get("past_key_values") is not None:
            output.logits.fill_(float("nan"))
        return output
    with patch.object(model, "forward", side_effect=corrupt), pytest.raises(ValueError, match="Nonfinite"):
        check_model(model, torch.randint(2, 41, (1, 64)), enforce_parity=False)


@pytest.mark.parametrize("name", ["cached", "padded", "prefill"])
def test_bf16_gate_rejects_large_distribution_drift(name):
    report = {key: {"max_kl": 0.02} for key in ("cached", "padded", "prefill")}
    # Actual failed MiMo prefill comparison, previously only recorded.
    report[name]["max_kl"] = 0.6673276424407959
    with pytest.raises(ValueError, match=f"BF16 {name}"):
        check_bf16_runtime(report, converted=True)


def test_bf16_gate_rejects_invalid_kl_and_missing_prefill():
    report = {"cached": {"max_kl": 0.0}, "padded": {"max_kl": float("nan")}}
    with pytest.raises(ValueError, match="BF16 padded"):
        check_bf16_runtime(report)
    report["padded"]["max_kl"] = 0.02
    check_bf16_runtime(report)
    with pytest.raises(KeyError, match="prefill"):
        check_bf16_runtime(report, converted=True)


def test_excessive_bf16_drift_cannot_write_a_passed_gpu_gate(tmp_path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    (model_path / "config.json").write_text('{"model_type": "transmla"}')
    gate = tmp_path / "gate.json"
    report = {"cached": {"max_kl": 0.0}, "padded": {"max_kl": 0.004},
              "prefill": {"max_kl": 0.6673276424407959}}
    with patch("torch.cuda.is_available", return_value=True), \
            patch("torch.cuda.reset_peak_memory_stats"), \
            patch("transmla.experiments.validate.load_model", return_value=(object(), object())), \
            patch("transmla.experiments.validate.probe_tokens", return_value=torch.ones(1, 64)), \
            patch("transmla.experiments.validate.check_model", return_value=(report, None)), \
            pytest.raises(ValueError, match="BF16 prefill"):
        run_validation(model_path, tmp_path, gate, "converted")
    assert not gate.exists()
