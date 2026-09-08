"""HF generation with explicit sampling and vLLM-compatible completion fields."""
from copy import deepcopy
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Sampling:
    n: int = 1
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = -1
    max_tokens: int = 128

    def __post_init__(self):
        if self.n < 1 or self.max_tokens < 1 or self.temperature < 0 or not 0 < self.top_p <= 1:
            raise ValueError("Invalid generation settings")


def decode_completion(tokens, tokenizer, eos_ids, cap):
    ids = tokens.tolist()
    finish = "length"
    for i, token in enumerate(ids):
        if token in eos_ids:
            ids = ids[:i + 1]
            finish = "stop"
            break
    if finish == "length" and len(ids) != cap:
        raise RuntimeError("Generation stopped without EOS before its token budget")
    return {"text": tokenizer.decode(ids, skip_special_tokens=True), "token_ids": ids,
            "finish_reason": finish, "gen_tokens": len(ids), "truncated": finish == "length"}


class HFGenerator:
    def __init__(self, model, tokenizer):
        self.model, self.tokenizer = model, tokenizer

    @torch.inference_mode()
    def generate(self, prompt, sampling, seed):
        inputs = self.tokenizer(prompt, add_special_tokens=False, return_token_type_ids=False,
                                return_tensors="pt").to(self.model.device)
        context = self.model.config.max_position_embeddings
        if inputs.input_ids.shape[1] + sampling.max_tokens > context:
            raise ValueError(f"Prompt plus generation exceeds native context {context}; truncation is forbidden")
        config = deepcopy(self.model.generation_config)
        config.update(do_sample=sampling.temperature > 0, num_beams=1,
                      num_return_sequences=sampling.n if sampling.temperature > 0 else 1,
                      max_new_tokens=sampling.max_tokens, min_new_tokens=0,
                      temperature=sampling.temperature if sampling.temperature > 0 else 1.0,
                      top_p=sampling.top_p if sampling.temperature > 0 else 1.0,
                      top_k=max(0, sampling.top_k) if sampling.temperature > 0 else 0,
                      repetition_penalty=1.0, no_repeat_ngram_size=0, length_penalty=1.0,
                      typical_p=1.0, min_p=None, penalty_alpha=None,
                      renormalize_logits=False, forced_bos_token_id=None, forced_eos_token_id=None,
                      suppress_tokens=None, begin_suppress_tokens=None,
                      use_cache=True, cache_implementation="dynamic", return_dict_in_generate=False,
                      output_scores=False, pad_token_id=self.tokenizer.pad_token_id)
        eos = config.eos_token_id
        eos_ids = set(eos if isinstance(eos, list) else [eos]) - {None}
        torch.manual_seed(seed)
        if self.model.device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        output = self.model.generate(**inputs, generation_config=config)
        completions = [decode_completion(row[inputs.input_ids.shape[1]:], self.tokenizer, eos_ids,
                                          sampling.max_tokens) for row in output]
        if sampling.temperature == 0 and sampling.n > 1:
            completions = completions * sampling.n
        return {"prompt_tokens": inputs.input_ids.shape[1], "eos_token_ids": sorted(eos_ids),
                "seed": seed, "outputs": completions}
