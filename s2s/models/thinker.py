"""Qwen3 "thinker": tokenizer + causal LM (+ LoRA) and the prompt layout.

Prompt layout for one user turn (Qwen3 chat template, thinking disabled):

    <|im_start|>system\n{system}{context}{tools}<|im_end|>\n
    <|im_start|>user\n [SPEECH_START] speech embeddings [SPEECH_END] {optional instruction}<|im_end|>\n
    <|im_start|>assistant\n<think>\n\n</think>\n\n {response or <tool_call>...</tool_call>}<|im_end|>

The template is rendered with a text placeholder where the speech goes and
split around it, so the layout always matches the model's own chat template.
"""

from __future__ import annotations

import json

import torch
import transformers
from packaging import version

PLACEHOLDER = "<<<SPEECH>>>"


def _dtype_kwargs(dtype: torch.dtype) -> dict:
    if version.parse(transformers.__version__) >= version.parse("4.56.0"):
        return {"dtype": dtype}
    return {"torch_dtype": dtype}


def ignore_incompatible_torchao() -> None:
    """Let PEFT create LoRA layers when an old torchao is installed.

    Recent PEFT raises ImportError while building *any* LoRA layer if torchao is
    installed but older than it supports (Kaggle images ship torchao 0.10). This
    project never uses torchao-quantised weights, so in that case PEFT is told
    torchao is unavailable. A compatible or absent torchao is left alone.
    """
    import sys

    import peft  # noqa: F401  (loads peft.import_utils)
    from peft import import_utils

    check = getattr(import_utils, "is_torchao_available", None)
    if check is None:
        return
    try:
        check()
        return
    except ImportError:
        pass
    for name, module in list(sys.modules.items()):
        if (name == "peft" or name.startswith("peft.")) and getattr(module, "is_torchao_available", None) is check:
            module.is_torchao_available = lambda: False
    try:  # modules imported later bind the patched name from import_utils
        import peft.tuners.lora.torchao as lora_torchao

        lora_torchao.is_torchao_available = lambda: False
    except ImportError:
        pass


def skip_peft_tp_sharding_without_tp() -> None:
    """Let PEFT load a saved LoRA under torch.distributed (DDP) with transformers 5.0.

    When a process group is initialised, PEFT (0.19) calls
    `_maybe_shard_state_dict_for_tp`, which first imports `EmbeddingParallel`
    from transformers - a name that transformers 5.0.0 does not have - and only
    then skips every layer that is not tensor-parallel. This project never uses
    tensor parallelism, so the function is skipped when no layer carries a TP
    plan / device mesh; otherwise PEFT's original function runs unchanged.
    """
    try:
        from peft.utils import save_and_load
    except ImportError:
        return
    original = getattr(save_and_load, "_maybe_shard_state_dict_for_tp", None)
    if original is None or getattr(original, "_s2s_wrapped", False):
        return

    def maybe_shard(model, state_dict, adapter_name):
        if not any(getattr(m, "_hf_tp_plan", None) is not None and getattr(m, "_hf_device_mesh", None) is not None
                   for m in model.modules()):
            return None  # not tensor-parallel: PEFT's function would change nothing
        return original(model, state_dict, adapter_name)

    maybe_shard._s2s_wrapped = True
    save_and_load._maybe_shard_state_dict_for_tp = maybe_shard


def load_causal_lm(name_or_path: str, dtype: torch.dtype, device: torch.device, attn_implementation: str = "sdpa"):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        name_or_path, attn_implementation=attn_implementation, **_dtype_kwargs(dtype)
    )
    return model.to(device)


class Thinker:
    def __init__(self, model_name: str, device: torch.device, dtype: torch.dtype,
                 lora_dir: str | None = None, new_lora: dict | None = None,
                 merge_lora: bool = False, attn_implementation: str = "sdpa",
                 tokenizer_name: str | None = None):
        from transformers import AutoTokenizer

        self.device = device
        self.dtype = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name or model_name)
        self.prompts = PromptBuilder(self.tokenizer)
        model = load_causal_lm(model_name, dtype, device, attn_implementation)
        for p in model.parameters():
            p.requires_grad_(False)
        if lora_dir or new_lora:
            ignore_incompatible_torchao()
            skip_peft_tp_sharding_without_tp()
        if lora_dir:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, lora_dir, is_trainable=not merge_lora and new_lora is not None)
            if merge_lora:
                model = model.merge_and_unload()
        elif new_lora:
            from peft import LoraConfig, get_peft_model

            model = get_peft_model(model, LoraConfig(
                r=new_lora["r"], lora_alpha=new_lora["alpha"], lora_dropout=new_lora["dropout"],
                target_modules=list(new_lora["target_modules"]), task_type="CAUSAL_LM",
            ))
        # trainable params (LoRA) must be fp32 for fp16 GradScaler training
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.float()
        self.model = model
        cfg = self.base_config
        self.hidden_size = int(cfg.hidden_size)
        self.num_layers = int(cfg.num_hidden_layers)
        tok = self.tokenizer
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.tool_call_start_id = tok.convert_tokens_to_ids("<tool_call>")
        self.tool_call_end_id = tok.convert_tokens_to_ids("</tool_call>")
        self.eos_ids = {self.im_end_id, tok.eos_token_id}
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # ------------------------------------------------------------------ model
    @property
    def base_config(self):
        m = self.model
        return m.get_base_model().config if hasattr(m, "get_base_model") else m.config

    def embed_tokens(self) -> torch.nn.Module:
        return self.model.get_input_embeddings()

    def embed(self, ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens()(ids.to(self.device))

    @torch.no_grad()
    def text_embedding_rms(self) -> float:
        w = self.embed_tokens().weight.float()
        return float(w.pow(2).mean().sqrt())

    def hidden_layer_indices(self, fractions: list[float]) -> list[int]:
        """Map fractions of depth to indices into `outputs.hidden_states` (0 = embeddings)."""
        idx = sorted({max(1, min(self.num_layers, int(round(f * self.num_layers)))) for f in fractions})
        return idx

    def enable_gradient_checkpointing(self) -> None:
        m = self.model
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(m, "config"):
            m.config.use_cache = False

    @staticmethod
    def system_content(system_prompt: str, context: str | None = None) -> str:
        return PromptBuilder.system_content(system_prompt, context)

    @staticmethod
    def parse_tool_call(text: str) -> dict | None:
        try:
            obj = json.loads(text.strip())
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict) or "name" not in obj:
            return None
        args = obj.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                return None
        return {"name": obj["name"], "arguments": args if isinstance(args, dict) else {}}


class PromptBuilder:
    """Tokenizer-only helper (picklable, used in DataLoader workers)."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self._parts_cache: dict = {}
        tok = tokenizer
        self.im_end_id = tok.convert_tokens_to_ids("<|im_end|>")
        self.tool_call_start_id = tok.convert_tokens_to_ids("<tool_call>")
        self.tool_call_end_id = tok.convert_tokens_to_ids("</tool_call>")

    @classmethod
    def from_pretrained(cls, name: str) -> "PromptBuilder":
        from transformers import AutoTokenizer

        return cls(AutoTokenizer.from_pretrained(name))

    def ids(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False)["input_ids"]

    def _render(self, messages: list[dict], tools: list | None, add_generation_prompt: bool) -> str:
        return self.tokenizer.apply_chat_template(
            messages, tools=tools or None, add_generation_prompt=add_generation_prompt,
            enable_thinking=False, tokenize=False,
        )

    @staticmethod
    def system_content(system_prompt: str, context: str | None = None) -> str:
        if context:
            return f"{system_prompt}\n\n{context}"
        return system_prompt

    def prompt_parts(self, system: str, tools: list | None = None, instruction: str | None = None,
                     history: list[dict] | None = None) -> tuple[list[int], list[int]]:
        """(prefix_ids, suffix_ids): speech embeddings go between them.
        `history`: earlier turns as text chat messages ({"role", "content"}), placed before the spoken turn."""
        key = (system, json.dumps(tools, sort_keys=True) if tools else "", instruction or "",
               json.dumps(history) if history else "")
        if key not in self._parts_cache:
            if len(self._parts_cache) > 4096:  # histories make keys unique; keep the cache bounded
                self._parts_cache.clear()
            self._parts_cache[key] = self._build_parts(system, tools, instruction, history)
        return self._parts_cache[key]

    @staticmethod
    def _messages(system: str, history: list[dict] | None, user: str) -> list[dict]:
        return ([{"role": "system", "content": system}] + [dict(m) for m in (history or [])]
                + [{"role": "user", "content": user}])

    def _build_parts(self, system: str, tools: list | None, instruction: str | None,
                     history: list[dict] | None = None) -> tuple[list[int], list[int]]:
        user = PLACEHOLDER + (f"\n{instruction}" if instruction else "")
        text = self._render(self._messages(system, history, user), tools, add_generation_prompt=True)
        before, after = text.split(PLACEHOLDER)
        return self.ids(before), self.ids(after)

    def text_prompt_ids(self, system: str, user_text: str, tools: list | None = None,
                        history: list[dict] | None = None) -> list[int]:
        """Full prompt for a *text* user turn, ending with the assistant header."""
        text = self._render(self._messages(system, history, user_text), tools, add_generation_prompt=True)
        return self.ids(text)

    def user_turn_parts(self) -> tuple[list[int], list[int]]:
        """Token ids to open / close a *follow-up* user speech turn (cache ends at <|im_end|>)."""
        text = self._render([{"role": "system", "content": "x"}, {"role": "user", "content": PLACEHOLDER}],
                            None, add_generation_prompt=True)
        turn = text[text.rfind("<|im_start|>user"):]
        before, after = turn.split(PLACEHOLDER)
        return self.ids("\n" + before), self.ids(after)

    def target_text(self, system: str, tools: list | None, content: str = "",
                    tool_calls: list[dict] | None = None, history: list[dict] | None = None) -> str:
        """Assistant reply exactly as the chat template renders it, ending with <|im_end|>."""
        base = self._messages(system, history, PLACEHOLDER)
        prompt = self._render(base, tools, add_generation_prompt=True)
        msg: dict = {"role": "assistant", "content": content}
        if tool_calls:
            msg["tool_calls"] = [{"type": "function", "function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                                 for c in tool_calls]
        full = self._render(base + [msg], tools, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise RuntimeError("chat template produced an unexpected layout for the assistant turn")
        target = full[len(prompt):]
        end = target.rfind("<|im_end|>")
        return target[: end + len("<|im_end|>")]

    def target_ids(self, system: str, tools: list | None, content: str = "",
                   tool_calls: list[dict] | None = None, history: list[dict] | None = None) -> list[int]:
        return self.ids(self.target_text(system, tools, content, tool_calls, history))

    def conversation_ids(self, messages: list[dict], tools: list | None) -> tuple[list[int], list[int]]:
        """Text fine-tuning sample: (input_ids, labels), labels -100 except on the assistant turns
        after the last user message (e.g. tool call, then the reply once the tool results are in).

        Built from the same pieces the runtime feeds the model (prompt, reply ending in <|im_end|>,
        then "\\n" + the tool-response turn and assistant header), so training matches inference.
        """
        last_user = max(i for i, m in enumerate(messages) if m["role"] == "user")
        first = last_user + 1
        if first >= len(messages) or messages[first]["role"] != "assistant":
            raise ValueError("the conversation must end with assistant turn(s) after the last user message")
        ids = self.ids(self._render(messages[:first], tools, add_generation_prompt=True))
        labels = [-100] * len(ids)
        for i in range(first, len(messages)):
            m = messages[i]
            if m["role"] == "assistant":
                if i > first:  # tool results between the previous assistant turn and this one
                    prompt = self._render(messages[:i], tools, add_generation_prompt=True)
                    seg = prompt[prompt.rfind("<|im_start|>user"):]
                    if "<tool_response>" not in seg:
                        raise ValueError("expected tool results between assistant turns")
                    seg_ids = self.ids("\n" + seg)
                    ids += seg_ids
                    labels += [-100] * len(seg_ids)
                target = self.ids(self._assistant_text(messages[:i], tools, m))
                ids += target
                labels += target
            elif m["role"] != "tool":
                raise ValueError(f"unexpected {m['role']} message after the last user turn")
        return ids, labels

    def _assistant_text(self, base: list[dict], tools: list | None, msg: dict) -> str:
        prompt = self._render(base, tools, add_generation_prompt=True)
        full = self._render(base + [msg], tools, add_generation_prompt=False)
        if not full.startswith(prompt):
            raise RuntimeError("chat template produced an unexpected layout for the assistant turn")
        target = full[len(prompt):]
        return target[: target.rfind("<|im_end|>") + len("<|im_end|>")]

    def tool_response_ids(self, results: list) -> list[int]:
        """Tokens appended after a tool-call turn (cache ends at <|im_end|>) with the tool results."""
        call = {"type": "function", "function": {"name": "f", "arguments": {}}}
        base = [{"role": "system", "content": "x"}, {"role": "user", "content": "u"},
                {"role": "assistant", "content": "", "tool_calls": [call]}]
        tool_msgs = [{"role": "tool", "content": r if isinstance(r, str) else json.dumps(r, ensure_ascii=False)}
                     for r in results]
        text = self._render(base + tool_msgs, None, add_generation_prompt=True)
        first = text.find("<tool_response>")
        start = text.rfind("<|im_start|>user", 0, first)
        return self.ids("\n" + text[start:])
