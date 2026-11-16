# Copyright 2026 OpenAI
# Apache 2.0.

"""Shared configuration helpers for S2P2G adapter experiments.

The prompt contract is intentionally adapter-agnostic:

* ``prompt_template`` is a layout template with exactly one ``{s}`` and one
  ``{y}``.
* ``prompt`` is plain prefix text. It is placed before the speech/text sequence
  by the caller and is never parsed for placeholders.

Training code then maps the resolved ``PromptSpec`` to its own tensor layout.
For example, the non-streaming adapter uses ``prefix + speech + postfix + y``,
while the streaming adapter uses ``prompt + chunk interleave`` where template
mode contributes ``prefix + postfix`` as the front prompt.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple


@dataclass(frozen=True)
class PromptSpec:
    """Resolved prompt layout used by S2P2G training and decoding.

    Attributes:
        mode:
            ``"template"`` when the source was a ``prompt_template`` and
            ``"prefix"`` when the source was a plain ``prompt``.
        prompt:
            Plain text that should be embedded before the streaming chunk
            sequence. In template mode this is ``prefix_prompt + postfix_prompt``.
        prompt_template:
            The original template in template mode. For prefix mode this is a
            compatibility rendering ``prompt + "{s}{y}"`` so older debugging
            code can still display a template-shaped string.
        prefix_prompt / postfix_prompt:
            Text pieces around the speech embedding. Non-streaming construction
            maps these to ``[prefix, speech, postfix, target]``. In prefix mode
            ``postfix_prompt`` is empty.
    """

    mode: str
    prompt: str
    prompt_template: str
    prefix_prompt: str
    postfix_prompt: str
    source: str
    explicit: bool


def _clean_config_dict(cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return cfg if isinstance(cfg, dict) else {}


def _pick_first(candidates: Tuple[Tuple[str, Any], ...]) -> Tuple[Optional[str], Optional[str]]:
    """Return the first non-empty string value and its source name."""

    for source, value in candidates:
        if value is None:
            continue
        if not isinstance(value, str):
            raise TypeError(f"{source} must be a string, got {type(value).__name__}.")
        if value != "":
            return source, value
    return None, None


def split_prompt_template(prompt_template: str) -> Tuple[str, str]:
    """Split ``prefix + {s} + postfix + {y}`` into prefix/postfix text.

    The returned text pieces are later converted to token ids and embedded into
    tensors shaped like ``[L_prompt, D_llm]``. Keeping this parser centralized is
    important because one misplaced ``{y}`` changes which positions receive LM
    loss after the causal shift.
    """

    if not isinstance(prompt_template, str):
        raise TypeError(
            f"prompt_template must be a string, got {type(prompt_template).__name__}."
        )
    if prompt_template.count("{s}") != 1 or prompt_template.count("{y}") != 1:
        raise ValueError(
            "prompt_template must contain exactly one {s} and exactly one {y}."
        )
    s_pos = prompt_template.index("{s}")
    y_pos = prompt_template.index("{y}")
    if s_pos > y_pos:
        raise ValueError("prompt_template must place {s} before {y}.")
    prefix_prompt, after_speech = prompt_template.split("{s}", 1)
    postfix_prompt, suffix = after_speech.split("{y}", 1)
    if suffix:
        raise ValueError("prompt_template must end with {y}.")
    return prefix_prompt, postfix_prompt


def _spec_from_template(prompt_template: str, source: str, explicit: bool) -> PromptSpec:
    prefix_prompt, postfix_prompt = split_prompt_template(prompt_template)
    return PromptSpec(
        mode="template",
        prompt=prefix_prompt + postfix_prompt,
        prompt_template=prompt_template,
        prefix_prompt=prefix_prompt,
        postfix_prompt=postfix_prompt,
        source=source,
        explicit=explicit,
    )


def _spec_from_prefix(prompt: str, source: str, explicit: bool) -> PromptSpec:
    if not isinstance(prompt, str):
        raise TypeError(f"{source} must be a string, got {type(prompt).__name__}.")
    if "{s}" in prompt or "{y}" in prompt:
        raise ValueError(
            f"{source} contains {{s}}/{{y}} placeholders. Use prompt_template for "
            "layout templates, or remove the placeholders for a plain prefix prompt."
        )
    return PromptSpec(
        mode="prefix",
        prompt=prompt,
        prompt_template=prompt + "{s}{y}",
        prefix_prompt=prompt,
        postfix_prompt="",
        source=source,
        explicit=explicit,
    )


def resolve_prompt_spec(
    cfg: Dict[str, Any],
    streaming_train_config: Optional[Dict[str, Any]] = None,
    default_prompt: Optional[str] = None,
) -> PromptSpec:
    """Resolve prompt fields with one semantic rule across all adapters.

    Canonical fields live under ``prompt_config``:
        ``prompt_config.prompt_template`` for template mode and
        ``prompt_config.prompt`` for prefix mode.

    Legacy fields are still read so older experiment configs keep working.
    A legacy ``cfg.prompt`` that already contains ``{s}``/``{y}`` is treated as a
    template for compatibility, but canonical ``prompt_config.prompt`` remains a
    strict plain prefix.
    """

    cfg = _clean_config_dict(cfg)
    prompt_cfg = _clean_config_dict(cfg.get("prompt_config"))
    streaming_cfg = _clean_config_dict(
        streaming_train_config if streaming_train_config is not None else cfg.get("streaming_train_config")
    )

    template_source, prompt_template = _pick_first(
        (
            ("prompt_config.prompt_template", prompt_cfg.get("prompt_template")),
            ("prompt_config.template", prompt_cfg.get("template")),
            ("cfg.prompt_template", cfg.get("prompt_template")),
            ("streaming_train_config.prompt_template", streaming_cfg.get("prompt_template")),
        )
    )

    prompt_source, prompt = _pick_first(
        (
            ("prompt_config.prompt", prompt_cfg.get("prompt")),
            ("cfg.prompt", cfg.get("prompt")),
            ("streaming_train_config.prompt", streaming_cfg.get("prompt")),
        )
    )

    if prompt_template is not None and prompt is not None:
        raise ValueError(
            "Configure only one prompt style: prompt_config.prompt_template for "
            "layout templates, or prompt_config.prompt for a plain prefix prompt."
        )

    if prompt_template is not None:
        return _spec_from_template(prompt_template, template_source or "prompt_template", True)

    if prompt is not None:
        # Backward compatibility: old top-level/streaming ``prompt`` sometimes
        # carried a full template. Keep that behavior outside canonical
        # ``prompt_config.prompt`` so new configs stay unambiguous.
        if prompt_source != "prompt_config.prompt" and ("{s}" in prompt or "{y}" in prompt):
            return _spec_from_template(prompt, prompt_source or "legacy prompt", True)
        return _spec_from_prefix(prompt, prompt_source or "prompt", True)

    if default_prompt is None:
        default_prompt = ""
    if "{s}" in default_prompt or "{y}" in default_prompt:
        return _spec_from_template(default_prompt, "default_prompt", False)
    return _spec_from_prefix(default_prompt, "default_prompt", False)


def parse_llm_left_context_chunks(value: Any) -> Optional[int]:
    """Parse LLM left-context chunk count.

    Returns:
        ``None`` for full left context (configured as ``"all"``), otherwise a
        non-negative integer number of previous chunks visible to the current
        chunk. Callers use ``None`` to select the standard 2D padding mask when
        HuggingFace's internal decoder-only causal mask is sufficient.
    """

    if isinstance(value, str):
        value = value.strip()
        if value.lower() == "all":
            return None
        parsed = int(value)
    elif value is None:
        return None
    else:
        parsed = int(value)
    if parsed < 0:
        raise ValueError(f"llm_left_context_chunks must be >= 0 or 'all', got {value!r}.")
    return parsed


def resolve_decode_prompt_override(
    cfg: Dict[str, Any],
    default_prompt: Optional[str] = None,
) -> Optional[PromptSpec]:
    """Return a decode-time prompt override, if the config explicitly has one.

    Decode defaults to the prompt reconstructed by the training config. Only a
    nested ``decode.prompt_config`` or direct ``decode.prompt`` /
    ``decode.prompt_template`` should override that model-level prompt.
    """

    cfg = _clean_config_dict(cfg)
    decode_cfg = _clean_config_dict(cfg.get("decode"))
    if not decode_cfg:
        decode_cfg = _clean_config_dict(cfg.get("decode_config"))
    if not decode_cfg:
        return None

    prompt_cfg = _clean_config_dict(decode_cfg.get("prompt_config"))
    has_prompt_fields = any(
        key in prompt_cfg for key in ("prompt", "prompt_template", "template")
    ) or any(key in decode_cfg for key in ("prompt", "prompt_template", "template"))
    if not has_prompt_fields:
        return None
    return resolve_prompt_spec(decode_cfg, default_prompt=default_prompt)
