"""Reasoning-control translation: what maps exactly, what is approximated, what is lost.

Each control gets one of three outcomes, decided from the request, the destination
protocol and the route's capability facts (never from model names):

exact        same semantics: thinking on/off, a token budget on both sides, a Chat
             effort and a Responses effort (one enum). Applied silently.
approximated a mapping exists but means something different: an effort level and a
             token budget, or Anthropic effort (which governs all output) and OpenAI
             effort (reasoning only). Reported as ``approximated``.
unsupported  the destination has no such control. Reported as ``dropped``.

Strict policy rejects both reported outcomes; lossy policy applies them and reports.
Level names are never renamed (``minimal`` stays ``minimal``, never becomes ``low``),
and a value outside the route's known effort values is never clamped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .common import RouterError
from .constants import (
    DIALECT_ANTHROPIC,
    DIALECT_GEMINI,
    DIALECT_OPENAI_CHAT,
    DIALECT_OPENAI_RESPONSES,
    LLAMA_THINKING_BUDGETS,
)

# Level names that every effort-bearing protocol spells the same way. Used only
# when a route publishes no effort values of its own.
_COMMON_EFFORTS = ("low", "medium", "high")
# The levels LiteLLM's Gemini adapter can express (token budget or thinking level).
_GEMINI_EFFORTS = ("minimal", "low", "medium", "high")
# Levels a thinking-level Gemini model spells exactly as the Chat effort does.
# "minimal" is excluded: the adapter turns it into "low" on models without it.
_GEMINI_NATIVE_LEVELS = ("low", "medium", "high")


@dataclass
class ReasoningPlan:
    """Controls to remove from the ingress params and the native ones to send instead."""

    strip: list[str] = field(default_factory=list)
    losses: list[dict[str, str]] = field(default_factory=list)
    # Keyword arguments for the LiteLLM entry point (completion/responses/messages handler).
    completion: dict[str, Any] = field(default_factory=dict)
    # Raw request-body fields for the upstream (llama.cpp's native thinking controls).
    body: dict[str, Any] = field(default_factory=dict)
    # Which ingress field each completion keyword came from, for loss reports.
    origins: dict[str, str] = field(default_factory=dict)

    def lose(self, path: str, reason: str, *, approximate: bool = False, strip: bool = True) -> None:
        self.losses.append({"field": path, "reason": reason,
                            "action": "approximated" if approximate else "dropped"})
        if strip and not approximate:
            self.strip.append(path)


def plan_reasoning(
    ingress: str, target: str, llama: bool, params: dict[str, Any],
    fact: Callable[[str], Any], *, gemini_levels: bool = False,
) -> ReasoningPlan:
    """Plan reasoning translation for one request; ``params`` is not modified.

    ``gemini_levels`` says the Gemini adapter sends an effort as a native thinking
    level for this route rather than as a token budget derived from it.
    """
    plan = ReasoningPlan()
    extension = (params.get("extra_body") or {}).get("thinking")
    if ingress == DIALECT_ANTHROPIC:
        thinking, t_path = params.get("thinking"), "thinking"
        effort, e_path = None, ""
        scope_effort = (params.get("output_config") or {}).get("effort")
    elif ingress == DIALECT_OPENAI_CHAT:
        thinking, t_path = extension, "extra_body.thinking"
        effort, e_path = params.get("reasoning_effort"), "reasoning_effort"
        scope_effort = None
    else:
        thinking, t_path = extension, "extra_body.thinking"
        effort = (params.get("reasoning") or {}).get("effort")
        e_path = "reasoning.effort"
        scope_effort = None
    reasoning = params.get("reasoning") if isinstance(params.get("reasoning"), dict) else {}

    if effort is not None and thinking is not None:
        # Two explicit thinking controls cannot both be honored; never pick one silently.
        raise RouterError(400, "Unsupported or invalid reasoning for translation: "
                          "reasoning effort cannot be combined with a thinking object",
                          code="invalid_generation_parameter")

    # The Responses-only reasoning options have no counterpart outside Responses.
    for key in ("summary", "generate_summary"):
        if key in reasoning:
            plan.lose(f"reasoning.{key}",
                      "reasoning summaries are a Responses-only control; this destination "
                      "cannot be asked for them")
    if isinstance(thinking, dict) and "display" in thinking and target != DIALECT_ANTHROPIC:
        plan.lose(f"{t_path}.display",
                  "thinking display has no equivalent control; the destination's default applies",
                  strip=False)
        # The key is removed from a copy below; nothing else carries it.
        plan.strip.append(f"{t_path}.display")

    disable = effort == "none" or (isinstance(thinking, dict) and thinking.get("type") == "disabled")
    controls = [(e_path, effort)] if effort is not None else []
    if thinking is not None:
        controls.append((t_path, thinking))
    if scope_effort is not None:
        controls.append(("output_config.effort", scope_effort))
    if not controls:
        return plan

    def emit_effort(value: str, origin: str) -> None:
        responses = ingress == DIALECT_OPENAI_RESPONSES or (
            ingress == DIALECT_ANTHROPIC and target == DIALECT_OPENAI_RESPONSES)
        if responses:
            plan.completion["reasoning"] = {"effort": value}
            plan.origins["reasoning"] = origin
        else:
            plan.completion["reasoning_effort"] = value
            plan.origins["reasoning_effort"] = origin

    def own_effort_path() -> None:
        # The ingress effort is replaced by an explicit native value (or removed).
        # Responses carries it inside `reasoning`, whose other keys are handled apart.
        if effort is not None:
            plan.strip.append("reasoning" if e_path == "reasoning.effort" else e_path)

    def drop(path: str, reason: str) -> None:
        plan.lose(path, reason, strip=False)
        plan.strip.append("reasoning" if path == "reasoning.effort" else path)

    if fact("reasoning.supported") is False:
        for path, _ in controls:
            if disable:
                # The client asked for no thinking and the model does not think.
                plan.strip.append("reasoning" if path == "reasoning.effort" else path)
            else:
                drop(path, "model capability is unsupported")
        return plan

    values = fact("reasoning.effort.values")
    effort_off = fact("reasoning.effort.supported") is False
    # Anthropic effort can be carried only to a Chat/Responses destination, by name.
    carries_scope = (
        scope_effort is not None and not effort_off and not llama
        and target in (DIALECT_OPENAI_CHAT, DIALECT_OPENAI_RESPONSES)
        and scope_effort in (values if values is not None else _COMMON_EFFORTS))

    # -- thinking disabled --------------------------------------------------
    if disable:
        path = e_path if effort is not None else t_path
        if fact("reasoning.toggle.supported") is False:
            drop(path, "model cannot disable thinking")
        elif target == DIALECT_ANTHROPIC:
            plan.strip.append("reasoning" if path == "reasoning.effort" else path)
            native = fact("reasoning.toggle.native_parameter") or {}
            disabled = (native.get("values") or {}).get("disabled", "disabled")
            plan.completion["thinking"] = {"type": disabled}
            plan.origins["thinking"] = path
        elif llama:
            plan.strip.append("reasoning" if path == "reasoning.effort" else path)
            plan.body["chat_template_kwargs"] = {"enable_thinking": False}
        elif target in (DIALECT_OPENAI_CHAT, DIALECT_OPENAI_RESPONSES):
            if effort_off or (values is not None and "none" not in values):
                drop(path, "model's known effort values cannot express 'none'")
            else:
                plan.strip.append("reasoning" if path == "reasoning.effort" else path)
                emit_effort("none", path)
        elif target == DIALECT_GEMINI:
            plan.strip.append("reasoning" if path == "reasoning.effort" else path)
            emit_effort("none", path)
            if fact("reasoning.toggle.supported") is not True:
                plan.losses.append({
                    "field": path, "action": "approximated",
                    "reason": "some Gemini models cannot fully disable thinking; "
                              "the adapter picks the lowest level they allow"})
        if scope_effort is not None:
            plan.lose("output_config.effort", "conflicts with disabled thinking")
        return plan

    # -- Anthropic thinking object (enabled / adaptive) ---------------------
    if isinstance(thinking, dict):
        kind = thinking.get("type")
        budget = thinking.get("budget_tokens")
        if kind == "adaptive":
            if target != DIALECT_ANTHROPIC:
                plan.lose(t_path, "adaptive thinking has no equivalent; the destination's "
                                  "default applies")
            else:
                plan.losses.append({
                    "field": t_path, "action": "approximated",
                    "reason": "adaptive thinking may become a token budget on models without it"})
        elif kind == "enabled":
            if fact("reasoning.budget.supported") is False:
                plan.lose(t_path, "model budget control is unsupported")
            elif ((fact("reasoning.budget.min_tokens") is not None
                   and budget < fact("reasoning.budget.min_tokens"))
                  or (fact("reasoning.budget.max_tokens") is not None
                      and budget > fact("reasoning.budget.max_tokens"))):
                plan.lose(t_path, "budget is outside the model's known bounds")
            elif carries_scope:
                # One destination effort cannot carry both a budget and a level.
                plan.lose(t_path, "superseded by output_config.effort on this destination")
            elif target == DIALECT_ANTHROPIC:
                pass  # native; LiteLLM forwards it unchanged
            elif llama:
                plan.strip.append(t_path)
                plan.body["thinking_budget_tokens"] = budget
            elif target == DIALECT_GEMINI:
                plan.strip.append(t_path)
                plan.completion["thinking"] = {"type": "enabled", "budget_tokens": budget}
                plan.origins["thinking"] = t_path
            elif ingress == DIALECT_ANTHROPIC:
                # LiteLLM buckets the token budget into an effort level.
                plan.losses.append({
                    "field": t_path, "action": "approximated",
                    "reason": "token budget becomes an approximate effort level"})
            else:
                plan.lose(t_path, "native thinking control requires a native route")

    # -- effort requested by Chat/Responses ----------------------------------
    if effort is not None:
        if effort_off:
            drop(e_path, "model capability is unsupported")
        elif values is not None and effort not in values:
            drop(e_path, "effort is outside the model's known values")
        elif llama:
            if effort in LLAMA_THINKING_BUDGETS:
                own_effort_path()
                plan.body["thinking_budget_tokens"] = LLAMA_THINKING_BUDGETS[effort]
            else:
                drop(e_path, "llama.cpp has no control for this effort level")
        elif target == DIALECT_ANTHROPIC:
            plan.losses.append({
                "field": e_path, "action": "approximated",
                "reason": "Anthropic has no effort level equal to this one; the adapter maps it "
                          "to a token budget or an adaptive effort depending on the model"})
        elif target == DIALECT_GEMINI:
            if gemini_levels and values is not None and effort in _GEMINI_NATIVE_LEVELS:
                # The route publishes this level and the adapter sends it by name
                # as thinkingLevel: same semantics on both sides.
                pass
            elif effort in _GEMINI_EFFORTS:
                plan.losses.append({
                    "field": e_path, "action": "approximated",
                    "reason": "Gemini has no effort levels; the adapter maps this one to a "
                              "thinking budget or level depending on the model"})
            else:
                drop(e_path, "Gemini has no control for this effort level")

    # -- Anthropic effort ------------------------------------------------------
    if scope_effort is not None:
        if effort_off:
            plan.lose("output_config.effort", "model capability is unsupported")
        elif target == DIALECT_ANTHROPIC:
            pass
        elif carries_scope:
            plan.strip.append("output_config.effort")
            emit_effort(scope_effort, "output_config.effort")
            plan.losses.append({
                "field": "output_config.effort", "action": "approximated",
                "reason": "Anthropic effort governs all output while OpenAI effort governs "
                          "reasoning only; only an identically named level is carried"})
        else:
            plan.lose("output_config.effort", "the destination has no identically named "
                                              "effort level or no effort control")
    return plan
