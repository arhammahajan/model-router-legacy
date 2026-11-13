// Auto-discovers models from a local litellm-model-router and registers them
// as the "model-router" provider. Replaces a hand-written models.json entry.
//
// Includes local llama.cpp and Strata engines discovered by the router.
//
//   MODEL_ROUTER_URL      router origin (default http://127.0.0.1:4000)
//   MODEL_ROUTER_API_KEY  sent as the API key when the router requires one
//
// Each model is reached with its upstream's native protocol so the router can
// pass requests through untouched (no translation loss).
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

const ORIGIN = (process.env.MODEL_ROUTER_URL ?? "http://127.0.0.1:4000").replace(/\/+$/, "");
const PROVIDER = "model-router";

const val = (x: any) => x?.value ?? undefined;

// Pi offers minimal..high unless a level is marked unsupported (null), and
// xhigh/max only when they are mapped. Offer exactly the efforts the route
// publishes, so Pi never sends a level the router would have to refuse.
function thinkingLevels(values: unknown, canDisable: boolean): Record<string, string | null> | undefined {
  if (!Array.isArray(values)) return undefined;
  const map: Record<string, string | null> = {};
  for (const level of ["minimal", "low", "medium", "high"]) {
    if (!values.includes(level)) map[level] = null;
  }
  for (const level of ["xhigh", "max"]) {
    if (values.includes(level)) map[level] = level;
  }
  // Without a mapping pi sends nothing for "off" and the upstream keeps thinking.
  if (values.includes("none")) map.off = "none";
  else if (!canDisable) map.off = null;
  return map;
}

async function discover(signal?: AbortSignal): Promise<any[]> {
  const timeout = AbortSignal.timeout(5000);
  const res = await fetch(`${ORIGIN}/v1/models`, {
    signal: signal ? AbortSignal.any([signal, timeout]) : timeout,
    headers: process.env.MODEL_ROUTER_API_KEY
      ? { Authorization: `Bearer ${process.env.MODEL_ROUTER_API_KEY}` }
      : {},
  });
  if (!res.ok) throw new Error(`router /v1/models returned ${res.status}`);
  const body = (await res.json()) as { data?: any[] };

  return (body.data ?? []).map((m) => {
    const md = m.metadata ?? {};
    const caps = md.router ?? {};
    const dialect: string = md.dialect ?? "openai-chat";
    const isLlama = !!md.llama_cpp;
    const isStrata = md.engine === "strata";
    const name = m.display_name ?? md.display_name ?? (isLlama && !isStrata ? `${m.id} (llama.cpp)` : m.id);

    const model: any = {
      id: m.id,
      name: isStrata && !/\(strata\)$/i.test(name) ? `${name} (Strata)` : name,
      // Local engines advertise effort controls independently of whether the
      // model reports that it produces reasoning.
      reasoning:
        val(caps.reasoning?.supported) === true || val(caps.reasoning?.effort?.supported) === true,
      input: val(caps.modalities?.input?.image) ? ["text", "image"] : ["text"],
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
      // Local engines report these at the top level; other upstreams only under
      // router.limits. null means unknown, so fall back to pi's defaults.
      contextWindow: md.context_window ?? val(caps.limits?.context_window) ?? 128000,
      maxTokens: md.max_tokens ?? val(caps.limits?.max_output_tokens) ?? 16384,
    };

    const efforts = val(caps.reasoning?.effort?.values);
    // Apply the same published vocabulary on every protocol. Without a known
    // vocabulary Pi would invent minimal/low/medium/high, so leave reasoning
    // controls to the upstream's defaults for those models.
    const levels = model.reasoning
      ? thinkingLevels(efforts, val(caps.reasoning?.toggle?.supported) === true)
      : undefined;
    if (levels) model.thinkingLevelMap = levels;
    if (!Array.isArray(efforts) || efforts.length === 0) {
      model.reasoning = false;
    }

    if (dialect === "anthropic") {
      model.api = "anthropic-messages";
      model.baseUrl = ORIGIN;
      if (/^claude-/.test(m.id)) model.compat = { forceAdaptiveThinking: true };
    } else if (dialect === "openai-responses") {
      model.api = "openai-responses";
      model.baseUrl = `${ORIGIN}/v1`;
    }
    return model;
  });
}

export default async function (pi: ExtensionAPI) {
  // Pi's Chat adapter does not send a conversation header by default. Reuse
  // the durable Pi session ID so the router can provide upstream affinity.
  pi.on("before_provider_headers", (event, ctx) => {
    if (ctx.model?.provider === PROVIDER) {
      event.headers["x-session-id"] = ctx.sessionManager.getSessionId();
    }
  });

  let models: any[] = [];
  try {
    models = await discover();
  } catch {
    // Router not running yet; models appear after the next refresh.
  }

  pi.registerProvider(PROVIDER, {
    name: "Model Router",
    baseUrl: `${ORIGIN}/v1`,
    apiKey: process.env.MODEL_ROUTER_API_KEY || "unused",
    api: "openai-completions",
    models,
    refreshModels: ({ signal }) => discover(signal),
  } as any);
}
