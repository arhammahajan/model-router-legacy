// Drives the installed pi client against a running (test-local) router.
//
//   node --experimental-strip-types pi_harness.mjs <pi-ai dir> <extension.ts>
//
// MODEL_ROUTER_URL points at the router. The extension registers its models
// through a stand-in ExtensionAPI; for every model and every thinking level pi
// offers, pi's own request builder runs (nothing is reimplemented here) and the
// request goes over HTTP to the router. Prints one JSON record per request.
import { pathToFileURL } from "node:url";
import { join } from "node:path";

const [piAi, extensionPath] = process.argv.slice(2);
const load = (rel) => import(pathToFileURL(join(piAi, "dist", rel)).href);
const { getSupportedThinkingLevels } = await load("models.js");
const apis = {
  "openai-completions": (await load("api/openai-completions.js")).streamSimple,
  "openai-responses": (await load("api/openai-responses.js")).streamSimple,
  "anthropic-messages": (await load("api/anthropic-messages.js")).streamSimple,
};

const hooks = {};
let provider;
const { default: register } = await import(pathToFileURL(extensionPath).href);
await register({
  on: (event, handler) => { hooks[event] = handler; },
  registerProvider: (name, config) => { provider = { name, config }; },
});

const SESSION = "pi-session-0001";
const records = [];

for (const entry of provider.config.models) {
  const api = entry.api ?? provider.config.api;
  const model = {
    ...entry, api, provider: provider.name,
    baseUrl: entry.baseUrl ?? provider.config.baseUrl,
  };
  const levels = getSupportedThinkingLevels(model);
  for (const level of levels) {
    let seen;
    const fetchAndRecord = async (input, init) => {
      const url = typeof input === "string" ? input : input.url;
      const headers = Object.fromEntries(new Headers(init?.headers ?? input.headers).entries());
      // pi-coding-agent runs this hook on every provider request.
      const event = { headers };
      hooks.before_provider_headers?.(event, {
        model: { provider: provider.name },
        sessionManager: { getSessionId: () => SESSION },
      });
      const response = await fetch(url, { ...init, headers: event.headers });
      seen = {
        path: new URL(url).pathname,
        status: response.status,
        loss: response.headers.get("x-model-router-loss"),
        request: JSON.parse(init.body),
        headers: event.headers,
        error: response.ok ? null : await response.clone().text(),
      };
      return response;
    };
    const stream = apis[api](model, {
      systemPrompt: "system",
      messages: [{ role: "user", content: "hello", timestamp: Date.now() }],
    }, {
      apiKey: provider.config.apiKey,
      reasoning: level === "off" ? undefined : level,
      maxTokens: 64,
      sessionId: SESSION,
      fetch: fetchAndRecord,
    });
    const result = await stream.result();
    records.push({ model: entry.id, api, level, ...seen, stop: result.stopReason, text: JSON.stringify(result.content) });
  }
}
console.log(JSON.stringify({ models: provider.config.models, records }));
