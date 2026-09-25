// omp's own model resolver: which ollama model does a role selector resolve to, given the installed ollama ids?
// Consumer: localbench/park.py (`fallbacks`), which parks every model omp would hand a smol role while its target is
// parked. Calling omp's resolver (config/model-resolver.ts `resolveModelFromString`) instead of re-deriving it means
// a new omp release answers for itself: on 2026-09-23 it returned qwen3.8-uncensored:latest for
// `ollama/qwen3.8:27b-mlx`, a pair its fuzzy matcher scores as no match (scripts/omp-fuzzy-probe.ts).
// Run: bun run scripts/omp-resolve.ts <pi-coding-agent package dir> <selector> [ollama id ...]
// Prints one line: {"picked": "<ollama id>"} or {"picked": null}.
const [pkg, selector, ...ids] = process.argv.slice(2);
if (!pkg || !selector) {
	console.error("usage: omp-resolve.ts <pi-coding-agent package dir> <selector> [ollama id ...]");
	process.exit(2);
}
const { resolveModelFromString } = await import(`${pkg}/src/config/model-resolver.ts`);
const models = ids.map(id => ({
	provider: "ollama",
	id,
	name: id,
	api: "openai-responses",
	baseUrl: "http://127.0.0.1:11434/v1",
	reasoning: false,
	input: ["text"],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 131072,
	maxTokens: 8192,
}));
const m = resolveModelFromString(selector, models, undefined);
console.log(JSON.stringify({ picked: m?.provider === "ollama" ? m.id : null }));
