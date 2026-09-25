// Which ollama model ids would omp's provider-scoped fuzzy resolver pick for a smol selector?
// Consumer: the park.py ledger row (2026-09-22) retry predicate; anyone choosing a parked-model name.
// Run: bun run scripts/omp-fuzzy-probe.ts [query] [candidate ...]
// Imports pi-tui from the omp install the user runs (~/.bun/bin/omp); re-point if omp moves. Dynamic import:
// the module lives outside this repo under $HOME, so its specifier is only known at runtime.
const { fuzzyMatch } = await import(`${process.env.HOME}/.bun/install/global/node_modules/@oh-my-pi/pi-tui/src/fuzzy.ts`);

const [query = "qwen3.8:27b-mlx", ...given] = process.argv.slice(2);
const candidates = given.length
	? given
	: [
			"localbench-parked/qwen3.8:27b-mlx",
			"localbench-parked:5642e97495e1",
			"qwen3.6:35b-mlx",
			"qwen3.8-uncensored:latest",
			"minimax-m2.5:cloud",
			"nomic-embed-text:latest",
		];
for (const id of candidates) console.log(JSON.stringify({ query, id, ...fuzzyMatch(query, id) }));
