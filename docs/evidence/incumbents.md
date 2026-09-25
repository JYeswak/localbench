# Pinned incumbents and candidates (CHECKLIST A2)

<!--
  Anti-ceremony (A12):
  - Consumer: every A/B receipt and golden; the reviewer checking a comparison is against a PINNED incumbent.
  - Gate: A2 incumbent pinned before implementation; B3 generation binding (a receipt whose pins differ from this file is from another generation).
  - Defect class: comparisons against a drifting, unpinned baseline ("it got faster" when ollama auto-updated).
  - Delete when: goldens carry every pin in their provenance AND compare refuses cross-generation (LB-06); then this file is a human index only.
-->

Pinned 2026-09-22 on `mac-studio-apple-m3-ultra-512gb` (Mac15,14, M3 Ultra, 80 GPU cores,
512 GB, macOS 26.5.2 build 25F84). Binary hashes are the first 16 hex of `shasum -a 256`.

## The incumbent (what the user runs today)

| Component | Pin | How pinned |
|---|---|---|
| ollama | 0.34.4, binary sha256 `bba8b79eac84ab09` (Ollama.app from GitHub's Ollama-darwin.zip, sha256 `f7ed834269e98929`), installed 2026-09-24 ~23:30 local; model store `/Volumes/Models/ollama-models` set in the app's own settings (`models` in ~/Library/Application Support/Ollama/db.sqlite; 0.34.x ignores launchctl OLLAMA_MODELS); Ollama.app has Full Disk Access (granted by the owner) so its server can read the external volume; the app's auto-update is OFF since 2026-09-25 13:10Z (07:10 MDT) (the owner's call: `auto_update_enabled = 0` in that db, backup ~/.localbench/rollback/ollama-db-20260925T1310Z.sqlite; the staged same-version bundle removed; `localbench status` shows the setting) | `ollama --version`; `shasum -a 256 /Applications/Ollama.app/Contents/Resources/ollama`. Previous: 0.32.15 (`eee609f0a6da58b9`), rollback copy at ~/.localbench/rollback/Ollama-0.32.15.app (signature verified) |
| model | `qwen3.8:27b-mlx` digest `5642e97495e1` (qwen3_5 dense, 27.8B, nvfp4) | `ollama list`, `ollama show` |
| smol (live since 2026-09-25 16:18Z) | `mlx-smol/Qwen3.8-27B-MLX-Serve-4bit`: ddalcu/Qwen3.8-27B-MLX-Serve-4bit @ b543ed7c on mlx-serve 26.9.5 (f6b32efcbbaa3d2d) --mtp, 127.0.0.1:11235, all 8 omp profiles; adopted by the owner without the stage 2 gate (ledger UNKNOWN row). The ollama `qwen3.8:27b-mlx` row above is the revert target | `localbench smol status`; `localbench smol revert` |
| omp | 18.3.1 at `~/.bun/bin/omp` (the binary every user pane runs), sha256 `cacaf5726609a21b` since 2026-09-25 05:52Z (omp replaced its first 18.3.1 build `46cb390bb7e6def5`, which ran 3.2-5.4 s startup, under the same version string; 18.3.0 was `33cf63aab3a109b3`), default profile `~/.omp/agent` | `omp --version`; `shasum -a 256 $(readlink -f ~/.bun/bin/omp)`. Upgraded in place from 18.2.11 (`ce797fb3ed92e768`), not by localbench; first pinned by receipts/aa__ollama__localbench-parked_5642e97495e1__20260924T025154Z.json, and every receipt since pins the same sha. The version is recorded, not a golden key (4b9532d). Earlier: 18.2.10 (`acf06c76a4969558`) until 2026-09-23T05:28:44Z; its goldens are in commit d7a4ea2. The second install at `~/.local/bin/omp` (18.2.10) was gone by 2026-09-24; `LOCALBENCH_OMP` still overrides the binary |
| omp child overlay | `fixtures/omp/child-config.yml` (`memory.backend: off`), sha16 `62eed267e219ddde` | pinned as `omp_child_config` in every run; see ledger 2026-09-23 (memory feedback) |
| omp request shape | full default-profile prompt: 73,779 prompt tokens (ollama qwen3.6 tokenizer), 11 tools | `fixtures/omp/full.json` + `full.meta.json`, recorded 2026-09-23 under omp 18.2.11 with the child overlay (18.2.10 without overlay: 74,284) |

## Candidates

| Component | Pin | Notes |
|---|---|---|
| model (ollama) | `qwen3.6:35b-mlx` digest `e92a3e94bbca` (qwen3_5_moe, 36.0B total / ~3B active, nvfp4) | MoE: fewer active params per token |
| mlx-serve | 26.9.2, tap `ddalcu/mlx-serve@30f32ccc9f9f`, binary sha256 `4d09a3beb8d4c9be` (PATH's; stays the default: the tap still pins 26.9.2 after `brew update` on 2026-09-25). Candidate beside it, not adopted: 26.9.5 (GitHub release 2026-09-21, tarball sha256 `06c087e623a72907` verified) extracted at `~/.localbench/mlx-serve-26.9.5/mlx-serve-macos-arm64/mlx-serve`, binary sha256 `f6b32efcbbaa3d2d`, mlx 0.32.2 (26.9.2: mlx 0.30.3); used only through `LOCALBENCH_MLX_SERVE` or `ab --b-mlx-serve` (bf1e7ed) | native MLX server; multi-entry prefix cache; Qwen MTP |
| model (mlx-serve) | `ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit` @ HF `6122e2b20a1d2e6c811b02b7912a91f1e8548de8`; harness pin `files:68dedceb5da0` (sha256 of config.json + model.safetensors.index.json + tokenizer_config.json; the pull recorded no HF commit file) | ships MTP layer (`mtp_num_hidden_layers: 1`); served id `Qwen3.6-35B-A3B-MLX-Serve-4bit` |
| model (mlx-serve) | `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` @ HF `b543ed7cbafbf4984266e784c2ccb1f0730847b6` | not downloaded; dense control for mlx-serve |
| splash | 1.0, binary sha256 `6158ce6f1d4b2eb2` (`/opt/homebrew/bin/splash`) | `splash --version`; `shasum -a 256 $(which splash)`. Recorded on a run pin when Splash is the backend or `/v1/models` on port 8000 lists a model. Pinning the binary does not measure the model |
| omp request shape | lean: `--no-skills --no-rules --no-lsp --no-title --tools=read,bash,edit,write,grep,glob,todo` → 11,433 prompt tokens (ollama qwen3.6 tokenizer), 7 tools | `fixtures/omp/lean.json` + `lean.meta.json`, recorded 2026-09-23 under omp 18.2.11 with the child overlay (18.2.10 without overlay: 11,645) |

## Agent-workload oracle (LB-08)

| Component | Pin |
|---|---|
| swebench (PyPI) | 5.0.2 |
| dataset | `princeton-nlp/SWE-bench_Lite` @ `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2` |
| container runtime | OrbStack 2.2.3 (2020300) |

## Re-pinning

A change to a pinned row starts a new generation for the tiers that depend on it (golden.tier_keys): a backend,
model, or macOS change re-banks every tier; an omp or child-config change re-banks only e2e/rel
(`localbench aa <spec> --tiers e2e --write-golden`); re-recorded fixtures re-bank replay. Review `git diff goldens/` and
update this file in the same commit. While parked (`localbench park`), the incumbent model is measured as
`ollama:localbench-parked:5642e97495e1`
(same digest `5642e97495e1`).
