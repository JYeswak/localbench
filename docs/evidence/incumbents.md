# Pinned incumbents and candidates (CHECKLIST A2)

<!--
  Anti-ceremony (A12):
  - Consumer: every A/B receipt and golden; the reviewer checking a comparison is against a PINNED incumbent.
  - Gate: A2 incumbent pinned before implementation; B3 generation binding (a receipt whose pins differ from this file is from another generation).
  - Defect class: comparisons against a drifting, unpinned baseline ("it got faster" when ollama auto-updated).
  - Delete when: goldens carry every pin in their provenance AND compare refuses cross-generation (LB-06); then this file is a human index only.
-->

Hardware anchor recorded 2026-09-22 on `mac-studio-apple-m3-ultra-512gb` (Mac15,14, M3 Ultra,
80 GPU cores, 512 GB, macOS 26.5.2 build 25F84). Component observations below have their own dates;
binary hashes are the first 16 hex of `shasum -a 256`. A local configuration readback is not a
banked same-generation performance or live-session routing proof.

the owner reported an Ollama upgrade on 2026-09-30 after the 0.34.4 readback. The
0.34.4 binary row below is now historical; the active runtime version and hash
are not pinned in this document. Do not carry its receipts into the upgraded
runtime's generation without a fresh readback and sound comparison.

## Incumbent configuration (dated readbacks, not a live-session census)

| Component | Pin | How pinned |
|---|---|---|
| ollama | 0.35.0, binary sha16 `add45eb02df0252f` in the 2026-09-30 `localbench status` readback; Ollama.app auto-update was ON in that readback, so a later generation needs a fresh pin. Prior 0.34.4 binary sha16 `bba8b79eac84ab09` (Ollama.app from GitHub's Ollama-darwin.zip, archive sha256 `f7ed834269e98929`) was installed 2026-09-24 ~23:30 local; its model store `/Volumes/Models/ollama-models` was set in the app's own settings (`models` in ~/Library/Application Support/Ollama/db.sqlite; 0.34.x ignored launchctl OLLAMA_MODELS). Ollama.app has Full Disk Access (granted by the owner) so its server can read the external volume. Auto-update had been OFF since 2026-09-25 13:10Z (07:10 MDT), when `auto_update_enabled = 0` was backed up at ~/.localbench/rollback/ollama-db-20260925T1310Z.sqlite; this is historical, not the current setting. | `localbench status` for current backend version, binary pin and auto-update state; previous readbacks used `ollama --version` and `shasum -a 256 /Applications/Ollama.app/Contents/Resources/ollama`. Earlier: 0.32.15 (`eee609f0a6da58b9`), rollback copy at ~/.localbench/rollback/Ollama-0.32.15.app (signature verified) |
| model | `qwen3.8:27b-mlx` digest `5642e97495e1` (qwen3_5 dense, 27.8B, nvfp4) | `localbench models --json` (`installed[].digest`), read back 2026-09-30 |
| smol | `localbench models --json` readbacks on 2026-09-29 and 2026-09-30 enumerate 13 OMP profiles: default, agy, claude, codex, glm, grok, lab, muse, nv-deepseek-flash, nv-glm, nv-glm-flash, nv-kimi, omp-test. The resolved local-route inventory associates ollama `qwen3.8:27b-mlx` with 12 profiles (all except agy); agy has no installed local smol route in this inventory, which says nothing about its nonlocal roles. The older eight-profile index is superseded. The mlx-serve 26.9.5 pack (`mlx-smol/Qwen3.8-27B-MLX-Serve-4bit`) was adopted 2026-09-25 16:18Z and reverted ~22:58Z after raw tool-call text in a `--no-tools` mem tier; this does **not** prove tool-calling failure for real sessions that declare tools (`docs/evidence/NEGATIVE_EVIDENCE.md:800-810`, `docs/evidence/receipts/ab-smol-qwen38-ollama-vs-mlxserve-4bit-mtp-20260925.json`; the A/B is UNSOUND). The route inventory cannot establish that pre-migration OMP sessions reloaded configuration. | `localbench models --json` (`installed[].routes`, `profiles`); `localbench smol status` ("not managed") |
| omp | 18.4.5 at `~/.bun/bin/omp`, sha16 `63fc92a4e33fd7fe` in the 2026-09-30 `localbench status` readback; no banked successful run for that binary is cited here. Previous 18.4.4 at the same path had sha16 `ca9b8832ea05299f` in an earlier 2026-09-30 readback. The 2026-09-29 readback was 18.4.3, sha16 `b72ee39b7feb2d59` (also recorded in `docs/evidence/receipts/aa__mlx-serve__Qwen3.6-35B-A3B-MLX-Serve-4bit__20260929T085025Z.json:12-16`, a dirty-source micro-only A/A, not current live-OMP proof). Earlier 18.3.5 at the same path had sha16 `950b21faae3266b1` since 2026-09-27 20:00 local (updated in place, not by localbench; proj-c found it). `~/.local/bin/omp` is proj-c's omp trust guard (installed 2026-09-26, execs `OMP_REAL_BIN` from ~/.config/omp-trust/config = ~/.bun/bin/omp); park's resolver follows it. Before that: 18.3.1 at `~/.bun/bin/omp`, sha16 `cacaf5726609a21b` since 2026-09-25 05:52Z (omp replaced its first 18.3.1 build `46cb390bb7e6def5`, which ran 3.2-5.4 s startup, under the same version string; 18.3.0 was `33cf63aab3a109b3`), default profile `~/.omp/agent` | `localbench status`; `omp --version`; `shasum -a 256 $(readlink -f ~/.bun/bin/omp)`. Upgraded in place from 18.2.11 (`ce797fb3ed92e768`), not by localbench; first pinned by receipts/aa__ollama__localbench-parked_5642e97495e1__20260924T025154Z.json, and each later receipt pins its own sha. The version is recorded, not a golden key (4b9532d). Earlier: 18.2.10 (`acf06c76a4969558`) until 2026-09-23T05:28:44Z; its goldens are in commit d7a4ea2. The second install at `~/.local/bin/omp` (18.2.10) was gone by 2026-09-24; `LOCALBENCH_OMP` still overrides the binary |
| omp child overlay | `fixtures/omp/child-config.yml` (`memory.backend: off`), sha16 `62eed267e219ddde` | pinned as `omp_child_config` in every run; see ledger 2026-09-23 (memory feedback) |
| omp request shape | historical full default-profile prompt: 73,779 prompt tokens (ollama qwen3.6 tokenizer), 11 tools | `fixtures/omp/full.json` + `full.meta.json`, recorded 2026-09-23 under omp 18.2.11 and ollama 0.32.15 with the child overlay (18.2.10 without overlay: 74,284); historical fixture, not a current 18.4.5 request-shape observation |

## Candidates

These are comparison targets, not currently qualified replacements. Parameter counts, server features, and
the 2026-09-23 fixture-size observations do not establish current latency or smol-role quality against
the pinned incumbent; a comparison needs a sound, same-invocation A/B under the relevant pins.

| Component | Pin | Notes |
|---|---|---|
| model (ollama) | `qwen3.6:35b-mlx` digest `e92a3e94bbca` (qwen3_5_moe, 36.0B total / ~3B active, nvfp4) | MoE: fewer active params per token |
| mlx-serve | 26.9.2, tap `ddalcu/mlx-serve@30f32ccc9f9f`, binary sha256 `4d09a3beb8d4c9be` (PATH's at pinning; the tap still pinned 26.9.2 after `brew update` on 2026-09-25). Side-by-side 26.9.5 (GitHub release 2026-09-21, tarball sha256 `06c087e623a72907` verified) extracted at `~/.localbench/mlx-serve-26.9.5/mlx-serve-macos-arm64/mlx-serve`, binary sha256 `f6b32efcbbaa3d2d`, mlx 0.32.2 (26.9.2: mlx 0.30.3); trialed for smol on 2026-09-25, then reverted; selectable for benchmark legs via `LOCALBENCH_MLX_SERVE` or `ab --b-mlx-serve` (bf1e7ed) | native MLX server; multi-entry prefix cache; Qwen MTP |
| model (mlx-serve) | `ddalcu/Qwen3.6-35B-A3B-MLX-Serve-4bit` @ HF `6122e2b20a1d2e6c811b02b7912a91f1e8548de8`; harness pin `files:68dedceb5da0` (sha256 of config.json + model.safetensors.index.json + tokenizer_config.json; the pull recorded no HF commit file) | ships MTP layer (`mtp_num_hidden_layers: 1`); served id `Qwen3.6-35B-A3B-MLX-Serve-4bit` |
| model (mlx-serve) | `ddalcu/Qwen3.8-27B-MLX-Serve-4bit` @ HF `b543ed7cbafbf4984266e784c2ccb1f0730847b6` | downloaded and exercised in the 2026-09-25 smol trial (`docs/evidence/NEGATIVE_EVIDENCE.md:758-763,800-810`); whether its local files remain available now is not established by the installed-route inventory |
| splash | 1.0, binary sha256 `6158ce6f1d4b2eb2` (`/opt/homebrew/bin/splash`) | `splash --version`; `shasum -a 256 $(which splash)`. Recorded on a run pin when Splash is the backend or `/v1/models` on port 8000 lists a model. Pinning the binary does not measure the model |
| omp request shape | historical lean: `--no-skills --no-rules --no-lsp --no-title --tools=read,bash,edit,write,grep,glob,todo` → 11,433 prompt tokens (ollama qwen3.6 tokenizer), 7 tools | `fixtures/omp/lean.json` + `lean.meta.json`, recorded 2026-09-23 under omp 18.2.11 and ollama 0.32.15 with the child overlay (18.2.10 without overlay: 11,645); not a current 18.4.5 measurement |

## Agent-workload oracle (LB-08)

| Component | Pin |
|---|---|
| swebench (PyPI) | 5.0.2 |
| dataset | `princeton-nlp/SWE-bench_Lite` @ `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2` |
| container runtime | OrbStack 2.2.3 (2020300) |

## Re-pinning

A change to a pin a tier exercised starts a new generation for that tier (`golden.tier_keys`). Backend, model,
or macOS changes re-bank every tier; child-overlay or agent-config changes re-bank live-omp tiers
(e2e, rel, relcold, relfresh, mem, sess). An omp version or binary SHA change alone is recorded but does not stale
a golden; this is a key-selection rule, not evidence that live behavior under the new binary is equivalent.
Changed fixtures re-bank replay. Review `git diff goldens/` and update this file in the same commit.
While parked (`localbench park`), the incumbent model is measured as
`ollama:localbench-parked:5642e97495e1`
(same digest `5642e97495e1`).
