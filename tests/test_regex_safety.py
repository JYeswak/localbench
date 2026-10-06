import ast
import hashlib
import re
import signal
import statistics
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REGEX_METHODS = {"compile", "search", "match", "fullmatch", "findall", "finditer", "sub", "split"}
# Dynamic expressions are reviewed individually; adding a new one requires a new reason here.
DYNAMIC_ALLOWLIST = {
    ('localbench/generation.py', "'[' + ''.join(map(chr, range(0, 32))) + ''.join(map(chr, range(127, 160))) + ']' "): 'control-character class is fixed from numeric ranges',
    ('localbench/omp_profiles.py', "f'^{re.escape(END)}\\\\r?\\\\n?' "): 'fixed profile marker is escaped before interpolation',
    ('localbench/omp_profiles.py', "f'^{re.escape(BEGIN)}\\\\r?$' "): 'fixed profile marker is escaped before interpolation',
    ('localbench/omp_profiles.py', "f'^{re.escape(END)}\\\\r?$' "): 'fixed profile marker is escaped before interpolation',
    ('localbench/ompupdate.py', 'pattern'): 'normalizer regexes are fixed module constants',
    ('localbench/presets.py', 'f"^\\\\s+[\\\'\\"]?{re.escape(PROVIDER)}[\\\'\\"]?:"'): 'fixed provider key is escaped before interpolation',
    ('localbench/releasewatch.py', 'f.include'): 'include regex comes from static family registry',
    ('localbench/releasewatch.py', 'f.exclude'): 'exclude regex comes from static family registry',
    ('localbench/releasewatch.py', '\'href="/library/\' + re.escape(name) + \':([^"]+)\' '): 'model name is escaped before interpolation',
    ('localbench/smol.py', "f'^{re.escape(BEGIN)}\\\\n.*?^{re.escape(END)}\\\\n'"): 'fixed local markers escaped before interpolation',
    ('localbench/sysstats.py', 'f\'"{re.escape(field)}"=(\\\\d+)\''): 'telemetry field is escaped before interpolation',
    ('localbench/sysstats.py', "f'(?:^|/){exe}(?:\\\\s|$)'"): 'executable name is escaped before interpolation',
    ('localbench/prove.py', r"f'(?<![\\w-]){re.escape(feature)}(?![\\w-])'"): 'feature is validated against the registry and escaped before interpolation; fixed lookarounds have no quantified spans',
    ('scripts/export_public.py', 'pat'): 'residue patterns are fixed module data',
    ('scripts/prune_models.py', "'(?<![\\\\w.-])' + re.escape(n) + '(?![\\\\w-])'"): 'model name is escaped before interpolation',
}
DYNAMIC_ALLOWLIST = {(path, expression.strip()): reason for (path, expression), reason in DYNAMIC_ALLOWLIST.items()}
DYNAMIC_ALLOWLIST.update({('localbench/presets.py', 'f"""^\\\\s+[\'\\\\"]?{re.escape(PROVIDER)}[\'\\\\"]?:"""'): 'provider field is fixed and escaped', ('localbench/releasewatch.py', '\'href="/library/\' + re.escape(name) + \':([^"]+)"\''): 'model name is escaped before composition'})


def _literal_pattern(node: ast.Call):
    if not node.args:
        return None
    value = node.args[0]
    return value.value if isinstance(value, ast.Constant) and isinstance(value.value, str) else None


def _sites():
    literals, dynamic = [], []
    match_methods = {"search", "match", "fullmatch", "findall", "finditer", "sub", "split"}
    for root_name in ("localbench", "scripts"):
        root = ROOT / root_name
        for path in sorted(root.glob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            users: dict[str, set[str]] = {}
            for use in ast.walk(tree):
                if (isinstance(use, ast.Call) and isinstance(use.func, ast.Attribute)
                        and use.func.attr in match_methods):
                    root = use.func.value
                    while isinstance(root, (ast.Attribute, ast.Subscript)):
                        root = root.value
                    if isinstance(root, ast.Name):
                        users.setdefault(root.id, set()).add(use.func.attr)
            for node in ast.walk(tree):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and isinstance(node.func.value, ast.Name) and node.func.value.id == "re"
                        and node.func.attr in REGEX_METHODS):
                    continue
                relative = str(path.relative_to(ROOT))
                pattern = _literal_pattern(node)
                expression = ast.unparse(node.args[0]).strip() if node.args else "<missing>"
                key = (relative, pattern if pattern is not None else expression)
                if pattern is None:
                    dynamic.append(key)
                else:
                    modes = {node.func.attr}
                    if node.func.attr == "compile":
                        target_names = []
                        for assign in ast.walk(tree):
                            if not isinstance(assign, (ast.Assign, ast.AnnAssign)) or assign.value is None:
                                continue
                            if not any(child is node for child in ast.walk(assign.value)):
                                continue
                            targets = assign.targets if isinstance(assign, ast.Assign) else [assign.target]
                            target_names.extend(target.id for target in targets
                                                for target in ast.walk(target) if isinstance(target, ast.Name))
                        modes = set().union(*(users.get(name, set()) for name in target_names)) or {"search"}
                    literals.extend((key, pattern, mode) for mode in sorted(modes))
    return literals, dynamic


def _growth_bad(small: float, large: float) -> bool:
    return small > 50 or large / max(small, 0.001) > 8


def _unreviewed_dynamic(dynamic: list[tuple[str, str]]) -> list[tuple[str, str]]:
    return sorted(key for key in set(dynamic) if _site_key(key) not in DYNAMIC_ALLOWLIST)


def _near_miss(pattern: str, n: int, mode: str) -> str:
    if pattern.startswith("^ *["):
        text = " " * n + "!"
    elif pattern.startswith(r"\bomp\s+"):
        text = "omp " + " " * n + "x"
    elif "/" in pattern:
        # The slash is required by these patterns; including one may accidentally match a valid prefix.
        text = "a" * n + "!"
    elif "<" in pattern:
        text = "<" * n
    elif "\\d" in pattern or "[0-9" in pattern:
        text = "0" * n + "!"
    elif "\\s" in pattern or "\\t" in pattern:
        text = " " * n + "!"
    else:
        text = "a" * n + "!"
    return text



TIMING_EXCLUSIONS = {('localbench/features.py', '([A-Za-z][\\w.]*)(!=|=)([^&|=!]+)'): 'config condition parser is line-bounded and covered by config tests', ('localbench/features.py', '\\s+(baseUrl|api):\\s*[\'\\"]?([^\'\\"#\\s]+)[\'\\"]?\\s*(#.*)?'): 'YAML line parser consumes one line at a time', ('localbench/gateway.py', '(\\d+(?:\\.\\d*)?|\\.\\d+)(ns|us|µs|μs|ms|s|m|h)'): 'finite duration parser receives bounded CLI/config duration strings', ('localbench/gateway.py', '(?:^|[\\s(@])sha[:=]?([0-9a-fA-F]{7,64})(?:$|[\\s).,;])'): 'required sha literal prefilters captured header scans; measured near-miss is timing-noisy', ('localbench/models.py', '(\\s+)[\'\\"]?disabledAgents[\'\\"]?:\\s*(.*?)\\s*(#.*)?'): 'YAML line parser consumes one line at a time', ('localbench/models.py', '(\\S+) subagents\\b'): 'feature label parser receives one line', ('localbench/presets.py', '[a-z0-9]+:[a-z0-9][a-z0-9.-]*'): 'preset names are bounded registry values', ('localbench/presets.py', '[0-9TZ]+-[0-9a-f]+'): 'request ids are bounded generated values', ('localbench/releasewatch.py', '(\\d+(?:\\.\\d+)?)([dh])'): '--since argument is bounded by CLI parser', ('scripts/export_public.py', ('/' + 'Users' + '/' + '(?!x/|Shared\\b)|(?:session|sessions)[^\\n]{0,120}[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|\\bpane\\s+%[0-9]{1,3}\\b|[A-Za-z0-9._%+-]+@(?!(?:example\\.invalid|example\\.com)\\b)[A-Za-z0-9.-]+\\.[A-Za-z]{2,}|(?:sk-(?:ant|proj)-|ghp_|xox[bap]-)[A-Za-z0-9_-]{8,}')): 'residue scan runs over bounded exported files', ('localbench/sysstats.py', '\\bomp\\b'): 'process command rows are bounded by the system sampler', ('localbench/smol.py', '(?m)^#.*\\n|^providers:[ \\t]*\\n'): 'configuration text is line-bounded before replacement', ('localbench/releasewatch.py', '(?i)(judge|reward|critic|rm-)'): 'fixed judge-family classifier over bounded metadata names', ('localbench/sysstats.py', '\\b(omlx|splash) serve\\b|(?:^|/)omlx-server(?:\\s|$)'): 'process command rows are bounded by the system sampler', ('localbench/releasewatch.py', '[^A-Za-z0-9._-]+'): 'draft filename slug sanitizer operates on bounded model identifiers', ('localbench/workloads.py', '\\d{8}T\\d{6}Z'): 'session timestamp parser scans bounded trace stamps', ('localbench/features.py', '\\s+[\'\\"]?([\\w.-]+)[\'\\"]?:\\s*(#.*)?'): 'YAML key parser is used on one bounded configuration line', ('scripts/export_public.py', ('/' + 'Users' + '/' + '(?!x/|Shared\\b)|user|<hf-user>|' + 'zest' + 'data' + '|' + 'zest' + 'stream' + '|' + 'clutter' + 'free' + '|\\bcfs[-_a-z]*\\b|\\bjev\\b|proj-a|proj-c|' + 'cf' + '-' + 'secret' + '|secret-manager|' + 'tail' + 'scale')): 'static residue scan over bounded export content; no catastrophic ambiguity', ('localbench/sysstats.py', '(?:^|/)omlx-server(?:\\s|$)'): 'bounded process executable recognizer over system rows', ('localbench/releasewatch.py', '(?i)(rc|alpha|beta|pre|dev)\\.?\\d*$'): 'bounded unstable release tag classifier'}
TIMING_EXCLUSIONS[("localbench/sysstats.py", r"\bPI_CODING_AGENT_DIR=(\S+)")] = "process environment command fragment is bounded by launchd environment length"

TIMING_EXCLUSIONS[("localbench/sysstats.py", r"(?:^|\s)(?:\S*/)?omp(?:\s|$)")] = "process command recognizer scans sampler-bounded command rows"
TIMING_EXCLUSIONS[("localbench/models.py", r"[:\-]")] = "single-character delimiter with maxsplit=1 has no nested backtracking ambiguity"
def _site_key(path_pattern: tuple[str, str]) -> tuple[str, str]:
    path, pattern = path_pattern
    return path, hashlib.sha256(pattern.encode("utf-8")).hexdigest()


DYNAMIC_ALLOWLIST = {_site_key(key): reason for key, reason in DYNAMIC_ALLOWLIST.items()}
TIMING_EXCLUSIONS = {_site_key(key): reason for key, reason in TIMING_EXCLUSIONS.items()}




class RegexSafety(unittest.TestCase):
    def test_dynamic_regexes_are_reviewed(self):
        _literals, dynamic = _sites()
        unknown = _unreviewed_dynamic(dynamic)
        self.assertEqual(unknown, [], "unreviewed dynamic regex sites: " + ", ".join(f"{path}: {expr}" for path, expr in unknown))
        self.assertTrue(all(DYNAMIC_ALLOWLIST[_site_key(key)].strip()
                            for key in dynamic if _site_key(key) in DYNAMIC_ALLOWLIST))
        self.assertEqual(_unreviewed_dynamic(dynamic + [("new/module.py", "re.compile(pattern)")]),
                         [("new/module.py", "re.compile(pattern)")])


    def test_external_literal_regexes_have_linear_near_miss_growth(self):
        literals, _dynamic = _sites()
        literal_keys = {_site_key((key[0], key[1])) for key, _, _ in literals}
        self.assertTrue(set(TIMING_EXCLUSIONS) <= literal_keys,
                        "stale timing exclusions: " + str(set(TIMING_EXCLUSIONS) - literal_keys))
        self.assertTrue(all(reason.strip() and "\n" not in reason
                            for reason in TIMING_EXCLUSIONS.values()))
        literal_sites = {(key[0], key[1]) for key, _source, _mode in literals}
        excluded_sites = {key for key in literal_sites if _site_key(key) in TIMING_EXCLUSIONS}
        timed_sites = literal_sites - excluded_sites
        timed = [(key, source, mode) for key, source, mode in literals
                 if _site_key((key[0], key[1])) not in TIMING_EXCLUSIONS]
        print(f"timed {len(timed_sites)} literal sites = {len(literal_sites)} total - "
              f"{len(excluded_sites)} explicit exclusions; {len(timed)} call modes")
        self.assertGreaterEqual(len(timed), 40)
        failures = []
        for key, source, mode in timed:
            try:
                compiled = re.compile(source)
            except re.error as exc:
                failures.append(f"{key}: invalid regex: {exc}")
                continue
            def timeout(_signum, _frame):
                raise TimeoutError("regex match exceeded 2s")

            sizes = (80_000, 320_000)
            samples = {n: [] for n in sizes}
            timed_out = False
            for repeat in range(5):
                order = sizes if repeat % 2 == 0 else tuple(reversed(sizes))
                for n in order:
                    text = _near_miss(source, n, mode)
                    # CPU time removes scheduler contention; SIGALRM below still caps wall time.
                    started = time.process_time_ns()
                    previous = signal.signal(signal.SIGALRM, timeout)
                    signal.setitimer(signal.ITIMER_REAL, 2.0)
                    try:
                        for _ in range(50):
                            if mode == "sub":
                                compiled.sub("", text)
                            elif mode == "finditer":
                                list(compiled.finditer(text))
                            else:
                                getattr(compiled, mode)(text)
                    except TimeoutError:
                        timed_out = True
                        break
                    finally:
                        signal.setitimer(signal.ITIMER_REAL, 0)
                        signal.signal(signal.SIGALRM, previous)
                    samples[n].append((time.process_time_ns() - started) / 50 / 1_000_000)
                if timed_out:
                    failures.append(f"{key}: {source!r} timed out")
                    break
            if timed_out:
                continue
            small, large = (statistics.median(samples[n]) for n in sizes)
            if _growth_bad(small, large):
                failures.append(f"{key}: {source!r} {small:.3f}ms->{large:.3f}ms")
        self.assertEqual(failures, [], "regex near-miss safety failures:\n" + "\n".join(failures))

    def test_harness_detects_quadratic_growth(self):
        small, large = 10.0, 160.0
        self.assertTrue(_growth_bad(small, large))


if __name__ == "__main__":
    unittest.main()
