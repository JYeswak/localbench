"""The Rust-port interface map (docs/port/interfaces.tsv) against the code it maps. Defended contract: every stdlib
module, subprocess executable, HTTP endpoint literal, environment variable and SQLite database that localbench/*.py or
scripts/*.py uses has a row naming its Rust equivalent, and no row of those kinds names something the code stopped
using. Extraction is static (ast: nothing is imported or run), so a new `import`, `subprocess.run([...])`, URL, API
path or `os.environ` read fails here until the map carries it. Kinds the AST cannot see (file, text-format, service,
tool, os) are hand-curated: they must be present and mapped, and are not checked for staleness."""

import ast
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT / "docs" / "port" / "interfaces.tsv"
HEADER = ["kind", "name", "used_in", "purpose", "rust", "notes"]
AST_KINDS = ("py-stdlib", "subprocess", "http", "sqlite", "env")
CURATED_KINDS = ("file", "text-format", "service", "tool", "os")
HTTP_METHODS = {"GET", "POST", "PUT", "DELETE", "ANY"}

# Calls that stand for a fixed executable or for a copy of the environment, mapped by name (test_helper_maps_are_live
# fails when a mapped helper disappears, so a rename cannot silently blind the extractor).
EXEC_HELPERS = {"omp_bin": "omp", "mlx_serve_bin": "mlx-serve", "server_bin": "mlx-serve"}
ENV_HELPERS = {"omp_env"}
# Functions returning a whole argv list, and the program at its head.
ARGV_HELPERS = {"server_argv": "mlx-serve"}
SUBPROCESS_CALLS = {"run", "Popen", "call", "check_call", "check_output"}
OS_SHELL_CALLS = {"system", "popen"}
ENV_METHODS = {"get", "pop", "setdefault"}
URL = re.compile(r"(https?)://([A-Za-z0-9.\-]+|\[[0-9A-Fa-f:]+\])(:\d+)?(/[^\s\"'?#{}]*)?")
DB_FILE = re.compile(r"([\w.\-]+\.(?:db|sqlite3?))$")
PARAM = object()  # resolution reached a parameter of the enclosing function: that function is a helper


def sources(root: Path = ROOT) -> list[Path]:
    return sorted((root / "localbench").glob("*.py")) + sorted((root / "scripts").glob("*.py"))


def rows(path: Path = TSV) -> list[dict]:
    lines = path.read_text().splitlines()
    assert lines and lines[0].split("\t") == HEADER, f"{path}: header must be {HEADER}"
    out = []
    for n, line in enumerate(lines[1:], 2):
        cells = line.split("\t")
        assert len(cells) == len(HEADER), f"{path}:{n}: {len(cells)} cells, want {len(HEADER)}"
        out.append(dict(zip(HEADER, cells, strict=True)) | {"line": n})
    return out


def _name(func: ast.expr) -> str | None:
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else None


def _is(node: ast.expr, module: str, attr: str) -> bool:
    return (isinstance(node, ast.Attribute) and node.attr == attr and isinstance(node.value, ast.Name)
            and node.value.id == module)


def _consts(node: ast.expr) -> list[str]:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return [s for e in node.elts for s in _consts(e)]
    return []


class _Module:
    """One parsed source: enclosing function of every node, assignments per scope, parameters per function."""

    def __init__(self, rel: str, tree: ast.Module):
        self.rel, self.tree = rel, tree
        self.scope: dict[ast.AST, ast.AST | None] = {}
        self.assigns: dict[ast.AST | None, dict[str, list[tuple[int, ast.expr]]]] = {}
        self.params: dict[ast.AST, tuple[list[str], str | None]] = {}
        self.helper_name: dict[ast.AST, str] = {}
        self._visit(tree, None, None)
        self.docstrings = {id(s.value) for s in ast.walk(tree)
                           if isinstance(s, ast.Expr) and isinstance(s.value, ast.Constant)}

    def _visit(self, node: ast.AST, func: ast.AST | None, cls: ast.ClassDef | None) -> None:
        for child in ast.iter_child_nodes(node):
            self.scope[child] = func
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names = [a.arg for a in child.args.posonlyargs + child.args.args]
                static = any(_name(d) == "staticmethod" for d in child.decorator_list)
                if cls is not None and not static and names:
                    names = names[1:]
                self.params[child] = (names, child.args.vararg.arg if child.args.vararg else None)
                self.helper_name[child] = cls.name if cls is not None and child.name == "__init__" else child.name
                self._visit(child, child, None)
            elif isinstance(child, ast.ClassDef):
                self._visit(child, func, child)
            else:
                targets = (child.targets if isinstance(child, ast.Assign) else
                           [child.target] if isinstance(child, ast.AnnAssign) and child.value is not None else [])
                for t in targets:
                    if isinstance(t, ast.Name):
                        self.assigns.setdefault(func, {}).setdefault(t.id, []).append((child.lineno, child.value))
                self._visit(child, func, cls)

    def lookup(self, name: str, node: ast.AST) -> ast.expr | None:
        """The value last assigned to `name` before `node` in its function; else the module's last assignment (a
        function body runs after the whole module has loaded, so a module constant defined below it still counts)."""
        func = self.scope.get(node)
        if func is not None:
            prior = [v for ln, v in self.assigns.get(func, {}).get(name, []) if ln <= node.lineno]
            if prior:
                return prior[-1]
        module = [(ln, v) for ln, v in self.assigns.get(None, {}).get(name, [])
                  if func is not None or ln <= node.lineno]
        return module[-1][1] if module else None

    def is_param(self, name: str, node: ast.AST) -> bool:
        func = self.scope.get(node)
        if func is None:
            return False
        names, vararg = self.params[func]
        return name in names or name == vararg


class Extraction:
    """Interfaces the sources use, as {kind: {name: {"file:line", ...}}}, plus call sites whose executable could not be
    resolved statically (`unresolved`)."""

    def __init__(self, files: list[tuple[str, str]]):
        self.found: dict[str, dict[str, set[str]]] = {k: {} for k in AST_KINDS}
        self.unresolved: list[str] = []
        self.defined: set[str] = set()
        mods = [_Module(rel, ast.parse(text, rel)) for rel, text in files]
        self.helpers: dict[str, tuple[str, int]] = {}
        while self._find_helpers(mods):
            pass
        for m in mods:
            self.defined |= {n.name for n in ast.walk(m.tree) if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
            for node in ast.walk(m.tree):
                self._imports(m, node)
                self._subprocess(m, node)
                self._env(m, node)
                self._strings(m, node)

    def add(self, kind: str, name: str, m: _Module, node: ast.AST) -> None:
        self.found[kind].setdefault(name, set()).add(f"{m.rel}:{getattr(node, 'lineno', 0)}")

    # ------------------------------------------------------------------ imports

    def _imports(self, m: _Module, node: ast.AST) -> None:
        mods = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
        for mod in mods:
            if mod.split(".")[0] != "localbench":
                self.add("py-stdlib", mod, m, node)

    # --------------------------------------------------------------- subprocess

    def _argv_sites(self, node: ast.AST) -> tuple[str, list[ast.expr]] | None:
        """(form, argv elements or [argv expr]) for a call that launches a program, else None."""
        if not isinstance(node, ast.Call):
            return None
        f = node.func
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "subprocess" \
                and f.attr in SUBPROCESS_CALLS:
            arg = node.args[0] if node.args else next((k.value for k in node.keywords if k.arg == "args"), None)
            return ("argv", [arg]) if arg is not None else None
        if isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "os" \
                and (f.attr in OS_SHELL_CALLS or f.attr.startswith(("exec", "spawn"))) and node.args:
            return ("shell", [node.args[0]]) if f.attr in OS_SHELL_CALLS else ("argv", [node.args[0]])
        helper = self.helpers.get(_name(f) or "")
        if helper:
            form, i = helper
            if form == "vararg":
                return ("elts", node.args[i:]) if len(node.args) > i else None
            return ("argv", [node.args[i]]) if len(node.args) > i else None
        return None

    def _head(self, m: _Module, expr: ast.expr, at: ast.AST, depth: int = 0) -> object:
        if depth > 8:
            return None
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            return expr.value
        if isinstance(expr, ast.Call) and _name(expr.func) in EXEC_HELPERS:
            return EXEC_HELPERS[_name(expr.func)]
        if isinstance(expr, ast.Starred):
            return self._argv(m, expr.value, at, depth + 1)
        if isinstance(expr, ast.Name):
            if m.is_param(expr.id, at):
                return PARAM
            value = m.lookup(expr.id, at)
            return self._head(m, value, at, depth + 1) if value is not None else None
        return None

    def _argv(self, m: _Module, expr: ast.expr, at: ast.AST, depth: int = 0) -> object:
        if depth > 8:
            return None
        if isinstance(expr, (ast.List, ast.Tuple)):
            return self._head(m, expr.elts[0], at, depth + 1) if expr.elts else None
        if isinstance(expr, ast.Call) and _name(expr.func) in ARGV_HELPERS:
            return ARGV_HELPERS[_name(expr.func)]
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.Add):
            return self._argv(m, expr.left, at, depth + 1)
        if isinstance(expr, ast.Name):
            if m.is_param(expr.id, at):
                return PARAM
            value = m.lookup(expr.id, at)
            return self._argv(m, value, at, depth + 1) if value is not None else None
        return None

    def _elements(self, m: _Module, form: str, exprs: list[ast.expr], at: ast.AST) -> list[ast.expr]:
        if form == "elts":
            return exprs
        expr = exprs[0]
        if isinstance(expr, ast.Name) and not m.is_param(expr.id, at):
            expr = m.lookup(expr.id, at) or expr
        return list(expr.elts) if isinstance(expr, (ast.List, ast.Tuple)) else [expr]

    def _resolve(self, m: _Module, node: ast.Call) -> tuple[object, list[str]]:
        form, exprs = self._argv_sites(node)
        if form == "shell":
            text = exprs[0].value if isinstance(exprs[0], ast.Constant) and isinstance(exprs[0].value, str) else None
            return (text.split()[0] if text and text.split() else None), []
        elts = self._elements(m, form, exprs, node)
        head = self._head(m, elts[0], node) if form == "elts" else self._argv(m, exprs[0], node)
        extra = []
        if head == "sudo":  # the privileged program sudo runs is an interface of its own
            extra = [c for e in elts[1:] for c in _consts(e) if c.startswith("/")][:1]
        return head, extra

    def _find_helpers(self, mods: list[_Module]) -> bool:
        """Register functions whose own parameter becomes the argv of a launch: calls to them are launches too."""
        changed = False
        for m in mods:
            for node in ast.walk(m.tree):
                site = self._argv_sites(node)
                func = m.scope.get(node)
                if not site or func is None:
                    continue
                form, exprs = site
                first = exprs[0] if exprs else None
                target = first.value if isinstance(first, ast.Starred) else first
                if not isinstance(target, ast.Name) or not m.is_param(target.id, node):
                    continue
                names, vararg = m.params[func]
                if target.id == vararg:
                    spec = ("vararg", len(names))
                elif form == "elts" and not isinstance(first, ast.Starred):
                    continue  # a parameter as the executable itself: not a pattern the sources use
                else:
                    spec = ("argv", names.index(target.id))
                name = m.helper_name[func]
                if name not in self.helpers:  # first definition wins: a name clash cannot make the fixpoint oscillate
                    self.helpers[name] = spec
                    changed = True
        return changed

    def _subprocess(self, m: _Module, node: ast.AST) -> None:
        if not self._argv_sites(node):
            return
        head, extra = self._resolve(m, node)
        if head is PARAM:
            return  # a helper's own launch site: its callers are resolved instead
        if not isinstance(head, str):
            self.unresolved.append(f"{m.rel}:{node.lineno}")
            return
        for name in (head, *extra):
            self.add("subprocess", name, m, node)

    # ---------------------------------------------------------------------- env

    def _is_environ(self, m: _Module, expr: ast.expr, at: ast.AST, depth: int = 0) -> bool:
        """os.environ itself, a call to an ENV_HELPERS function, or a local name bound to a copy of either."""
        if depth > 8:
            return False
        if _is(expr, "os", "environ"):
            return True
        if isinstance(expr, ast.Call):
            if _name(expr.func) in ENV_HELPERS:
                return True
            if isinstance(expr.func, ast.Attribute) and expr.func.attr in ("copy", "items", "keys"):
                return self._is_environ(m, expr.func.value, at, depth + 1)
            if isinstance(expr.func, ast.Name) and expr.func.id == "dict" and expr.args:
                return self._is_environ(m, expr.args[0], at, depth + 1)
        if isinstance(expr, (ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            return any(self._is_environ(m, g.iter, at, depth + 1) for g in expr.generators)
        if isinstance(expr, ast.Name) and not m.is_param(expr.id, at):
            value = m.lookup(expr.id, at)
            return value is not None and self._is_environ(m, value, at, depth + 1)
        return False

    def _env(self, m: _Module, node: ast.AST) -> None:
        if isinstance(node, ast.Subscript) and self._is_environ(m, node.value, node):
            for c in _consts(node.slice):
                self.add("env", c, m, node)
        elif isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute) and f.attr in ENV_METHODS and node.args \
                    and self._is_environ(m, f.value, node):
                for c in _consts(node.args[0]):
                    self.add("env", c, m, node)
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "os" \
                    and f.attr in ("getenv", "putenv", "unsetenv") and node.args:
                for c in _consts(node.args[0]):
                    self.add("env", c, m, node)
            # Implicit reads: Path.home()/expanduser -> HOME, shutil.which -> PATH, tempfile -> TMPDIR.
            elif isinstance(f, ast.Attribute) and (
                    f.attr == "expanduser" or (f.attr == "home" and _name(f.value) == "Path")):
                self.add("env", "HOME", m, node)
            elif _is(f, "shutil", "which"):
                self.add("env", "PATH", m, node)
            elif isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "tempfile":
                self.add("env", "TMPDIR", m, node)
        elif isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], (ast.In, ast.NotIn)) \
                and self._is_environ(m, node.comparators[0], node):
            for c in _consts(node.left):
                self.add("env", c, m, node)
        elif isinstance(node, (ast.DictComp, ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            # {k: v for k, v in os.environ.items() if k not in ("A", "B")}: the filtered keys are environment reads.
            for g in node.generators:
                if not self._is_environ(m, g.iter, node):
                    continue
                key = g.target.elts[0] if isinstance(g.target, ast.Tuple) else g.target
                for cond in g.ifs:
                    for cmp in ast.walk(cond):
                        if isinstance(cmp, ast.Compare) and isinstance(cmp.left, ast.Name) \
                                and isinstance(key, ast.Name) and cmp.left.id == key.id:
                            for c in (c for comp in cmp.comparators for c in _consts(comp)):
                                self.add("env", c, m, node)

    # ------------------------------------------------------- http + sqlite literals

    def _strings(self, m: _Module, node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in m.docstrings:
            for u in URL.finditer(node.value):
                path = (u.group(4) or "").rstrip("/")
                self.add("http", f"{u.group(1)}://{u.group(2)}{u.group(3) or ''}{path}".lower(), m, node)
            db = DB_FILE.search(node.value)
            if db:
                self.add("sqlite", db.group(1), m, node)
        # API path literals: "/..." appended to a base (x + "/api/ps"), following a placeholder in an f-string
        # (f"{OLLAMA}/api/tags"), or matched against a request path (path.endswith("/chat/completions")).
        paths = []
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            paths = _consts(node.right) if not _consts(node.left) else []
        elif isinstance(node, ast.JoinedStr):
            after_value = False
            for v in node.values:
                if isinstance(v, ast.FormattedValue):
                    after_value = True
                elif after_value:
                    paths += _consts(v)
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and node.func.attr in ("endswith", "startswith", "removeprefix", "removesuffix") \
                and re.search(r"path|url", ast.unparse(node.func.value), re.I) and node.args:
            paths = _consts(node.args[0])
        for p in paths:
            p = p.split("?")[0]
            if re.fullmatch(r"/[A-Za-z0-9][\w.\-/]*", p):
                self.add("http", p, m, node)


def extract(root: Path = ROOT) -> Extraction:
    return Extraction([(p.relative_to(root).as_posix(), p.read_text()) for p in sources(root)])


def _http_row(name: str) -> tuple[str, str, str]:
    method, host, path = name.split(" ", 2)
    return method, host, path.split("?")[0]


def _contains_path(row_path: str, literal: str) -> bool:
    """`literal` occurs in `row_path` as whole segments ("/models" in "/v1/models", not in "/v1/modelsx")."""
    start = row_path.find(literal)
    while start >= 0:
        end = start + len(literal)
        if literal.endswith("/") or end == len(row_path) or row_path[end] in "/?":
            return True
        start = row_path.find(literal, start + 1)
    return False


def _bases(table: list[dict]) -> dict[str, str]:
    """host-kind -> base URL, from `service` rows whose purpose carries `base=<url>`."""
    out = {}
    for r in table:
        m = re.search(r"base=(https?://\S+)", r["purpose"]) if r["kind"] == "service" else None
        if m:
            out[r["name"]] = m.group(1).rstrip("/").lower()
    return out


def uncovered_http(found: dict[str, set[str]], table: list[dict]) -> list[str]:
    """Extracted URL/path literals no http row covers: a URL needs a service with its base and an http row of that
    service whose path starts with the URL's path; a bare path must occur (whole segments) in some http row's path."""
    bases = _bases(table)
    http = [_http_row(r["name"]) for r in table if r["kind"] == "http"]
    missing = []
    for lit, where in sorted(found.items()):
        if lit.startswith("http"):
            u = URL.match(lit)
            base, path = f"{u.group(1)}://{u.group(2)}{u.group(3) or ''}", u.group(4) or ""
            kinds = {k for k, b in bases.items() if b == base}
            ok = any(h in kinds and (not path or p == path or p.startswith(path + "/")) for _, h, p in http)
        else:
            ok = any(_contains_path(p, lit) for _, _, p in http)
        if not ok:
            missing.append(f"{lit} ({', '.join(sorted(where))})")
    return missing


class TableShape(unittest.TestCase):
    def setUp(self):
        self.table = rows()

    def test_every_row_is_well_formed_and_mapped(self):
        bad = [f"line {r['line']}: {r['kind']} {r['name']!r}" for r in self.table
               if r["kind"] not in AST_KINDS + CURATED_KINDS or not r["name"].strip() or not r["rust"].strip()
               or not r["purpose"].strip()]
        self.assertEqual(bad, [], "rows with an unknown kind or an empty name/purpose/rust cell")

    def test_no_duplicate_rows(self):
        seen, dups = set(), []
        for r in self.table:
            key = (r["kind"], r["name"])
            dups += [f"line {r['line']}: {key}"] if key in seen else []
            seen.add(key)
        self.assertEqual(dups, [])

    def test_every_kind_is_present(self):
        present = {r["kind"] for r in self.table}
        self.assertEqual([k for k in AST_KINDS + CURATED_KINDS if k not in present], [])

    def test_the_readers_counts_match_the_table(self):
        # INTERFACES.md's count table drifted three kinds out of date (subprocess 17 vs 22) with nothing to notice.
        doc = {m[1]: int(m[2]) for m in re.finditer(r"^\| ([a-z-]+) \| (\d+) \|", (TSV.parent / "INTERFACES.md")
                                                     .read_text(), re.MULTILINE)}
        table: dict[str, int] = {}
        for r in self.table:
            table[r["kind"]] = table.get(r["kind"], 0) + 1
        self.assertEqual(doc, table)

    def test_stdlib_rows_name_stdlib_modules(self):
        # localbench is stdlib-only (pyproject dependencies = []); a third-party import is dependency smuggling.
        self.assertEqual([r["name"] for r in self.table if r["kind"] == "py-stdlib"
                          and r["name"].split(".")[0] not in sys.stdlib_module_names], [])

    def test_http_rows_name_a_method_and_a_service_with_a_base(self):
        bases = _bases(self.table)
        bad = []
        for r in (r for r in self.table if r["kind"] == "http"):
            parts = r["name"].split(" ", 2)
            if len(parts) != 3 or parts[0] not in HTTP_METHODS or parts[1] not in bases or not parts[2].startswith("/"):
                bad.append(f"line {r['line']}: {r['name']!r}")
        self.assertEqual(bad, [], "http rows are `METHOD host-kind /path`; host-kind is a service row with base=<url>")


class CodeIsMapped(unittest.TestCase):
    """Every interface the sources use has a row."""

    @classmethod
    def setUpClass(cls):
        cls.x = extract()
        cls.table = rows()

    def missing(self, kind: str) -> list[str]:
        names = {r["name"] for r in self.table if r["kind"] == kind}
        return [f"{n} ({', '.join(sorted(w))})" for n, w in sorted(self.x.found[kind].items()) if n not in names]

    def test_every_import_has_a_row(self):
        self.assertEqual(self.missing("py-stdlib"), [], "imports without a py-stdlib row in docs/port/interfaces.tsv")

    def test_every_launched_executable_has_a_row(self):
        self.assertEqual(self.missing("subprocess"), [], "executables without a subprocess row")

    def test_every_launch_site_resolves_to_an_executable(self):
        self.assertEqual(self.x.unresolved, [], "launch sites whose argv[0] is not a literal, a mapped helper "
                                                "(EXEC_HELPERS), or a local bound to one")

    def test_every_environment_read_has_a_row(self):
        self.assertEqual(self.missing("env"), [], "environment variables without an env row")

    def test_every_sqlite_database_has_a_row(self):
        dbs = {r["name"].rsplit("/", 1)[-1] for r in self.table if r["kind"] == "sqlite"}
        self.assertEqual([f"{n} ({', '.join(sorted(w))})" for n, w in sorted(self.x.found["sqlite"].items())
                          if n not in dbs], [], "database files without a sqlite row")

    def test_every_url_and_api_path_has_a_row(self):
        self.assertEqual(uncovered_http(self.x.found["http"], self.table), [],
                         "URL/API path literals no http row covers")


class MapIsCurrent(unittest.TestCase):
    """Every row of an AST-visible kind still names something the sources use."""

    @classmethod
    def setUpClass(cls):
        cls.x = extract()
        cls.table = rows()

    def stale(self, kind: str) -> list[str]:
        return [f"line {r['line']}: {r['name']}" for r in self.table
                if r["kind"] == kind and r["name"] not in self.x.found[kind]]

    def test_no_stale_stdlib_rows(self):
        self.assertEqual(self.stale("py-stdlib"), [])

    def test_no_stale_subprocess_rows(self):
        self.assertEqual(self.stale("subprocess"), [])

    def test_no_stale_env_rows(self):
        # A variable only shell scripts read is checked against the scripts named in used_in.
        stale = []
        for r in (r for r in self.table if r["kind"] == "env" and r["name"] not in self.x.found["env"]):
            files = [ROOT / f.strip() for f in r["used_in"].split(",") if f.strip()]
            if any(f.suffix == ".py" for f in files) or not files or not all(
                    f.is_file() and re.search(rf"\b{re.escape(r['name'])}\b", f.read_text()) for f in files):
                stale.append(f"line {r['line']}: {r['name']}")
        self.assertEqual(stale, [])

    def test_no_stale_sqlite_rows(self):
        self.assertEqual([f"line {r['line']}: {r['name']}" for r in self.table if r["kind"] == "sqlite"
                          and r["name"].rsplit("/", 1)[-1] not in self.x.found["sqlite"]], [])

    def test_no_stale_http_rows(self):
        # A row is live when the last concrete segment of its path ({templates} skipped) is a whole segment of some URL
        # or path literal in the code. Containment of any prefix kept rows alive on "/v1/" alone: a planted row
        # `GET splash /v1/nonexistent-endpoint` passed (parent review, 2026-09-23).
        literals = [lit if not lit.startswith("http") else (URL.match(lit).group(4) or "")
                    for lit in self.x.found["http"]]
        segments = {s for lit in literals for s in lit.split("?")[0].split("/") if s}

        def last(path: str) -> str:
            concrete = [s for s in path.split("/") if s and not s.startswith("{")]
            return concrete[-1] if concrete else ""

        self.assertEqual([f"line {r['line']}: {r['name']}" for r in self.table if r["kind"] == "http"
                          and last(_http_row(r["name"])[2]) not in segments], [])

    def test_helper_maps_are_live(self):
        self.assertEqual(sorted((set(EXEC_HELPERS) | set(ARGV_HELPERS) | ENV_HELPERS) - self.x.defined), [])


class ExtractorSeesEachForm(unittest.TestCase):
    """The extractor on a known source: each form the real sources use (and a few they might) is found, and nothing
    else is. Without this, a broken extractor would report an empty code side and pass the missing-row checks."""

    SRC = '''
"""Docstring naming http://example.invalid/doc and a.db is not code."""
import json, urllib.request
from collections.abc import Callable
from . import golden
from localbench.proxy import Proxy
import os, shutil, subprocess, tempfile
from pathlib import Path

ROOT = "http://127.0.0.1:11434"
DB = Path("runs") / "observe.db"

def _run(*cmd, timeout=10):
    return subprocess.run(cmd, capture_output=True)

def omp_bin():
    return os.environ.get("LOCALBENCH_OMP") or shutil.which("omp")

def omp_env():
    return {k: v for k, v in os.environ.items() if k not in ("OMP_PROFILE", "PI_PROFILE")}

class _Rpc:
    def __init__(self, argv, cwd):
        self.proc = subprocess.Popen(argv, cwd=cwd)

def use(base, path):
    import plistlib
    _run("sysctl", "-n", "hw.model")
    subprocess.run(["sudo", "-n", "/usr/sbin/purge"])
    argv = [omp_bin(), "-p", "hi"]
    _Rpc(argv, "/tmp")
    subprocess.run(argv)
    env = omp_env()
    env["EXTRA_KEY"] = "1"
    omp_env().get("OMP_PROFILE", "default")
    os.getenv("DEBUG_X")
    Path.home() / ".omp"
    tempfile.mkdtemp()
    urllib.request.urlopen(ROOT + "/api/ps")
    urllib.request.urlopen(f"{base}/api/tags?x=1")
    path.endswith("/chat/completions")
    subprocess.run(build_argv())
'''

    def test_forms(self):
        x = Extraction([("mod.py", self.SRC)])
        self.assertEqual(set(x.found["py-stdlib"]), {"json", "urllib.request", "collections.abc", "os", "shutil",
                                                     "subprocess", "tempfile", "pathlib", "plistlib"})
        self.assertEqual(set(x.found["subprocess"]), {"sysctl", "sudo", "/usr/sbin/purge", "omp"})
        # subprocess.run(build_argv()): an argv only running the code could name must be mapped, not skipped.
        self.assertEqual(x.unresolved, ["mod.py:42"])
        self.assertEqual(set(x.found["env"]), {"LOCALBENCH_OMP", "PATH", "OMP_PROFILE", "PI_PROFILE", "EXTRA_KEY",
                                               "DEBUG_X", "HOME", "TMPDIR"})
        self.assertEqual(set(x.found["http"]), {"http://127.0.0.1:11434", "/api/ps", "/api/tags", "/chat/completions"})
        self.assertEqual(set(x.found["sqlite"]), {"observe.db"})


if __name__ == "__main__":
    unittest.main()
