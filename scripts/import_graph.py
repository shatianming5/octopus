"""The in-repo import graph of octopus/ between its packages: runtime, serving, rl, envs.swe, envs.search, eval, tools.

    python scripts/import_graph.py [ROOT]   # ROOT: a checkout (default: this one); exit 1 on a forbidden edge or cycle

Every import counts, a function's own too. A package imports only packages of a lower layer (LAYER): runtime and
serving (0) import neither each other nor anything above; rl (1) the layer below; the environments (2) rl and below,
never each other; eval (3) everything below; tools (4) anything.
"""

from __future__ import annotations

import ast
import collections
import graphlib
import sys
from pathlib import Path

LAYER = {"runtime": 0, "serving": 0, "rl": 1, "envs.swe": 2, "envs.search": 2, "eval": 3, "tools": 4}


def group(module: str) -> str:
    parts = module.split(".")[1:]  # octopus.<...>
    name = ".".join(parts[:2]) if parts[:1] == ["envs"] else parts[0] if parts else ""
    return name if name in LAYER else "other"


def forbidden(a: str, b: str) -> bool:
    return a != b and (a == "other" or b == "other" or LAYER[b] >= LAYER[a])


def imports(path: Path, module: str) -> set[str]:
    out = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            out |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:  # relative
                base = ".".join(module.split(".")[:-node.level] + ([base] if base else []))
            out |= {base} | {f"{base}.{a.name}" for a in node.names}
    return {m for m in out if m.startswith("octopus.")}


def main() -> int:
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1])
    modules = {".".join(p.relative_to(root).with_suffix("").parts): p for p in sorted((root / "octopus").rglob("*.py"))}
    edges, bad = collections.Counter(), []
    for module, path in modules.items():
        # from module import name -> module
        targets = {t if t in modules else t.rpartition(".")[0] for t in imports(path, module)}
        for target in sorted(targets & set(modules) - {module}):
            a, b = group(module), group(target)
            if a != b:
                edges[a, b] += 1
                if forbidden(a, b):
                    bad.append(f"{module} -> {target}")
    for (a, b), n in sorted(edges.items()):
        print(f"{a:>11} -> {b:<11} {n:3d}{'  FORBIDDEN' if forbidden(a, b) else ''}")
    print("\n".join(sorted(set(bad))) or "no forbidden edge")
    deps = collections.defaultdict(set)
    for a, b in edges:
        deps[a].add(b)
    try:
        graphlib.TopologicalSorter(deps).prepare()
        print("no package cycle")
    except graphlib.CycleError as e:
        print(f"package cycle: {e.args[1]}")
        bad.append("cycle")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
