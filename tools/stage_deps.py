"""Print Make variables DEPS_<module> listing the epngcat sources each module imports, transitively.

Imports are read statically (including those inside functions), so a stage is rebuilt only when
its own code or code it can reach changes.
"""

import ast
from pathlib import Path

PKG = Path(__file__).resolve().parent.parent / "epngcat"


def direct_imports(path):
    """Names of epngcat modules imported anywhere in one file."""
    mods = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "epngcat":
            parts = node.module.split(".")
            mods |= {parts[1]} if len(parts) > 1 else {a.name for a in node.names}
        elif isinstance(node, ast.Import):
            mods |= {a.name.split(".")[1] for a in node.names if a.name.startswith("epngcat.")}
    return mods


def main():
    """Write one DEPS_<module> line per module to stdout."""
    direct = {p.stem: direct_imports(p) for p in PKG.glob("*.py")}
    for mod in sorted(direct):
        seen, todo = {mod, "__init__"}, [mod]
        while todo:
            for dep in direct.get(todo.pop(), ()):
                if dep not in seen:
                    seen.add(dep)
                    todo.append(dep)
        print(f"DEPS_{mod} := " + " ".join(f"epngcat/{m}.py" for m in sorted(seen) if m in direct))


if __name__ == "__main__":
    main()
