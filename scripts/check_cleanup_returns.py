"""Fail on a bare `return` in cleanup functions and on any `return` in a `finally:`.

Cleanup code must run to the end of the function; guard single steps with `if`, never exit early.
"""

import ast
import re
import sys

CLEANUP_NAME = re.compile(
    r"^_*(destroy|cleanup|close|stop|shutdown|teardown|release|kill|terminate|reap|dispose|rm)(_|$)"
    r"|^__(exit|aexit|del)__$"
)


def returns_in(body: list) -> list:
    """Return statements in `body`, not descending into nested functions or classes."""
    found = []
    stack = list(body)
    while stack:
        node = stack.pop()
        if isinstance(node, ast.Return):
            found.append(node)
        elif not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)
        ):
            stack.extend(ast.iter_child_nodes(node))
    return found


def check(path: str) -> list:
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and CLEANUP_NAME.match(
            node.name
        ):
            problems += [
                f"{path}:{r.lineno}: `return` in cleanup function {node.name}()"
                for r in returns_in(node.body)
                if r.value is None
            ]
        if isinstance(node, ast.Try):
            problems += [
                f"{path}:{r.lineno}: `return` in a finally block"
                for r in returns_in(node.finalbody)
            ]
    return problems


if __name__ == "__main__":
    errors = [problem for path in sys.argv[1:] for problem in check(path)]
    print("\n".join(errors))
    sys.exit(1 if errors else 0)
