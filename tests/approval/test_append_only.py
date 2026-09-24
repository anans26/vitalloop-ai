"""§4.6's immutability claim, extended to the Week 9 packages.

The same check `tests/gate/test_persistence.py` runs over `loop/gate/`: strip
docstrings and comments, then assert no executable line could update or delete
a row. A decision is final and a shadow score is a record, so nothing in
`loop/approval/`, `loop/shadow/` or `loop/narrate/` may rewrite either.
"""

import ast
from pathlib import Path

import pytest

from ml.config import PROJECT_ROOT

FORBIDDEN = (
    ".delete(",  # session.delete / Query.delete
    "sqlalchemy.delete",
    "sqlalchemy.update",
    "delete from ",  # raw SQL
    "update ",
)


def executable_code(text: str) -> str:
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        first = node.body[0] if node.body else None
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            node.body.pop(0)
            if not node.body:
                node.body.append(ast.Pass())
    # `unparse` drops comments, so only real statements survive.
    return ast.unparse(tree).lower()


@pytest.mark.parametrize("package", ["approval", "shadow", "narrate"])
def test_no_week_9_module_issues_an_update_or_delete(package):
    paths = sorted((PROJECT_ROOT / "loop" / package).glob("*.py"))
    assert paths
    for path in paths:
        code = executable_code(Path(path).read_text(encoding="utf-8"))
        found = [token for token in FORBIDDEN if token in code]
        assert not found, f"loop/{package}/{path.name} contains {found}"
