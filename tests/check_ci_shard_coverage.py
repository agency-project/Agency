"""Fail when a test file under tests/sandbox or tests/harness is missing
from the CI shard list.

The CI harness job runs tests/ci_shards.py's explicit file list, not the
directories -- a new test file that is not added there silently loses all
CI coverage. This check keeps the list honest: every tests/sandbox/** and
tests/harness/** test_*.py must be named by some shard, and every shard
entry must point at an existing file (tests/ci_shards.py already raises on
unknown shard names, but not on missing files).

Shared helper modules that merely begin with test_ (imported by the real
test files) are still listed in the shard entries, so they are covered by
the same existence check.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# The directories the general CI job ignores -- exactly the ones the shard
# job must cover.
SHARDED_DIRS = ("tests/sandbox", "tests/harness")


def check() -> "list[str]":
    sys.path.insert(0, str(REPO))
    from tests.ci_shards import _SANDBOX_SHARD_FILES

    shard_files = {file for files in _SANDBOX_SHARD_FILES.values() for file in files}

    on_disk = {
        str(path.relative_to(REPO))
        for directory in SHARDED_DIRS
        for path in (REPO / directory).rglob("test_*.py")
    }

    problems = []
    for path in sorted(on_disk - shard_files):
        problems.append(f"test file not in any CI shard: {path}")
    for path in sorted(shard_files - on_disk):
        problems.append(f"shard lists a file that no longer exists: {path}")
    return problems


if __name__ == "__main__":
    errors = check()
    print("\n".join(errors))
    sys.exit(1 if errors else 0)
