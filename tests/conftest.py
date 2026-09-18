import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pytest_configure():
    # Point every module at a throwaway state dir before llmctl.paths is
    # imported, so tests never touch the real registry or manifest.
    import tempfile

    tmp = tempfile.mkdtemp(prefix="llmctl-test-")
    os.environ["DEKHO_RUN_DIR"] = tmp
    os.environ["DEKHO_MANIFEST"] = str(Path(tmp) / "manifest.json")
