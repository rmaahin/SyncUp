"""Root conftest: add backend/src to sys.path so tests can import modules directly."""

import pathlib
import sys

_src = str(pathlib.Path(__file__).resolve().parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)
