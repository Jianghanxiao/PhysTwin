"""Helpers for loading qqtt modules on a machine without a GPU.

Importing the ``qqtt`` package runs ``qqtt/__init__.py``, which imports the
Warp simulator, which calls ``wp.set_device("cuda:0")`` at import time. The
CPU tests therefore load the individual module files directly instead.

To run them: pip install torch warp-lang pytest, then pytest tests
"""

import importlib.util
import pathlib
import sys
import types

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load_module_from_path(name, relative_path):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / relative_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def spring_io():
    return load_module_from_path("phystwin_spring_io", "qqtt/utils/spring_io.py")


@pytest.fixture(scope="session")
def spring_mass_warp():
    """The Warp simulator module, loaded on CPU with a minimal stand in for
    ``qqtt.utils`` (it only needs ``logger`` and ``cfg`` at import time)."""
    wp = pytest.importorskip("warp")
    import logging

    cfg = types.SimpleNamespace(use_graph=True, device="cpu", data_type="synthetic")
    fake_utils = types.ModuleType("qqtt.utils")
    fake_utils.logger = logging.getLogger("qqtt-test")
    fake_utils.cfg = cfg
    fake_pkg = types.ModuleType("qqtt")
    fake_pkg.__path__ = []
    fake_pkg.utils = fake_utils

    saved = {k: sys.modules.get(k) for k in ("qqtt", "qqtt.utils")}
    sys.modules["qqtt"] = fake_pkg
    sys.modules["qqtt.utils"] = fake_utils
    original_set_device = wp.set_device
    wp.set_device = lambda *args, **kwargs: None
    try:
        module = load_module_from_path(
            "phystwin_spring_mass_warp",
            "qqtt/model/diff_simulator/spring_mass_warp.py",
        )
    finally:
        wp.set_device = original_set_device
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value
    wp.set_device("cpu")
    return module
