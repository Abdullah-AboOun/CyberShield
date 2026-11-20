import importlib

import pytest

PACKAGES = ["api", "auth", "db", "ingestion", "inventory", "detection", "ml", "assistant", "audit", "dashboard"]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name):
    importlib.import_module(f"cybershield.{name}")
