"""
data/synthetic/generate_lincs_library.py

Re-export shim so `from data.synthetic.generate_lincs_library import generate_library`
works in tests — the actual implementation lives in discovery/generate_lincs_library.py.
"""
from discovery.generate_lincs_library import generate_library  # noqa: F401
