"""Make the tutorial package importable when running pytest from anywhere."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "data")
CORPUS = os.path.join(DATA, "sample_corpus.json")
HELDOUT = os.path.join(DATA, "sample_heldout.json")
