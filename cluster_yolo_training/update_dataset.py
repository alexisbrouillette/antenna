#!/usr/bin/env python3
"""
Wrapper script to update the training dataset:
Invokes build_dataset.py to reconstruct the dataset from all master and verified annotations.
"""

import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BUILD_SCRIPT = os.path.join(SCRIPT_DIR, "build_dataset.py")

if __name__ == "__main__":
    ret = subprocess.call([sys.executable, BUILD_SCRIPT])
    sys.exit(ret)
