#!/usr/bin/env python3
"""Host-side wrapper so tag regeneration works without a colcon install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wojtek_benchmark.generate_tags import main

if __name__ == "__main__":
    sys.exit(main())
