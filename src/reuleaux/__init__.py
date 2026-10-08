"""reuleaux: exact, backward-stable area of the common overlap of three circles."""
from .core import *  # noqa: F401,F403
from .core import (Arc, Overlap, overlap, overlap_area, overlap_area_batch,  # noqa: F401
                   reference_area, fewell_area, lens_area, pandora_er,
                   pixelart_er, EXAMPLES, HAVE_NUMBA, example_circles, selftest, bench, main)

__version__ = "0.1.0"
