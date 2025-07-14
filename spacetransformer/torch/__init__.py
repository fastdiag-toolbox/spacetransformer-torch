"""GPU accelerated resampling functionality. Users typically import directly from top-level ``spacetransformer``."""

try:
    from ._version import __version__
except ImportError:
    # Fallback for development installations
    __version__ = "dev"

from .affine_builder import build_grid  # noqa: F401
from .image_warpers import (
    warp_image,  # noqa: F401
    warp_image_batch,  # noqa: F401
    warp_image_with_argmax,  # noqa: F401
    warp_dcb_image,  # noqa: F401
)

__all__ = [
    "build_grid",
    "warp_image",
    "warp_image_batch",
    "warp_image_with_argmax",
    "warp_dcb_image",
] 