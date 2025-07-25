"""Utility functions for PyTorch operations in SpaceTransformer.

This module provides utility functions for PyTorch operations, including
tensor dimension normalization, type conversion, and device management.
"""

from typing import Union, Tuple, Optional
import numpy as np
import torch

TensorLike = Union[np.ndarray, torch.Tensor]


def norm_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Normalize tensor dimensions to 5D (batch, channel, depth, height, width).
    
    This function converts input tensors of various dimensions to a standard
    5D format used in medical image processing. This simplifies operations by
    ensuring consistent dimension ordering.
    
    Args:
        tensor: Input tensor of dimensions 3D, 4D, or 5D
            - 3D: interpreted as (depth, height, width)
            - 4D: interpreted as (channel, depth, height, width)
            - 5D: interpreted as (batch, channel, depth, height, width)
            
    Returns:
        torch.Tensor: Normalized 5D tensor
        
    Raises:
        ValueError: If input dimensions are invalid (< 3D or > 5D)
        
    Example:
        >>> import torch
        >>> img3d = torch.rand(50, 100, 100)  # D,H,W
        >>> img5d = norm_dim(img3d)
        >>> img5d.shape
        torch.Size([1, 1, 50, 100, 100])
    """
    
    if tensor.ndim < 3 or tensor.ndim > 5:
        raise ValueError(f"Expected 3D, 4D or 5D tensor, got {tensor.ndim}D")
    
    # Normalize dimensions
    if tensor.ndim == 3:  # D,H,W → 1,1,D,H,W
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:  # C,D,H,W → 1,C,D,H,W
        return tensor.unsqueeze(0)
    else:  # B,C,D,H,W (already 5D)
        return tensor


