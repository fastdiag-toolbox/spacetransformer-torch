"""Grid generation utilities for PyTorch grid_sample operations.

This module provides functions for generating 5D grids compatible with
PyTorch's grid_sample function, supporting both regular and half-precision
computations for 3D medical image processing.

Example:
    Generate a grid for image sampling:
    
    >>> import torch
    >>> from spacetransformer.torch.affine_builder import build_grid
    >>> 
    >>> # Create affine transformation matrix
    >>> theta = torch.eye(3, 4)  # Identity transformation
    >>> shape = (100, 100, 50)
    >>> 
    >>> # Build grid for sampling
    >>> grid = build_grid(theta, shape)
    >>> print(grid.shape)
    torch.Size([1, 100, 100, 50, 3])
"""

from __future__ import annotations

from typing import Tuple

import torch

__all__ = ["build_grid"]


def _make_base_grid(shape: Tuple[int, int, int], device, dtype):
    """Generate normalized coordinate grid in NDC space (-1, 1).
    
    This function creates a base coordinate grid with normalized device
    coordinates ranging from -1 to 1, compatible with PyTorch's grid_sample.
    
    Args:
        shape: Target volume dimensions (D, H, W)
        device: Device for tensor creation
        dtype: Data type for tensor creation
        
    Returns:
        torch.Tensor: Coordinate grid with shape (D, H, W, 3)
        
    Example:
        >>> device = torch.device('cuda')
        >>> dtype = torch.float32
        >>> grid = _make_base_grid((50, 100, 100), device, dtype)
        >>> print(grid.shape)
        torch.Size([50, 100, 100, 3])
    """
    D, H, W = shape
    # Note: torch.linspace is closed interval → align_corners=True maps [0,D-1] → [-1,1]
    zs = torch.linspace(-1.0, 1.0, steps=D, device=device, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, steps=H, device=device, dtype=dtype)
    xs = torch.linspace(-1.0, 1.0, steps=W, device=device, dtype=dtype)
    z, y, x = torch.meshgrid(zs, ys, xs, indexing="ij")  # D,H,W
    grid = torch.stack((x, y, z), dim=-1)  # D,H,W,3  (x,y,z) order
    return grid


def build_grid(theta: torch.Tensor, shape: Tuple[int, int, int], *, half: bool = False) -> torch.Tensor:
    """Generate 5D grid for PyTorch grid_sample operations.
    
    This function creates a 5D sampling grid by applying affine transformations
    to a normalized base grid. The grid is suitable for direct use with
    PyTorch's F.grid_sample function.
    
    Args:
        theta: Affine transformation matrix(es) with shape (3, 4) or (N, 3, 4).
               Row order is fixed as (x, y, z)
        shape: Target volume dimensions (D, H, W)
        half: Whether to use float16 precision for the grid
        
    Returns:
        torch.Tensor: 5D sampling grid with shape (N, D, H, W, 3)
        
    Example:
        Single transformation:
        
        >>> import torch
        >>> theta = torch.eye(3, 4)  # Identity transformation
        >>> grid = build_grid(theta, (50, 100, 100))
        >>> print(grid.shape)
        torch.Size([1, 50, 100, 100, 3])
        
        Batch of transformations:
        
        >>> batch_theta = torch.eye(3, 4).unsqueeze(0).repeat(5, 1, 1)
        >>> batch_grid = build_grid(batch_theta, (50, 100, 100))
        >>> print(batch_grid.shape)
        torch.Size([5, 50, 100, 100, 3])
        
        Half precision for memory efficiency:
        
        >>> half_grid = build_grid(theta, (50, 100, 100), half=True)
        >>> print(half_grid.dtype)
        torch.float16
    """
    if theta.ndim == 2:
        theta = theta.unsqueeze(0)
    assert theta.shape[1:] == (3, 4), "theta shape must be (3,4) or (N,3,4)"

    N = theta.shape[0]
    device = theta.device
    dtype = torch.float16 if half else theta.dtype

    base_grid = _make_base_grid(shape, device, dtype)  # D,H,W,3 in ndc
    base_grid = base_grid.reshape(-1, 3).T  # 3,L

    # Construct homogeneous coordinates (3,L) → (4,L)
    ones = torch.ones((1, base_grid.shape[1]), device=device, dtype=dtype)
    hom = torch.cat((base_grid, ones), dim=0)  # 4,L

    # θ is (N,3,4) → (N,3,4) @ (4,L) = (N,3,L)
    warped = torch.bmm(theta.to(dtype), hom.expand(N, *hom.shape))  # (N,3,L)
    warped = warped.transpose(1, 2)  # N,L,3
    grid = warped.view(N, *shape, 3)
    return grid 