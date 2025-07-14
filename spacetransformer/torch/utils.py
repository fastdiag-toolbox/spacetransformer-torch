"""PyTorch utilities for data type, device, and dimension normalization.

This module provides utility functions for standardizing tensor types, devices,
and dimensions specifically for 3D medical image processing workflows.
Contains only the minimal set required for resampling operations.

Example:
    Normalize tensor format:
    
    >>> import torch
    >>> import numpy as np
    >>> from spacetransformer.torch.utils import norm_dim, norm_type
    >>> 
    >>> # Convert 3D array to 5D tensor format
    >>> image_3d = np.random.rand(100, 100, 50)
    >>> image_5d = norm_dim(image_3d)
    >>> print(image_5d.shape)
    torch.Size([1, 1, 100, 100, 50])
    >>> 
    >>> # Normalize to CUDA tensor
    >>> cuda_tensor = norm_type(image_3d, cuda=True)
    >>> print(cuda_tensor.device)
    cuda:0
"""

from __future__ import annotations

from typing import Union, Any

import numpy as np
import torch

# Type aliases
TensorLike = Union[np.ndarray, torch.Tensor]


# -----------------------------------------------------------------------
# Data type & device handling
# -----------------------------------------------------------------------

def norm_type(
    x: TensorLike,
    *,
    cuda: bool = False,
    half: bool = False,
    dtype: torch.dtype | None = torch.float32,
    cuda_device: torch.device | str | int = "cuda:0",
) -> torch.Tensor:
    """Convert input to tensor with specified dtype and device.
    
    This function converts numpy arrays or tensors to PyTorch tensors with
    the specified data type and device. Provides common conversion patterns
    used in medical image processing.
    
    Args:
        x: Input tensor or array to convert
        cuda: Whether to move tensor to CUDA device
        half: Whether to use float16 precision (overrides dtype)
        dtype: Target data type for the tensor
        cuda_device: CUDA device specification
        
    Returns:
        torch.Tensor: Converted tensor with specified properties
        
    Example:
        Convert numpy array to CUDA tensor:
        
        >>> import numpy as np
        >>> import torch
        >>> arr = np.random.rand(100, 100, 50)
        >>> tensor = norm_type(arr, cuda=True, half=True)
        >>> print(tensor.device, tensor.dtype)
        cuda:0 torch.float16
        
        Convert with specific dtype:
        
        >>> int_arr = np.random.randint(0, 255, (100, 100, 50))
        >>> tensor = norm_type(int_arr, dtype=torch.uint8)
        >>> print(tensor.dtype)
        torch.uint8
    """
    if half:
        dtype = torch.float16

    if not isinstance(x, torch.Tensor):
        # numpy → tensor (preserve original dtype, then convert if needed)
        x = torch.as_tensor(x)

    if cuda and not x.is_cuda:
        x = x.to(cuda_device)

    if dtype is not None and x.dtype != dtype:
        x = x.to(dtype)

    return x


# -----------------------------------------------------------------------
# Dimension normalization
# -----------------------------------------------------------------------

def norm_dim(x: TensorLike) -> torch.Tensor:
    """Ensure tensor has 5D format (B, C, D, H, W).
    
    This function standardizes input tensors to the 5D format expected by
    PyTorch's 3D operations. Supports input dimensions 3D, 4D, and 5D.
    
    Args:
        x: Input tensor or array with 3, 4, or 5 dimensions
        
    Returns:
        torch.Tensor: 5D tensor with shape (B, C, D, H, W)
        
    Raises:
        ValueError: If input tensor dimensions are not 3, 4, or 5
        
    Example:
        Convert different input formats:
        
        >>> import torch
        >>> 
        >>> # 3D input (D, H, W)
        >>> img_3d = torch.rand(50, 100, 100)
        >>> norm_3d = norm_dim(img_3d)
        >>> print(norm_3d.shape)
        torch.Size([1, 1, 50, 100, 100])
        >>> 
        >>> # 4D input (C, D, H, W)
        >>> img_4d = torch.rand(3, 50, 100, 100)
        >>> norm_4d = norm_dim(img_4d)
        >>> print(norm_4d.shape)
        torch.Size([1, 3, 50, 100, 100])
        >>> 
        >>> # 5D input (already correct format)
        >>> img_5d = torch.rand(2, 3, 50, 100, 100)
        >>> norm_5d = norm_dim(img_5d)
        >>> print(norm_5d.shape)
        torch.Size([2, 3, 50, 100, 100])
    """
    if isinstance(x, torch.Tensor):
        pass
    else:
        x = torch.as_tensor(x)

    if x.ndim == 3:  # D,H,W
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 4:  # C,D,H,W
        x = x.unsqueeze(0)
    elif x.ndim == 5:
        pass
    else:
        raise ValueError("Input tensor dimensions must be 3/4/5")
    return x


# -----------------------------------------------------------------------
# Shape / bbox utilities
# -----------------------------------------------------------------------

def to_numpy(a: TensorLike) -> np.ndarray:
    """Safely convert tensor to numpy array (detach + cpu).
    
    This utility function handles the common conversion from PyTorch tensors
    to numpy arrays, ensuring proper detachment from computation graph and
    CPU transfer.
    
    Args:
        a: Input tensor or array
        
    Returns:
        np.ndarray: Numpy array representation
        
    Example:
        Convert CUDA tensor to numpy:
        
        >>> import torch
        >>> cuda_tensor = torch.rand(100, 100, 50, device='cuda')
        >>> numpy_array = to_numpy(cuda_tensor)
        >>> print(type(numpy_array))
        <class 'numpy.ndarray'>
        >>> print(numpy_array.shape)
        (100, 100, 50)
        
        Handle numpy arrays (passthrough):
        
        >>> import numpy as np
        >>> arr = np.random.rand(100, 100, 50)
        >>> result = to_numpy(arr)
        >>> print(arr is result)  # Same object for numpy input
        False  # But same data
    """
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a) 