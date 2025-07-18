"""Utility functions for PyTorch operations in SpaceTransformer.

This module provides utility functions for PyTorch operations, including
tensor dimension normalization, type conversion, and device management.
"""

from typing import Union, Tuple, Optional
import numpy as np
import torch

TensorLike = Union[np.ndarray, torch.Tensor]


def norm_dim(tensor: TensorLike) -> torch.Tensor:
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
    # Validate input dimensions before processing
    from .validation import validate_tensor
    
    # Basic validation - ensure it's a tensor-like object
    tensor = validate_tensor(tensor, name="input tensor")
    
    if tensor.ndim < 3 or tensor.ndim > 5:
        raise ValueError(f"Expected 3D, 4D or 5D tensor, got {tensor.ndim}D")
    
    # Normalize dimensions
    if tensor.ndim == 3:  # D,H,W → 1,1,D,H,W
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 4:  # C,D,H,W → 1,C,D,H,W
        return tensor.unsqueeze(0)
    else:  # B,C,D,H,W (already 5D)
        return tensor


def norm_type(
    tensor: TensorLike, 
    cuda: bool = False, 
    half: bool = False,
    dtype: Optional[torch.dtype] = None,
    cuda_device: Union[str, torch.device] = "cuda:0"
) -> torch.Tensor:
    """Normalize tensor type, device, and precision.
    
    This function converts the input tensor to the specified type, device,
    and precision, handling both NumPy arrays and PyTorch tensors seamlessly.
    
    Args:
        tensor: Input tensor or array
        cuda: Whether to move tensor to CUDA device
        half: Whether to convert tensor to half precision (float16)
        dtype: Specific dtype to convert tensor to (overrides half)
        cuda_device: CUDA device to use if cuda=True
        
    Returns:
        torch.Tensor: Normalized tensor with specified properties
        
    Example:
        >>> import numpy as np
        >>> array = np.random.rand(100, 100, 50).astype(np.float32)
        >>> tensor = norm_type(array, cuda=True, half=True)
        >>> tensor.device, tensor.dtype
        (device(type='cuda', index=0), torch.float16)
    """
    # First validate and convert to PyTorch tensor
    from .validation import validate_tensor, validate_device
    
    # Set target device
    device = None
    if cuda:
        device = validate_device(cuda_device)
    
    # Set target dtype
    target_dtype = None
    if dtype is not None:
        target_dtype = dtype
    elif half:
        target_dtype = torch.float16
    
    # Validate tensor with device and dtype conversion
    return validate_tensor(tensor, dtype=target_dtype, device=device, name="input tensor") 