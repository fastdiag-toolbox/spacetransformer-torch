"""Validation utilities for PyTorch-specific operations.

This module provides validation functions for PyTorch tensors, GPU devices,
and other parameters specific to GPU-accelerated operations in SpaceTransformer.

Example:
    Validating tensors in a PyTorch function:
    
    >>> from spacetransformer.torch.validation import validate_image_tensor, validate_device
    >>> def process_image(image, device="cuda:0"):
    ...     image_tensor = validate_image_tensor(image, expected_dim=5, return_ndim=False)
    ...     device = validate_device(device)
    ...     # Proceed with validated inputs
"""

from typing import Any, Optional, Tuple, Union
import numpy as np
import torch

from spacetransformer.core.exceptions import ValidationError, CudaError


def validate_device(device: Any) -> torch.device:
    """Validate and return a PyTorch device.
    
    Args:
        device: Device specification (string or torch.device)
        
    Returns:
        torch.device: Validated PyTorch device
        
    Raises:
        ValidationError: If device is invalid
        CudaError: If CUDA device is specified but not available
        
    Example:
        >>> validate_device("cpu")
        device(type='cpu')
        >>> validate_device("cuda:0")  # Will raise CudaError if CUDA unavailable
    """
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        
    try:
        if isinstance(device, str):
            if device.startswith("cuda") and not torch.cuda.is_available():
                available_devices = []
                device_info = " No CUDA devices available."
                raise CudaError(
                    f"CUDA device '{device}' requested but CUDA is not available.{device_info} "
                    f"Use 'cpu' device or install CUDA."
                )
            return torch.device(device)
        elif isinstance(device, torch.device):
            if device.type == "cuda" and not torch.cuda.is_available():
                raise CudaError(
                    f"CUDA device requested but CUDA is not available. "
                    f"Use 'cpu' device or install CUDA."
                )
            return device
        else:
            raise ValidationError(
                f"Device must be a string or torch.device, got {type(device).__name__}"
            )
    except (ValueError, RuntimeError) as e:
        if "Invalid device string" in str(e) or "CUDA error" in str(e):
            raise CudaError(f"Invalid CUDA device: {e}")
        raise ValidationError(f"Invalid device specification: {e}")


def validate_interpolation_mode(mode: str, name: str = "mode") -> str:
    """Validate interpolation mode for grid_sample and other operations.
    
    Args:
        mode: Interpolation mode string
        name: Parameter name for error messages
        
    Returns:
        str: Validated interpolation mode
        
    Raises:
        ValidationError: If mode is invalid
        
    Example:
        >>> validate_interpolation_mode("trilinear")
        'trilinear'
        >>> validate_interpolation_mode("invalid")  # Will raise ValidationError
    """
    valid_modes = ["nearest", "bilinear", "trilinear", "bicubic"]
    if mode not in valid_modes:
        raise ValidationError(
            f"{name} must be one of {valid_modes}, got '{mode}'"
        )
    return mode


def validate_padding_mode(mode: str, name: str = "pad_mode") -> str:
    """Validate padding mode for grid_sample and other operations.
    
    Args:
        mode: Padding mode string
        name: Parameter name for error messages
        
    Returns:
        str: Validated padding mode
        
    Raises:
        ValidationError: If mode is invalid
        
    Example:
        >>> validate_padding_mode("zeros")
        'zeros'
        >>> validate_padding_mode("invalid")  # Will raise ValidationError
    """
    valid_modes = ["border", "reflection", "constant"]
    
    
    if mode not in valid_modes:
        raise ValidationError(
            f"{name} must be one of {valid_modes}, got '{mode}'"
        )
    return mode


def validate_image_tensor(
    tensor: Any,
    expected_dim: Optional[int] = None,
    min_dim: int = 3,
    max_dim: int = 5,
    dtype: Optional[torch.dtype] = None,
    device: Optional[Union[str, torch.device]] = None,
    name: str = "tensor",
    return_ndim: bool = True
) -> Union[torch.Tensor, Tuple[torch.Tensor, int]]:
    """Validate and optionally convert input to PyTorch tensor.
    
    This function combines general tensor validation and image-specific validation.
    
    Args:
        tensor: Input tensor or array-like object
        expected_dim: Expected exact number of dimensions (None to use min_dim/max_dim range)
        min_dim: Minimum allowed dimensions when expected_dim is None (default 3 for 3D medical images)
        max_dim: Maximum allowed dimensions when expected_dim is None (default 5 for batched images)
        dtype: Target data type (None to keep original)
        device: Target device (None to keep original)
        name: Parameter name for error messages
        return_ndim: Whether to return original dimensions count as second return value
        
    Returns:
        If return_ndim is True:
            Tuple[torch.Tensor, int]: (Validated PyTorch tensor, original dimensions)
        If return_ndim is False:
            torch.Tensor: Validated PyTorch tensor
        
    Raises:
        ValidationError: If tensor is invalid
        CudaError: If CUDA device is specified but not available
        
    Example:
        >>> import numpy as np
        >>> image = np.random.rand(100, 100, 50)
        >>> tensor, ndim = validate_image_tensor(image)  # Image validation with dimensions
        >>> tensor = validate_image_tensor(image, expected_dim=3, return_ndim=False)  # General validation
        >>> tensor = validate_image_tensor(image, dtype=torch.float32, device="cuda:0", return_ndim=False)
    """
    try:
        if not isinstance(tensor, (np.ndarray, torch.Tensor, list, tuple)):
            raise ValidationError(
                f"{name} must be a tensor, array, or list/tuple, got {type(tensor).__name__}"
            )
            
        # Convert to torch.Tensor if necessary
        if not isinstance(tensor, torch.Tensor):
            try:
                tensor = torch.as_tensor(tensor)
            except (ValueError, RuntimeError, TypeError) as e:
                raise ValidationError(f"Could not convert {name} to tensor: {e}")
        
        # Store original dimensions
        original_ndim = tensor.ndim
        
        # Check dimensions - either exact match or range
        if expected_dim is not None:
            if tensor.ndim != expected_dim:
                raise ValidationError(
                    f"{name} must have {expected_dim} dimensions, got {tensor.ndim}"
                )
        else:
            # Check dimension range
            if tensor.ndim < min_dim or tensor.ndim > max_dim:
                raise ValidationError(
                    f"{name} must have between {min_dim} and {max_dim} dimensions, "
                    f"got {tensor.ndim}"
                )
        
        # Convert dtype if specified
        if dtype is not None:
            try:
                tensor = tensor.to(dtype)
            except RuntimeError as e:
                raise ValidationError(f"Could not convert {name} to {dtype}: {e}")
        
        # Move to device if specified
        if device is not None:
            try:
                device = validate_device(device)
                tensor = tensor.to(device)
            except CudaError as e:
                raise e
            except RuntimeError as e:
                raise CudaError(f"Device transfer error for {name}: {e}")
        
        # Check for any nan/inf values
        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            raise ValidationError(
                f"{name} contains NaN or infinite values"
            )
        
        return (tensor, original_ndim) if return_ndim else tensor
    except CudaError:
        # Let CUDA errors pass through
        raise
    except Exception as e:
        if isinstance(e, ValidationError):
            raise
        raise ValidationError(f"Invalid {name}: {e}") 