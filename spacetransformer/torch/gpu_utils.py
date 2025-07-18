"""GPU error handling utilities for SpaceTransformer PyTorch operations.

This module provides utilities for handling CUDA and GPU-related errors in
PyTorch operations, converting low-level GPU errors into clear, actionable
error messages for users.

Example:
    Handling CUDA errors in image processing:
    
    >>> try:
    ...     # Some GPU operation that might fail
    ...     result = torch.cuda.operation()
    ... except RuntimeError as e:
    ...     handle_cuda_error(e, "image warping")
    CudaError: GPU out of memory during image warping. Try reducing batch size,
               using smaller images, or switching to CPU processing.
"""

from typing import NoReturn
import torch

# Import from core package
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..', 'spacetransformer-core'))
from spacetransformer.core.exceptions import CudaError


def handle_cuda_error(error: Exception, operation: str) -> NoReturn:
    """Convert CUDA errors to clear, informative error messages.
    
    This function analyzes CUDA runtime errors and converts them into
    user-friendly error messages with specific suggestions for resolution.
    It handles common GPU issues encountered in medical image processing.
    
    Args:
        error: The original CUDA or PyTorch error
        operation: Description of the operation that failed (e.g., "image warping")
        
    Raises:
        CudaError: Always raises with clear error message and suggestions
        
    Example:
        Converting out-of-memory errors:
        
        >>> try:
        ...     large_tensor = torch.zeros(10000, 10000, 10000).cuda()
        ... except RuntimeError as e:
        ...     handle_cuda_error(e, "tensor allocation")
        CudaError: GPU out of memory during tensor allocation. Try reducing batch size,
                   using smaller images, or switching to CPU processing.
    """
    error_msg = str(error).lower()
    original_error = str(error)
    
    if "out of memory" in error_msg or "cuda out of memory" in error_msg:
        # Extract memory information if available
        memory_info = ""
        if torch.cuda.is_available():
            try:
                current_memory = torch.cuda.memory_allocated() / 1024**3  # GB
                max_memory = torch.cuda.max_memory_allocated() / 1024**3  # GB
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
                memory_info = (f" Current GPU memory usage: {current_memory:.1f}GB, "
                             f"Peak: {max_memory:.1f}GB, Total: {total_memory:.1f}GB.")
            except:
                pass
        
        raise CudaError(
            f"GPU out of memory during {operation}.{memory_info} "
            f"Suggestions: "
            f"1) Reduce image size or batch size, "
            f"2) Use CPU processing by setting cuda_device='cpu', "
            f"3) Enable half-precision with half=True, "
            f"4) Process images in smaller chunks. "
            f"Original error: {original_error}"
        )
    
    elif "device-side assert" in error_msg or "device assertion" in error_msg:
        raise CudaError(
            f"GPU kernel assertion failed during {operation}. "
            f"This usually indicates invalid tensor indices, out-of-bounds access, "
            f"or incompatible tensor operations. "
            f"Suggestions: "
            f"1) Check tensor shapes and indices are within bounds, "
            f"2) Verify input data ranges are valid, "
            f"3) Enable CUDA debugging with CUDA_LAUNCH_BLOCKING=1. "
            f"Original error: {original_error}"
        )
    
    elif "no kernel image is available" in error_msg or "invalid device function" in error_msg:
        raise CudaError(
            f"CUDA kernel compatibility error during {operation}. "
            f"The GPU compute capability may not support the required operations. "
            f"Suggestions: "
            f"1) Update GPU drivers to the latest version, "
            f"2) Check PyTorch CUDA version compatibility, "
            f"3) Use CPU processing as fallback. "
            f"Original error: {original_error}"
        )
    
    elif "device" in error_msg and ("not found" in error_msg or "unavailable" in error_msg):
        available_devices = []
        if torch.cuda.is_available():
            available_devices = [f"cuda:{i}" for i in range(torch.cuda.device_count())]
        
        device_info = f" Available devices: {available_devices}" if available_devices else " No CUDA devices available."
        
        raise CudaError(
            f"CUDA device error during {operation}.{device_info} "
            f"Suggestions: "
            f"1) Check if CUDA is properly installed, "
            f"2) Verify GPU is detected by running 'nvidia-smi', "
            f"3) Use CPU processing by setting cuda_device='cpu'. "
            f"Original error: {original_error}"
        )
    
    elif "different devices" in error_msg or "device mismatch" in error_msg:
        raise CudaError(
            f"Tensor device mismatch during {operation}. "
            f"All tensors in an operation must be on the same device (CPU or same GPU). "
            f"Suggestions: "
            f"1) Move all tensors to the same device using .to(device), "
            f"2) Check input tensor devices before operations, "
            f"3) Use consistent device specification throughout the pipeline. "
            f"Original error: {original_error}"
        )
    
    elif "cuda" in error_msg or "gpu" in error_msg:
        # Generic CUDA error
        cuda_available = torch.cuda.is_available()
        driver_info = ""
        if cuda_available:
            try:
                driver_info = f" CUDA version: {torch.version.cuda}, Driver version available."
            except:
                driver_info = " CUDA version information unavailable."
        else:
            driver_info = " CUDA not available on this system."
        
        raise CudaError(
            f"CUDA error during {operation}.{driver_info} "
            f"Suggestions: "
            f"1) Check GPU availability and drivers, "
            f"2) Verify PyTorch CUDA installation, "
            f"3) Try CPU processing as fallback, "
            f"4) Check system compatibility. "
            f"Original error: {original_error}"
        )
    
    else:
        # Non-CUDA GPU error or unknown error
        raise CudaError(
            f"GPU operation failed during {operation}: {original_error}. "
            f"This may be a PyTorch or hardware-related issue. "
            f"Suggestions: "
            f"1) Try CPU processing as fallback, "
            f"2) Check PyTorch installation and version compatibility, "
            f"3) Verify input data validity."
        )


def check_cuda_availability() -> bool:
    """Check if CUDA is available and provide helpful information if not.
    
    This function checks CUDA availability and provides detailed information
    about the GPU setup, helping users understand their hardware configuration.
    
    Returns:
        bool: True if CUDA is available and working, False otherwise
        
    Example:
        Checking CUDA before GPU operations:
        
        >>> if check_cuda_availability():
        ...     device = "cuda:0"
        ... else:
        ...     device = "cpu"
        ...     print("Using CPU processing")
    """
    if not torch.cuda.is_available():
        return False
    
    try:
        # Test basic CUDA operation
        test_tensor = torch.tensor([1.0]).cuda()
        _ = test_tensor + 1
        return True
    except Exception:
        return False


def get_gpu_memory_info() -> dict:
    """Get current GPU memory usage information.
    
    This function provides detailed information about GPU memory usage,
    which is helpful for debugging memory-related issues.
    
    Returns:
        dict: Dictionary containing memory information in GB, or empty dict if CUDA unavailable
        
    Example:
        Checking memory before large operations:
        
        >>> memory_info = get_gpu_memory_info()
        >>> if memory_info and memory_info['free'] < 2.0:  # Less than 2GB free
        ...     print("Warning: Low GPU memory available")
    """
    if not torch.cuda.is_available():
        return {}
    
    try:
        device = torch.cuda.current_device()
        allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
        cached = torch.cuda.memory_reserved(device) / 1024**3  # GB
        total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
        free = total - cached
        
        return {
            'allocated': allocated,
            'cached': cached,
            'total': total,
            'free': free,
            'device': device,
            'device_name': torch.cuda.get_device_name(device)
        }
    except Exception:
        return {}


def validate_tensor_device(tensor: torch.Tensor, expected_device: str) -> None:
    """Validate that a tensor is on the expected device.
    
    This function checks tensor device placement and raises a clear error
    if the tensor is on the wrong device.
    
    Args:
        tensor: PyTorch tensor to check
        expected_device: Expected device string (e.g., "cuda:0", "cpu")
        
    Raises:
        CudaError: If tensor is on wrong device
        
    Example:
        Validating tensor placement:
        
        >>> tensor = torch.rand(100, 100).cuda()
        >>> validate_tensor_device(tensor, "cuda:0")  # No error
        >>> validate_tensor_device(tensor, "cpu")  # Raises CudaError
    """
    actual_device = str(tensor.device)
    
    if actual_device != expected_device:
        raise CudaError(
            f"Tensor device mismatch: expected {expected_device}, got {actual_device}. "
            f"Move tensor to correct device using tensor.to('{expected_device}') "
            f"or ensure consistent device usage throughout the operation."
        )