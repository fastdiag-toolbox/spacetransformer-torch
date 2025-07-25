"""Tests for PyTorch validation functions and error handling.

This module tests the validation functions for PyTorch-specific components.
"""

import os
import sys
import unittest
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import from core package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..', 'spacetransformer-core'))
from spacetransformer.core.exceptions import ValidationError, CudaError
from spacetransformer.core import Space

# Import torch package validation
from spacetransformer.torch.validation import (
    validate_device,
    validate_interpolation_mode,
    validate_padding_mode,
    validate_image_tensor
)


class TestTorchValidation(unittest.TestCase):
    """Test suite for PyTorch validation functions."""
    
    def test_validate_device(self):
        """Test device validation."""
        # CPU should always work
        cpu_device = validate_device("cpu")
        self.assertEqual(str(cpu_device), "cpu")
        
        # Test device object
        cpu_obj = validate_device(torch.device("cpu"))
        self.assertEqual(str(cpu_obj), "cpu")
        
        # Test CUDA if available
        if torch.cuda.is_available():
            cuda_device = validate_device("cuda:0")
            self.assertEqual(str(cuda_device), "cuda:0")
        
        # Test invalid device string
        with self.assertRaises(ValidationError):
            validate_device("invalid_device")
        
        # Test invalid type
        with self.assertRaises(ValidationError):
            validate_device(123)
        
        # Test None
        device = validate_device(None)
        self.assertTrue(str(device) == "cuda:0" or str(device) == "cpu")
    
    def test_validate_tensor(self):
        """Test tensor validation with validate_image_tensor."""
        # Test numpy array conversion
        np_array = np.random.rand(10, 10, 10)
        tensor = validate_image_tensor(np_array, return_ndim=False)
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (10, 10, 10))
        
        # Test dimension check
        with self.assertRaises(ValidationError):
            validate_image_tensor(np_array, expected_dim=2, return_ndim=False)
        
        tensor2 = validate_image_tensor(np_array, expected_dim=3, return_ndim=False)
        self.assertEqual(tensor2.ndim, 3)
        
        # Test dtype conversion
        float_tensor = validate_image_tensor(np_array, dtype=torch.float32, return_ndim=False)
        self.assertEqual(float_tensor.dtype, torch.float32)
        
        # Test invalid input
        with self.assertRaises(ValidationError):
            validate_image_tensor("not a tensor", return_ndim=False)
    
    def test_validate_interpolation_mode(self):
        """Test interpolation mode validation."""
        # Test valid modes
        self.assertEqual(validate_interpolation_mode("nearest"), "nearest")
        self.assertEqual(validate_interpolation_mode("bilinear"), "bilinear")
        self.assertEqual(validate_interpolation_mode("trilinear"), "trilinear")
        
        # Test invalid mode
        with self.assertRaises(ValidationError):
            validate_interpolation_mode("invalid_mode")
    
    def test_validate_padding_mode(self):
        """Test padding mode validation."""
        # Test valid modes
        self.assertEqual(validate_padding_mode("border"), "border")
        self.assertEqual(validate_padding_mode("reflection"), "reflection")
        
        
        # Test invalid mode
        with self.assertRaises(ValidationError):
            validate_padding_mode("invalid_mode")
    
    def test_validate_image_tensor(self):
        """Test image tensor validation."""
        # Test 3D image
        image_3d = torch.rand(100, 100, 50)
        tensor_3d, ndim = validate_image_tensor(image_3d)
        self.assertEqual(ndim, 3)
        
        # Test 4D image
        image_4d = torch.rand(3, 100, 100, 50)
        tensor_4d, ndim = validate_image_tensor(image_4d)
        self.assertEqual(ndim, 4)
        
        # Test numpy array
        image_np = np.random.rand(100, 100, 50)
        tensor_np, ndim = validate_image_tensor(image_np)
        self.assertIsInstance(tensor_np, torch.Tensor)
        self.assertEqual(ndim, 3)
        
        # Test invalid dimensions
        with self.assertRaises(ValidationError):
            validate_image_tensor(torch.rand(2, 2), min_dim=3)
        
        # Test NaN values
        image_nan = torch.rand(10, 10, 10)
        image_nan[5, 5, 5] = float('nan')
        with self.assertRaises(ValidationError):
            validate_image_tensor(image_nan)
        
        # Test with dtype
        tensor_half, _ = validate_image_tensor(image_3d, dtype=torch.float16)
        self.assertEqual(tensor_half.dtype, torch.float16)


class TestErrorHandling(unittest.TestCase):
    """Test the error handling for image warping operations."""
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_cuda_error_detection(self):
        """Test CUDA error detection and conversion."""
        from spacetransformer.torch.gpu_utils import handle_cuda_error
        
        # Test with out of memory simulation
        error_msg = "CUDA out of memory. Tried to allocate 20.00 GiB."
        with self.assertRaises(CudaError):
            handle_cuda_error(RuntimeError(error_msg), "test operation")
        
        # Test with device assertion error
        error_msg = "CUDA error: device-side assert triggered"
        with self.assertRaises(CudaError):
            handle_cuda_error(RuntimeError(error_msg), "test operation")
    
    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_validation_error_in_warping(self):
        """Test validation errors in image warping."""
        from spacetransformer.torch.image_warpers import warp_image
        
        # Create test spaces and image
        source = Space(shape=(10, 10, 10))
        target = Space(shape=(5, 5, 5))
        image = torch.rand(10, 10, 10)
        
        # Test with invalid mode
        with self.assertRaises(ValidationError):
            warp_image(image, source, target, pad_value=0, mode="invalid_mode")
        
        # Test with invalid padding mode
        with self.assertRaises(ValidationError):
            warp_image(image, source, target, pad_value=0, pad_mode="invalid_mode")
        
        # Test with invalid image
        with self.assertRaises(ValidationError):
            warp_image("not an image", source, target, pad_value=0)
        
        # Test with invalid source
        with self.assertRaises(ValidationError):
            warp_image(image, "not a space", target, pad_value=0)
        
        # Test with invalid target
        with self.assertRaises(ValidationError):
            warp_image(image, source, "not a space", pad_value=0)


if __name__ == '__main__':
    unittest.main() 