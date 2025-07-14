"""GPU-accelerated image resampling for 3D medical images.

This module provides efficient GPU-accelerated image warping capabilities for
3D medical images using PyTorch. It supports various interpolation modes and
optimization strategies for different transformation scenarios.

Design Philosophy:
    The core workflow follows A(source) → C → D → B(target) pattern:
    - A→C: Only cropping/padding (_trans_shift)
    - C→D: Either flip/permute operations or general grid_sample (_trans_general)
    - D→B: Same as A→C (_trans_shift)
    
    This approach optimizes performance by using fast tensor operations when
    possible and falling back to general interpolation only when necessary.

Example:
    Basic image warping between spaces:
    
    >>> import torch
    >>> import numpy as np
    >>> from spacetransformer.core import Space
    >>> from spacetransformer.torch.image_warpers import warp_image
    >>> 
    >>> # Create test image and spaces
    >>> image = torch.rand(100, 100, 50)
    >>> source = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
    >>> target = Space(shape=(50, 50, 25), spacing=(2.0, 2.0, 4.0))
    >>> 
    >>> # Warp image
    >>> warped = warp_image(image, source, target, pad_value=0.0)
    >>> print(warped.shape)
    torch.Size([50, 50, 25])
"""

from __future__ import annotations

from typing import Any, List, Optional, Union

import numpy as np
import torch
from torch.nn import functional as F

from spacetransformer.core.space import Space
from spacetransformer.core import relation_check as rc
from spacetransformer.core.pointset_warpers import calc_transform

from .affine_builder import build_grid
from .utils import norm_dim, norm_type

Array = Union[np.ndarray, List[float], List[int], tuple]
TensorLike = Union[np.ndarray, torch.Tensor]

__all__ = [
    "warp_image",
    "warp_image_batch",
    "warp_image_with_argmax",
    "warp_dcb_image",
]


# -------------------------------------------------------------------------
# Basic utilities
# -------------------------------------------------------------------------

def __normback(output: torch.Tensor | np.ndarray, numpy: bool, ndim: int, device: Any):
    """Restore output dimensions, type, and device.
    
    This helper function converts the output back to the expected format,
    removing batch/channel dimensions and converting between tensor types.
    
    Args:
        output: Output tensor or array
        numpy: Whether to return numpy array
        ndim: Original input dimensions
        device: Target device for tensor output
        
    Returns:
        Processed output in the correct format
    """
    if numpy and isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    # Remove batch/channel dimensions
    if ndim == 3:
        output = output[0, 0]
    elif ndim == 4:
        output = output[0]
    if isinstance(output, torch.Tensor) and device is not None:
        output = output.to(device)
    return output


# -------------------------------------------------------------------------
# Cropping and padding operations
# -------------------------------------------------------------------------

def crop(
    img: torch.Tensor | np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    *,
    pad_mode: str = "constant",
    pad_value: float = 0.0,
):
    """Crop tensor with automatic padding for out-of-bounds regions.
    
    This function performs cropping on arbitrary dimension tensors with support
    for automatic padding when the crop region extends beyond the tensor bounds.
    Input and output dimensions remain unchanged.
    
    Args:
        img: Input tensor or array
        start: Start indices for cropping
        end: End indices for cropping (exclusive)
        pad_mode: Padding mode for out-of-bounds regions
        pad_value: Padding value for constant mode
        
    Returns:
        Cropped tensor with same type as input
        
    Example:
        >>> import torch
        >>> import numpy as np
        >>> img = torch.rand(1, 1, 100, 100, 50)
        >>> start = np.array([10, 20, 5])
        >>> end = np.array([90, 80, 45])
        >>> cropped = crop(img, start, end)
        >>> print(cropped.shape)
        torch.Size([1, 1, 80, 60, 40])
    """
    ndim = img.ndim
    start = start.astype(int)
    end = end.astype(int)
    assert len(start) == len(end)
    assert np.all(end > start), "end must be greater than start"

    if len(start) == ndim - 1:
        # (C,D,H,W) → fake batch
        start = np.concatenate(([0], start))
        end = np.concatenate(([img.shape[0]], end))
        pad_cutoff = 1
    elif len(start) == ndim - 2:
        start = np.concatenate(([0, 0], start))
        end = np.concatenate(([img.shape[0], img.shape[1]], end))
        pad_cutoff = 2
    else:
        pad_cutoff = 0

    left_pad = np.where(start < 0, -start, 0)
    right_pad = np.where(end > np.array(img.shape), end - np.array(img.shape), 0)

    slices = [slice(s + lp, e - rp) for s, e, lp, rp in zip(start, end, left_pad, right_pad)]
    img_crop = img[tuple(slices)]

    need_pad = np.any(left_pad + right_pad)
    if need_pad:
        if isinstance(img_crop, torch.Tensor):
            pads = []
            for l, r in (list(zip(left_pad, right_pad))[pad_cutoff:][::-1]):
                pads.extend([int(l), int(r)])
            img_crop = F.pad(img_crop, pads, mode=pad_mode, value=pad_value)
        else:
            pad_width = list(zip(left_pad, right_pad))
            img_crop = np.pad(img_crop, pad_width, mode=pad_mode, constant_values=pad_value)
    return img_crop


def _trans_shift(
    img: torch.Tensor,
    source: Space,
    target: Space,
    *,
    pad_mode: str,
    pad_value: float,
) -> torch.Tensor:
    """仅支持 base/spacing 完全一致且 align_corners=True 的纯平移裁剪。"""
    img = norm_type(img, dtype=None)  # 保持 dtype
    R = source._orientation_matrix()
    M = R * np.array(source.spacing)[None, :]
    offset_origin = np.round(np.linalg.solve(M, np.array(target.origin) - np.array(source.origin))).astype(int)
    offset_end = np.round(np.linalg.solve(M, np.array(target.end) - np.array(source.origin))).astype(int) + 1
    return crop(img, offset_origin, offset_end, pad_mode=pad_mode, pad_value=pad_value)


# -------------------------------------------------------------------------
# flip / permute ----------------------------------------------------------
# -------------------------------------------------------------------------

def _trans_flip(img: torch.Tensor, flip_dims: List[int]):
    img = norm_type(img, dtype=None)
    dims = []
    for idx in range(3):
        if flip_dims[idx]:
            dims.append(idx - 3)  # 转到负索引, 因为后面要对应 D,H,W (倒数123位)
    if dims:
        img = torch.flip(img, dims)
    return img


def _trans_permute(img: torch.Tensor, axis_order: List[int]):
    # batch(0) channel(1) 保持，后面 +2
    order = [0, 1] + [ax + 2 for ax in axis_order]
    return img.permute(order)


# -------------------------------------------------------------------------
# 通用 grid_sample -------------------------------------------------------
# -------------------------------------------------------------------------

def _do_warping(img: torch.Tensor, grid: torch.Tensor, *, mode: str, pad_mode: str, pad_value: float):
    if mode == "trilinear":
        mode = "bilinear"
    if pad_mode == "constant":
        warped = (
            F.grid_sample(img - pad_value, grid, mode=mode, padding_mode="zeros", align_corners=True) + pad_value
        )
    else:
        warped = F.grid_sample(img, grid, mode=mode, padding_mode=pad_mode, align_corners=True)
    return warped


# -----------------------------------------------------------------------------
# 内部几何辅助：index ↔ ndc 变换矩阵 & θ(target.ndc→source.ndc)
# -----------------------------------------------------------------------------


def to_ndc_space(space:Space) -> Space:

    """
    Returns the NDC (Normalized Device Coordinate) proxy space of the current space.
    This space simplifies various PyTorch operations.

    Returns:
        Space: The NDC proxy space of the current space.
    """
    
    new_origin = np.array(space.origin) + np.array(space.physical_span) / 2
    new_shape = (2, 2, 2)
    new_spacing = np.array(space.spacing) * (np.array(space.shape) - 1) / 2
    return Space(origin=new_origin, spacing=new_spacing, shape=new_shape,
                 x_orientation=space.x_orientation,
                 y_orientation=space.y_orientation,
                 z_orientation=space.z_orientation)
    

def _calc_theta_ndc(source: Space, target: Space) -> np.ndarray:
    """计算 target.ndc → source.ndc 的 3×4 θ 矩阵 (float32)。"""
    # target.index → source.index
    source = to_ndc_space(source)
    target = to_ndc_space(target)
    theta_4x4 = calc_transform(target,source).matrix  # 4×4
    theta_3x4 = theta_4x4[:3].astype(np.float32)
    theta_3x4 = theta_3x4[[2, 1, 0]][:, [2, 1, 0, 3]]
    return theta_3x4


def _trans_general(
    img: torch.Tensor,
    source: Space,
    target: Space,
    *,
    mode: str,
    pad_mode: str,
    pad_value: float,
    half: bool = False,
) -> torch.Tensor:
    """通用路径：一次 grid_sample 完成任意仿射/旋转/缩放。"""
    img = norm_type(img, cuda=True, half=half)

    theta_np = _calc_theta_ndc(source, target)  # 3×4
    theta = torch.from_numpy(theta_np).to(img.device)
    if half:
        theta = theta.half()

    grid = build_grid(theta, target.shape, half=half)
    return _do_warping(img, grid, mode=mode, pad_mode=pad_mode, pad_value=pad_value)

def _trans_zoom(img: torch.Tensor, source: Space, target: Space, pad_mode: str, pad_value: float, mode: str, half: bool):
    if mode == 'nearest':
        mode = 'nearest-exact'
    return F.interpolate(img, target.shape, align_corners=True, mode=mode)
# -------------------------------------------------------------------------
# 空输出 ------------------------------------------------------------------
# -------------------------------------------------------------------------

def _trans_empty(img: torch.Tensor | np.ndarray, target: Space, pad_value: float):
    """返回全填充值的空图像 (shape 与 target 一致)。"""
    if isinstance(img, torch.Tensor):
        channel = img.shape[1] if img.ndim == 5 else 1
        return torch.full([1, channel] + list(target.shape), pad_value, dtype=img.dtype, device=img.device)
    else:
        channel = 1 if img.ndim == 3 else img.shape[0] if img.ndim == 4 else img.shape[2]
        return np.full([1, channel] + list(target.shape), pad_value, dtype=img.dtype)


# -------------------------------------------------------------------------
# 主接口 ------------------------------------------------------------------
# -------------------------------------------------------------------------

def warp_image(
    img: TensorLike,
    source: Space,
    target: Space,
    *,
    pad_value: float,
    mode: str = "trilinear",
    pad_mode: str = "constant",
    half: bool = False,
    numpy: bool = False,
    cuda_device: torch.device | str = "cuda:0",
) -> TensorLike:
    """Resample image from source space to target space using GPU acceleration.
    
    This function performs efficient 3D image resampling from source to target
    coordinate space using various optimization strategies based on the geometric
    relationship between the spaces.
    
    The function automatically chooses the best strategy:
    - Direct copy for identical spaces
    - Empty output for non-overlapping spaces  
    - Fast tensor operations for flip/permute transformations
    - Optimized interpolation for zoom operations
    - General grid sampling for arbitrary transformations
    
    Args:
        img: Input image tensor or array. Supports 3D (D,H,W), 4D (C,D,H,W), 
             or 5D (B,C,D,H,W) formats
        source: Source geometric space defining input image coordinates
        target: Target geometric space for output image coordinates
        pad_value: Padding value for regions outside source image bounds
        mode: Interpolation mode ("trilinear", "nearest", "bicubic")
        pad_mode: Padding mode for boundary handling ("constant", "reflect", etc.)
        half: Whether to use half-precision (float16) for computation
        numpy: Whether to return numpy array instead of tensor
        cuda_device: CUDA device for GPU computation
        
    Returns:
        Resampled image in target space with same type as input (unless numpy=True)
        
    Raises:
        ValueError: If input dimensions are invalid
        RuntimeError: If CUDA operations fail
        
    Example:
        Basic image resampling:
        
        >>> import torch
        >>> from spacetransformer.core import Space
        >>> from spacetransformer.torch.image_warpers import warp_image
        >>> 
        >>> # Create test image and spaces
        >>> image = torch.rand(100, 100, 50)
        >>> source = Space(shape=(100, 100, 50), spacing=(1.0, 1.0, 2.0))
        >>> target = Space(shape=(50, 50, 25), spacing=(2.0, 2.0, 4.0))
        >>> 
        >>> # Resample to target space
        >>> resampled = warp_image(image, source, target, pad_value=0.0)
        >>> print(resampled.shape)
        torch.Size([50, 50, 25])
        
        Using different interpolation modes:
        
        >>> # Nearest neighbor for label images
        >>> labels = torch.randint(0, 5, (100, 100, 50))
        >>> resampled_labels = warp_image(labels, source, target, 
        ...                              pad_value=0, mode="nearest")
        >>> 
        >>> # Half precision for memory efficiency
        >>> resampled_half = warp_image(image, source, target, 
        ...                            pad_value=0.0, half=True)
    """
    # 记录输入信息
    input_ndim = img.ndim
    is_numpy_in = isinstance(img, np.ndarray) or numpy
    origin_device = img.device if isinstance(img, torch.Tensor) else None

    # 完全相同空间：直接返回，避免任何数值误差
    if source == target:
        print("source == target")
        if is_numpy_in:
            if isinstance(img, np.ndarray):
                return img
            else:  # torch.Tensor → numpy 输出
                return img.detach().cpu().numpy()
        else:
            # 输出 torch.Tensor，维持原 device
            return img if isinstance(img, torch.Tensor) else torch.as_tensor(img, device=origin_device)

    # 判断无交集
    if rc._check_no_overlap(source, target):
        tensor_like = img if isinstance(img, torch.Tensor) else torch.as_tensor(img)
        out = _trans_empty(tensor_like, target, pad_value)
        return __normback(out, is_numpy_in, input_ndim, origin_device)

    # 统一成 5D tensor（不立即转 dtype/device）
    img_5d = norm_dim(img)

    # ---------------------------------------------------------------
    # 1) 计算 C、D ----------------------------------------------------
    # ---------------------------------------------------------------
    bbox_B_in_A = rc.find_tight_bbox(target, source)  # B 对 A
    bbox_A_in_B = rc.find_tight_bbox(source, target)  # A 对 B
    if np.prod(bbox_B_in_A[:, 1] - bbox_B_in_A[:, 0]) / np.prod(source.shape) > 0.6:
        C = source
    else:
        C = source.apply_bbox(bbox_B_in_A)
    if np.prod(bbox_A_in_B[:, 1] - bbox_A_in_B[:, 0]) / np.prod(target.shape) > 0.6:
        D = target
    else:
        D = target.apply_bbox(bbox_A_in_B)

    # ---------------------------------------------------------------
    # 2) A → C (shift) ----------------------------------------------
    # ---------------------------------------------------------------
    if C == source:
        img_AC = img_5d  # no-op
    else:
        img_AC = _trans_shift(img_5d, source, C, pad_mode=pad_mode, pad_value=pad_value)

    # ---------------------------------------------------------------
    # 3) C → D -------------------------------------------------------
    # ---------------------------------------------------------------
    if C == D:
        img_CD = img_AC
    else:
        flag_flip, flip_dims, axis_order = rc._check_valid_flip_permute(C, D)
        if flag_flip:
            tmp = _trans_flip(img_AC, flip_dims)
            img_CD = _trans_permute(tmp, axis_order)
        else:  
            if rc._check_align_corner(C, D) and rc._check_same_base(C, D):
                img_CD = _trans_zoom(img_AC, C, D, pad_mode=pad_mode, pad_value=pad_value, mode=mode, half=half)
            else:  
                img_CD = _trans_general(
                    img_AC,
                    C,
                    D,
                    mode=mode,
                    pad_mode=pad_mode,
                    pad_value=pad_value,
                    half=half,
                )

    # ---------------------------------------------------------------
    # 4) D → B (shift) ----------------------------------------------
    # ---------------------------------------------------------------
    if D == target:
        img_DB = img_CD  # no-op
    else:
        img_DB = _trans_shift(img_CD, D, target, pad_mode=pad_mode, pad_value=pad_value)

    # ---------------------------------------------------------------
    # 5) 收尾 --------------------------------------------------------
    # ---------------------------------------------------------------
    output = img_DB
    return __normback(output, is_numpy_in, input_ndim, origin_device)


# -------------------------------------------------------------------------
# warp_image_batch --------------------------------------------------------
# -------------------------------------------------------------------------

def warp_image_batch(
    img: TensorLike,
    source: Space,
    targets: List[Space],
    *,
    pad_value: float,
    mode: str = "trilinear",
    pad_mode: str = "constant",
    half: bool = False,
    cuda_device: torch.device | str = "cuda:0",
) -> List[torch.Tensor]:
    """对同一 *img* 映射到多个目标空间。始终返回 GPU tensor 列表。"""
    with torch.no_grad():
        img_norm = norm_dim(img)
        img_norm = norm_type(img_norm, cuda=True, half=half, cuda_device=cuda_device)
        outs: List[torch.Tensor] = []
        for tgt in targets:
            out = warp_image(
                img_norm,
                source,
                tgt,
                pad_value=pad_value,
                mode=mode,
                pad_mode=pad_mode,
                half=half,
                numpy=False,
                cuda_device=cuda_device,
            )
            outs.append(out)
        return outs


# -------------------------------------------------------------------------
# warp_image_with_argmax --------------------------------------------------
# -------------------------------------------------------------------------

def warp_image_with_argmax(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("warp_image_with_argmax 尚未迁移，可后续按需实现。")


# -------------------------------------------------------------------------
# warp_dcb_image ----------------------------------------------------------
# -------------------------------------------------------------------------

def warp_dcb_image(
    img: "DicomCubeImage",
    target: Space,
    *,
    pad_value: float,
    mode: str = "trilinear",
    pad_mode: str = "constant",
    half: bool = False,
    numpy: bool = False,
    cuda_device: torch.device | str = "cuda:0",
) -> "DicomCubeImage":
    from dicube import DicomCubeImage
    """将 DicomCubeImage 从其内置空间重采样到目标空间。
    
    Args:
        img: 输入的 DicomCubeImage，使用其内置的 space 作为源空间
        target: 目标空间
        pad_value: 填充值
        mode: 插值模式，默认 "trilinear"
        pad_mode: 填充模式，默认 "constant"
        half: 是否使用半精度，默认 False
        numpy: 是否返回 numpy 格式的图像数据，默认 False
        cuda_device: CUDA 设备，默认 "cuda:0"
        
    Returns:
        DicomCubeImage: 重采样后的新 DicomCubeImage，空间为目标空间
        
    Raises:
        ValueError: 如果输入图像没有空间信息
    """
    if img.space is None:
        raise ValueError("DicomCubeImage 必须包含 space 信息才能进行重采样")
    
    # 使用现有的 warp_image 函数处理图像数据
    warped_data = warp_image(
        img.raw_image,
        source=img.space,
        target=target,
        pad_value=pad_value,
        mode=mode,
        pad_mode=pad_mode,
        half=half,
        numpy=numpy,
        cuda_device=cuda_device,
    )
    
    # 创建新的 DicomCubeImage，使用重采样后的数据和目标空间
    return DicomCubeImage(
        raw_image=warped_data,
        pixel_header=img.pixel_header,  # 保持像素头信息
        dicom_meta=img.dicom_meta,      # 保持元数据信息（注意：空间相关的元数据可能需要更新）
        space=target,                   # 使用目标空间
    ) 