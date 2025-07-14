import numpy as np
import pytest

import torch

from spacetransformer.core import Space
from spacetransformer.torch import warp_image

try:
    import SimpleITK as sitk  # type: ignore
    _has_sitk = True
except ImportError:  # pragma: no cover
    _has_sitk = False

pytestmark = pytest.mark.skipif(not _has_sitk, reason="SimpleITK not available")


def _space_to_sitk(space: Space, arr: np.ndarray):
    # 我们的数组顺序 (D,H,W)=(x,y,z)，而 SimpleITK 期望 (z,y,x)
    arr_sitk = arr.transpose(2, 1, 0)  # (D,H,W) -> (W,H,D) -> (z,y,x)
    img = sitk.GetImageFromArray(arr_sitk.astype(np.float32))

    # 方向矩阵：Space 已提供 column-major → row-major
    img.SetDirection(tuple(float(x) for x in space.to_sitk_direction()))
    img.SetOrigin(tuple(float(x) for x in space.origin))
    img.SetSpacing(tuple(float(x) for x in space.spacing))
    return img


def test_random_rotate_scale_align():
    # 源空间
    src = Space(shape=(32, 40, 24))
    img_np = np.random.rand(*src.shape).astype(np.float32)

    # 构造一个非正交的目标空间
    # 先绕 Z 轴 20°，再绕 Y 轴 10
    rz = np.deg2rad(20)
    ry = np.deg2rad(10)
    cz, sz = np.cos(rz), np.sin(rz)
    cy, sy = np.cos(ry), np.sin(ry)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    R = Rz @ Ry
    
    x_o, y_o, z_o = R[:, 0], R[:, 1], R[:, 2]
    tgt = Space(
        shape=src.shape,
        origin=src.origin,
        spacing=src.spacing,
        x_orientation=tuple(x_o),
        y_orientation=tuple(y_o),
        z_orientation=tuple(z_o),
    )

    # warp_image 结果
    out_st = warp_image(img_np, src, tgt, pad_value=0.0, cuda_device="cuda:0",mode='nearest')

    # SimpleITK 参考
    src_img = _space_to_sitk(src, img_np)

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(tgt.shape)
    resampler.SetOutputSpacing(tuple(float(x) for x in tgt.spacing))
    resampler.SetOutputOrigin(tuple(float(x) for x in tgt.origin))
    resampler.SetOutputDirection(tuple(float(x) for x in tgt.to_sitk_direction()))
    resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    res_img = resampler.Execute(src_img)
    out_sitk = sitk.GetArrayFromImage(res_img).transpose(2, 1, 0)
    assert np.allclose(out_st, out_sitk)


