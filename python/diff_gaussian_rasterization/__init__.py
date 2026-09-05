#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_gaussians(
    means3D,
    means2D,
    sh,
    colors_precomp,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
):
    return _RasterizeGaussians.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    )

class _RasterizeGaussians(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
    ):

        # 保持原 CUDA 入口的参数顺序，与 bindings/torch/api.h 的 forward 签名一致。
        # ``_C.rasterize_gaussians(*args)`` 会将这个元组逐项展开后传给该扩展。
        args = (
            raster_settings.bg,              # background：输出图像的背景颜色
            means3D,                         # means3D：每个 Gaussian 在世界坐标系中的 3D 中心，形状通常为 [P, 3]
            colors_precomp,                  # colors：预先计算的 RGB；若为空，则由 sh 在 Metal 中计算
            opacities,                       # opacity：每个 Gaussian 的不透明度
            scales,                          # scales：每个 Gaussian 沿三个轴的缩放/标准差
            rotations,                       # rotations：每个 Gaussian 的旋转（通常为四元数）
            raster_settings.scale_modifier,  # scale_modifier：对 scales 额外施加的全局缩放系数
            cov3Ds_precomp,                  # cov3D_precomp：预计算的 3D 协方差；提供它时可替代 scales 和 rotations
            raster_settings.viewmatrix,      # viewmatrix：世界坐标转换到相机坐标的视图矩阵
            raster_settings.projmatrix,      # projmatrix：包含 view 的世界坐标到裁剪空间矩阵
            raster_settings.tanfovx,         # tan_fovx：水平视场角的一半的正切值
            raster_settings.tanfovy,         # tan_fovy：垂直视场角的一半的正切值
            raster_settings.image_height,    # image_height：待渲染图像的高度（像素）
            raster_settings.image_width,     # image_width：待渲染图像的宽度（像素）
            sh,                              # sh：球谐函数（Spherical Harmonics）颜色系数
            raster_settings.sh_degree,       # degree：使用的球谐函数阶数
            raster_settings.campos,          # campos：相机在世界坐标系中的位置
            raster_settings.prefiltered,     # prefiltered：是否表示输入已经做过预滤波处理
            raster_settings.debug            # debug：是否开启 C++/Metal 调试相关行为
        )

        # Invoke C++/Metal rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                # C++/Metal 扩展返回：
                # - num_rendered：Gaussian 投影后分配到各个屏幕 tile 的实例总数；
                #   一个较大的 Gaussian 可能覆盖多个 tile，因此该数可能大于 Gaussian 数量
                # - color：渲染得到的彩色图像，形状通常为 [3, H, W]
                # - radii：每个 Gaussian 投影到屏幕后的像素半径；不可见的点通常为 0
                # - geomBuffer：几何预处理阶段的中间数据，供 Metal backward 使用
                # - binningBuffer：按屏幕 tile 分桶/排序的中间数据，供 backward 使用
                # - imgBuffer：图像合成阶段的中间数据，供 backward 使用
                num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            # 返回值含义与上方相同；这三个 buffer 会在后面由 ctx.save_for_backward
            # 保存，以便调用 loss.backward() 时传入 C++/Metal 的反向光栅化函数。
            num_rendered, color, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_gaussians(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer)
        return color, radii

    @staticmethod
    def backward(ctx, grad_out_color, _):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp, 
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations = _C.rasterize_gaussians_backward(*args)

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
        )

        return grads

class GaussianRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

class GaussianRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def markVisible(self, positions):
        # Mark visible points (based on frustum culling for camera) with a boolean 
        with torch.no_grad():
            raster_settings = self.raster_settings
            visible = _C.mark_visible(
                positions,
                raster_settings.viewmatrix,
                raster_settings.projmatrix)
            
        return visible

    def forward(self, means3D, means2D, opacities, shs = None, colors_precomp = None, scales = None, rotations = None, cov3D_precomp = None):
        
        raster_settings = self.raster_settings

        if (shs is None and colors_precomp is None) or (shs is not None and colors_precomp is not None):
            raise Exception('Please provide excatly one of either SHs or precomputed colors!')
        
        if ((scales is None or rotations is None) and cov3D_precomp is None) or ((scales is not None or rotations is not None) and cov3D_precomp is not None):
            raise Exception('Please provide exactly one of either scale/rotation pair or precomputed 3D covariance!')
        
        if shs is None:
            shs = torch.Tensor([])
        if colors_precomp is None:
            colors_precomp = torch.Tensor([])

        if scales is None:
            scales = torch.Tensor([])
        if rotations is None:
            rotations = torch.Tensor([])
        if cov3D_precomp is None:
            cov3D_precomp = torch.Tensor([])

        # Invoke C++/Metal rasterization routine
        return rasterize_gaussians(
            means3D,
            means2D,
            shs,
            colors_precomp,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings, 
        )
