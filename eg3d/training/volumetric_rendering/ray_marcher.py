# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""
The ray marcher takes the raw output of the implicit representation and uses the volume rendering equation to produce composited colors and depths.
Based off of the implementation in MipNeRF (this one doesn't do any cone tracing though!)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_efficient_distloss import eff_distloss, eff_distloss_native
import pdb

class MipRayMarcher2(nn.Module):
    def __init__(self):
        super().__init__()


    def run_forward(self, colors, densities, depths, rendering_options):
        deltas = depths[:, :, 1:] - depths[:, :, :-1]  # [N, 16384, 95, 1]
        colors_mid = (colors[:, :, :-1] + colors[:, :, 1:]) / 2  # [N, 16384, 95, 1]
        densities_mid = (densities[:, :, :-1] + densities[:, :, 1:]) / 2  # [N, 16384, 95, 1]
        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2  # [N, 16384, 95, 1]


        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        density_delta = densities_mid * deltas  # [N, 16384, 95, 1]

        alpha = 1 - torch.exp(-density_delta)  # [N, 16384, 95, 1]

        alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)  # [N, 16384, 96, 1]
        weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]  # [N, 16384, 95, 1]

        composite_rgb = torch.sum(weights * colors_mid, -2)
        weight_total = weights.sum(2)
        composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)

        return composite_rgb, composite_depth, weights


    def forward(self, colors, densities, depths, rendering_options):
        composite_rgb, composite_depth, weights = self.run_forward(colors, densities, depths, rendering_options)

        return composite_rgb, composite_depth, weights

    def blend_forward(self, depths, colors_o, densities_o, colors_f, densities_f, blending_weight, rendering_options, comp_dist_loss=False):
        """
        Args:
        colors_o: raw rgbs from decoder_o
        densities_o: raw densities from decoder_o
        colors_f: raw rgbs from decoder_f
        densities_f: raw densities from decoder_f
        depths: sampled depths
        blending_weight: blending weight from decoder_o
        """
        # deltas = depths[:, :, 1:] - depths[:, :, :-1]  # [N, 16384, 95, 1]
        # colors_mid = (colors_f[:, :, :-1] + colors_f[:, :, 1:]) / 2  # [N, 16384, 95, 1]
        # densities_mid = (densities_f[:, :, :-1] + densities_f[:, :, 1:]) / 2  # [N, 16384, 95, 1]
        # depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2  # [N, 16384, 95, 1]


        # if rendering_options['clamp_mode'] == 'softplus':
        #     densities_mid = F.softplus(densities_mid - 1) # activation bias of -1 makes things initialize better
        # else:
        #     assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"

        # density_delta = densities_mid * deltas  # [N, 16384, 95, 1]

        # alpha = 1 - torch.exp(-density_delta)  # [N, 16384, 95, 1]
        # pdb.set_trace()
        # alpha_shifted = torch.cat([torch.ones_like(alpha[:, :, :1]), 1-alpha + 1e-10], -2)  # [N, 16384, 96, 1]
        # weights = alpha * torch.cumprod(alpha_shifted, -2)[:, :, :-1]  # [N, 16384, 95, 1]

        # composite_rgb = torch.sum(weights * colors_mid, -2)
        # weight_total = weights.sum(2)
        # composite_depth = torch.sum(weights * depths_mid, -2) / weight_total

        # # clip the composite to min/max range of depths
        # composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        # composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        # if rendering_options.get('white_back', False):
        #     composite_rgb = composite_rgb + 1 - weight_total

        # composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)
        # weights_full = weights

        ############# TODO: DEBUG #################
        deltas = depths[:, :, 1:] - depths[:, :, :-1]
        # deltas = torch.cat([deltas, torch.Tensor([1e10]).expand(1, deltas.shape[1], 1, 1).to(deltas.device)], dim=2)

        colors_mid_o = (colors_o[:, :, :-1] + colors_o[:, :, 1:]) / 2
        densities_mid_o = (densities_o[:, :, :-1] + densities_o[:, :, 1:]) / 2
        colors_mid_f = (colors_f[:, :, :-1] + colors_f[:, :, 1:]) / 2
        densities_mid_f = (densities_f[:, :, :-1] + densities_f[:, :, 1:]) / 2

        blending_weight_mid = (blending_weight[:, :, :-1] + blending_weight[:, :, 1:]) / 2

        depths_mid = (depths[:, :, :-1] + depths[:, :, 1:]) / 2

        if rendering_options['clamp_mode'] == 'softplus':
            densities_mid_o = F.softplus(densities_mid_o - 1) # activation bias of -1 makes things initialize better
            densities_mid_f = F.softplus(densities_mid_f - 1)
        else:
            assert False, "MipRayMarcher only supports `clamp_mode`=`softplus`!"
        
        density_delta_o = densities_mid_o * deltas
        density_delta_f = densities_mid_f * deltas

        alpha_o = 1 - torch.exp(-density_delta_o)
        alpha_f = 1 - torch.exp(-density_delta_f)

        # Dynamic NeRF: https://github.com/gaochen315/DynamicNeRF/blob/c417fb207ef352f7e97521a786c66680218a13af/render_utils.py#L378
        # T_full = torch.cumprod(torch.cat([torch.ones((alpha_d.shape[0], 1)), (1. - alpha_d * blending) * (1. - alpha_s * (1. - blending)) + 1e-10], -1), -1)[:, :-1]
        alpha_shifted_full = torch.cat([torch.ones_like(alpha_f[:, :, :1]), (1-alpha_o*blending_weight_mid) * (1 - alpha_f*(1-blending_weight_mid)) + 1e-10], -2)
        alpha_shifted_full = torch.cumprod(alpha_shifted_full, -2)[:, :, :-1]
        weights_full = (alpha_o * blending_weight_mid + alpha_f * (1 - blending_weight_mid)) * alpha_shifted_full
        # weights_full_o = alpha_o * blending_weight_mid * alpha_shifted_full
        # weights_full = alpha * torch.cumprod(alpha_shifted_full, -2)[:, :, :-1]
        # pdb.set_trace()
        composite_rgb = torch.sum(alpha_shifted_full * (alpha_o * colors_mid_o * blending_weight_mid + alpha_f * colors_mid_f * (1 - blending_weight_mid)), -2)
        # composite_rgb = torch.sum((alpha_shifted_full*alpha_o*blending_weight_mid)*colors_mid_o+(alpha_shifted_full*alpha_f*(1-blending_weight_mid))*colors_mid_f, -2)
        # composite_rgb = torch.sum(weights_full * (colors_mid_o + colors_mid_f) / 2, -2)
        weight_total = weights_full.sum(2)

        composite_depth = torch.sum(weights_full * depths_mid, -2) / weight_total

        # clip the composite to min/max range of depths
        composite_depth = torch.nan_to_num(composite_depth, float('inf'))
        composite_depth = torch.clamp(composite_depth, torch.min(depths), torch.max(depths))

        if rendering_options.get('white_back', False):
            composite_rgb = composite_rgb + 1 - weight_total

        composite_rgb = composite_rgb * 2 - 1 # Scale to (-1, 1)
        
        ############# TODO: DEBUG #################
        if not comp_dist_loss:
            return composite_rgb, composite_depth, weights_full
        else:
            interval = 1/weights_full.shape[2]
            w = weights_full.squeeze(-1).reshape(-1, weights_full.shape[2])
            w = w / w.sum(-1, keepdim=True)
            dist_loss = eff_distloss(w, depths_mid.squeeze(-1).reshape(-1, depths_mid.shape[2]), interval)
            return composite_rgb, composite_depth, weights_full, dist_loss