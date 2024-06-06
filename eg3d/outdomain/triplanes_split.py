"""
Triplanes for new (out-of-distribution) item.
"""

import torch
from torch.nn import functional as f
import numpy as np
import kornia

from training.triplane import OSGDecoder
from training.volumetric_rendering.ray_sampler import RaySampler
from training.volumetric_rendering.renderer import ImportanceRenderer
from training.volumetric_rendering.ray_marcher import MipRayMarcher2
from training.networks_stylegan2 import Generator as StyleGAN2Backbone
from torch_utils import misc
from torch_utils.ops import upfirdn2d

import pdb

def smooth_superresolution(model, rgb, x, ws, smooth_kernel, smooth_toRGB, acc_rgb, acc_toRGB_input, **block_kwargs):
    ws = ws[:, -1:, :].repeat(1, 3, 1)

    if x.shape[-1] != model.input_resolution:
        x = torch.nn.functional.interpolate(x, size=(model.input_resolution, model.input_resolution),
                                            mode='bilinear', align_corners=False, antialias=model.sr_antialias)
        rgb = torch.nn.functional.interpolate(rgb, size=(model.input_resolution, model.input_resolution),
                                            mode='bilinear', align_corners=False, antialias=model.sr_antialias)

    # x, rgb = model.block0(x, rgb, ws, **block_kwargs)
    x, rgb = smooth_block_forward(model.block0, x, rgb, ws, smooth_kernel, acc_rgb, acc_toRGB_input, smooth_toRGB, **block_kwargs)
    
    
    x, rgb = model.block1(x, rgb, ws, **block_kwargs)
    # x, rgb = smooth_block_forward(model.block1, x, rgb, ws, smooth_kernel, acc_rgb, acc_toRGB_input, smooth_toRGB, **block_kwargs)

    return rgb

def smooth_block_forward(block, x, img, ws, smooth_kernel, acc_rgb, acc_toRGB_input, smooth_toRGB=False, force_fp32=False, fused_modconv=None, update_emas=False, **layer_kwargs):
    _ = update_emas # unused
    misc.assert_shape(ws, [None, block.num_conv + block.num_torgb, block.w_dim])
    w_iter = iter(ws.unbind(dim=1))
    if ws.device.type != 'cuda':
        force_fp32 = True
    dtype = torch.float16 if block.use_fp16 and not force_fp32 else torch.float32
    memory_format = torch.channels_last if block.channels_last and not force_fp32 else torch.contiguous_format
    if fused_modconv is None:
        fused_modconv = block.fused_modconv_default
    if fused_modconv == 'inference_only':
        fused_modconv = (not block.training)

    # Input.
    if block.in_channels == 0:
        x = block.const.to(dtype=dtype, memory_format=memory_format)
        x = x.unsqueeze(0).repeat([ws.shape[0], 1, 1, 1])
    else:
        misc.assert_shape(x, [None, block.in_channels, block.resolution // 2, block.resolution // 2])
        x = x.to(dtype=dtype, memory_format=memory_format)

    # Main layers.
    if block.in_channels == 0:
        x = block.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
    elif block.architecture == 'resnet':
        y = block.skip(x, gain=np.sqrt(0.5))
        x = block.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        x = block.conv1(x, next(w_iter), fused_modconv=fused_modconv, gain=np.sqrt(0.5), **layer_kwargs)
        x = y.add_(x)
    else:
        x = block.conv0(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)
        x = block.conv1(x, next(w_iter), fused_modconv=fused_modconv, **layer_kwargs)

    if smooth_toRGB:
        acc_toRGB_input.append(x.detach().clone())
        if len(acc_toRGB_input) >= smooth_kernel.shape[0]:
            x = torch.sum(smooth_kernel.view(smooth_kernel.shape[0], 1, 1, 1).to(dtype) * torch.cat(acc_toRGB_input, dim=0), dim=0, keepdim=True)
            acc_toRGB_input.pop(0)
        
        


    # ToRGB.
    if img is not None:
        misc.assert_shape(img, [None, block.img_channels, block.resolution // 2, block.resolution // 2])
        img = upfirdn2d.upsample2d(img, block.resample_filter)
        if smooth_toRGB:
            acc_rgb.append(img)
            if len(acc_rgb) >= smooth_kernel.shape[0]:
                img = torch.sum(smooth_kernel.view(smooth_kernel.shape[0], 1, 1, 1).to(dtype) * torch.cat(acc_rgb, dim=0), dim=0, keepdim=True)
                acc_rgb.pop(0)
    if block.is_last or block.architecture == 'skip':
        y = block.torgb(x, next(w_iter), fused_modconv=fused_modconv)
        y = y.to(dtype=torch.float32, memory_format=torch.contiguous_format)
        img = img.add_(y) if img is not None else y

    assert x.dtype == dtype
    assert img is None or img.dtype == torch.float32
    return x, img

def get_color_density(planes, renderer, decoder, ray_origins, ray_directions, rendering_options):
    renderer.plane_axes = renderer.plane_axes.to(ray_origins.device)
    if rendering_options['ray_start'] == rendering_options['ray_end'] == 'auto':
        ray_start, ray_end = math_utils.get_ray_limits_box(ray_origins, ray_directions, box_side_length=rendering_options['box_warp'])
        is_ray_valid = ray_end > ray_start
        if torch.any(is_ray_valid).item():
            ray_start[~is_ray_valid] = ray_start[is_ray_valid].min()
            ray_end[~is_ray_valid] = ray_start[is_ray_valid].max()
        depths_coarse = renderer.sample_stratified(ray_origins, ray_start, ray_end, rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])
    else:
        # Create stratified depth samples
        depths_coarse = renderer.sample_stratified(ray_origins, rendering_options['ray_start'], rendering_options['ray_end'], rendering_options['depth_resolution'], rendering_options['disparity_space_sampling'])

    batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape

    # Coarse Pass
    sample_coordinates_coarse = (ray_origins.unsqueeze(-2) + depths_coarse * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)
    sample_directions_coarse = ray_directions.unsqueeze(-2).expand(-1, -1, samples_per_ray, -1).reshape(batch_size, -1, 3)


    out = renderer.run_model(planes, decoder, sample_coordinates_coarse, sample_directions_coarse, rendering_options)
    colors_coarse = out['rgb']
    densities_coarse = out['sigma']
    colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
    densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)

    # Fine Pass
    N_importance = rendering_options['depth_resolution_importance']
    if N_importance > 0:
        _, _, weights = renderer.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)

        depths_fine = renderer.sample_importance(depths_coarse, weights, N_importance)

        sample_directions_fine = ray_directions.unsqueeze(-2).expand(-1, -1, N_importance, -1).reshape(batch_size, -1, 3)
        sample_coordinates_fine = (ray_origins.unsqueeze(-2) + depths_fine * ray_directions.unsqueeze(-2)).reshape(batch_size, -1, 3)

        out = renderer.run_model(planes, decoder, sample_coordinates_fine, sample_directions_fine, rendering_options)
        colors_fine = out['rgb']
        densities_fine = out['sigma']
        colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
        densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)

        all_depths, all_colors, all_densities, indices = renderer.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                depths_fine, colors_fine, densities_fine, return_indices=True)
        return all_depths, all_colors, all_densities, sample_coordinates_coarse, sample_directions_coarse, depths_coarse, sample_coordinates_fine, sample_directions_fine, depths_fine, indices

class ComposeTriplane(torch.nn.Module):
    def __init__(self, G, comp_dist_loss=False):
        super().__init__()
        self.G = G   # 3D GAN
        self.new_triplanes = TriPlaneItem()
        self.comp_dist_loss = comp_dist_loss

    def _rendering(self, ws, ray_sampler, renderer, decoder, cam2world_matrix, intrinsics, neural_rendering_resolution, update_emas=False, use_cached_backbone=False, cache_planes=None):
        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)  # [N, 128*128, 3], [N, 128*128, 3]

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape
        if use_cached_backbone and cache_planes is not None:
            planes = cache_planes
        else:
            planes = self.G.backbone.synthesis(ws, update_emas=update_emas, noise_mode='const')  # [N, 96, 256, 256]

        # Reshape output into three 32-channel planes
        if len(planes.shape) == 4:
            planes = planes.view(len(planes), 3, 32, planes.shape[-2], planes.shape[-1])  # [N, 3, 32, 256, 256]

        # sorted depths, colors, densities
        all_depths, all_colors, all_densities, sample_coordinates_coarse, sample_directions_coarse, depths_coarse, sample_coordinates_fine, sample_directions_fine, depths_fine, indices  = get_color_density(planes, renderer, decoder, ray_origins, ray_directions, self.G.rendering_kwargs)

        return all_depths, all_colors, all_densities, planes, sample_coordinates_coarse, sample_coordinates_fine, sample_directions_coarse, sample_directions_fine, depths_coarse, depths_fine, indices



    def forward(self, ws, c, phi, masks=None, smooth_kernel=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, return_raw_blendw=False, remove_ood=False, smooth_toRGB=False):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)
        
        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.G.neural_rendering_resolution
        else:
            self.G.neural_rendering_resolution = neural_rendering_resolution

        # Forward Pass 1: get densities and colors from original G
        depths, old_colors, old_densities, old_planes, pts_coarse, pts_fine, directions_coarse, directions_fine, depths_coarse, depths_fine, indices = self._rendering(
                        ws, 
                        self.G.ray_sampler, 
                        self.G.renderer, 
                        self.G.decoder, 
                        cam2world_matrix, 
                        intrinsics, 
                        neural_rendering_resolution, 
                        use_cached_backbone=False
        )
        pts = torch.cat([pts_coarse, pts_fine], dim=1)
        ray_directions = torch.cat([directions_coarse, directions_fine], dim=1)

        # Forward Pass 2: get densities and colors from new triplanes,
        ##################### TODO: DEBUG ########################
        # raw colors, densities, and blending weights for od

        out_ood = self.new_triplanes(
            phi,
            pts, 
            depths_coarse, 
            depths_fine, 
            ray_directions, 
            neural_rendering_resolution, 
            rendering_kwargs=self.G.rendering_kwargs,
            return_raw_blendw=return_raw_blendw
            )
        
        # collect output 
        feature_samples_o, depths_samples_o = out_ood['rgb_final'], out_ood['depth_final']  # rendered output
        new_colors, new_densities, blendings = out_ood['colors'], out_ood['densities'], out_ood['blendings']  # raw output
        blendw_coarse, blendw_fine = out_ood['blendw_coarse'], out_ood['blendw_fine']

        if remove_ood:
            ood_masks = 1 - masks
            ood_masks = torch.nn.functional.interpolate(ood_masks, size=self.G.neural_rendering_resolution, mode='nearest')
            
            # ood_masks_dilated = kornia.morphology.dilation(ood_masks, torch.ones(15, 15)).clamp(0.0, 1.0)
            ood_masks_dilated = kornia.morphology.dilation(ood_masks, torch.ones(5, 5)).clamp(0.0, 1.0)
            # ood_masks_dilated = kornia.morphology.dilation(ood_masks, torch.ones(3, 3)).clamp(0.0, 1.0)

            ood_masks_dilated = ood_masks_dilated[:, 0, :].reshape(-1, self.G.neural_rendering_resolution*self.G.neural_rendering_resolution)
            
            ood_pixel_ind = ood_masks_dilated == 1
            blendings[:] *= 1.2
            blendings[ood_pixel_ind] = 0

        # Forward Pass 3: combine color and density
        if not self.comp_dist_loss:
            dist_loss = 0.0
            feature_samples_full, depth_samples_full, weights_samples_full = self.G.renderer.ray_marcher.blend_forward(depths, new_colors, new_densities, old_colors, old_densities, blendings, self.G.rendering_kwargs, self.comp_dist_loss)
        else:
            feature_samples_full, depth_samples_full, weights_samples_full, dist_loss = self.G.renderer.ray_marcher.blend_forward(depths, new_colors, new_densities, old_colors, old_densities, blendings, self.G.rendering_kwargs, self.comp_dist_loss)
        ##################### TODO: DEBUG ########################
                
        # Reshape into 'raw' neural-rendered image
        H = W = self.G.neural_rendering_resolution
        N = pts.shape[0]
        feature_image_full = feature_samples_full.permute(0, 2, 1).view(N, feature_samples_full.shape[-1], H, W).contiguous()  # [B,32,128,128]
        depth_image_full = depth_samples_full.permute(0, 2, 1).view(N, 1, H, W)  # [B,1,128,128]
        feature_image_o = feature_samples_o.permute(0, 2, 1).view(N, feature_samples_o.shape[-1], H, W).contiguous()
        depth_image_o = depths_samples_o.permute(0, 2, 1).view(N, 1, H, W)  # [B,1,128,128]
        
        # Run superresolution to get final image
        rgb_image_full = feature_image_full[:, :3].clamp(-1.0, 1.0)  # raw image
        if not smooth_toRGB:
            sr_image_full = self.G.superresolution(rgb_image_full, feature_image_full, ws, noise_mode='const', ).clamp(-1.0, 1.0)
        else:
            sr_image_full = smooth_superresolution(self.G.superresolution, rgb_image_full, feature_image_full, ws, smooth_kernel, smooth_toRGB=smooth_toRGB, acc_rgb=self.acc_rgb, acc_toRGB_input=self.acc_toRGB_input, noise_mode='const', ).clamp(-1.0, 1.0)
        rgb_image_o = feature_image_o[:, :3].clamp(-1.0, 1.0)
        sr_image_o = self.G.superresolution(rgb_image_o, feature_image_o, ws, noise_mode='const', ).clamp(-1.0, 1.0)

        return {'image_full': sr_image_full, 'image_raw_full': rgb_image_full, 'image_depth_full': depth_image_full, 
                'image_o': sr_image_o, 'image_raw_o': rgb_image_o, 'image_depth_o': depth_image_o,
                'blendw_coarse': blendw_coarse, 'blendw_fine': blendw_fine, 'densities_o': new_densities, 'densities_f': old_densities,
                'dist_loss': dist_loss,
                }

class TriPlaneItem(torch.nn.Module):
    def __init__(self, c_dim=32, resolution=256, init_way='normal'):
        super().__init__()
        self.c_dim = c_dim
        self.resolution = resolution  # triplane resolution
        self.decoder =  OSGDecoder(c_dim*2, {'decoder_lr_mul': 1.0, 'decoder_output_dim': 1 + c_dim})  # color and density decoder
        # self.ray_sampler = RaySampler()
        self.renderer = ImportanceRenderer()
        self.triplanes = self.init_triplane(init_way) # [N, 3, 32, 256, 256]

    def init_triplane(self, init_way="normal"):
        if init_way == "normal":
            return torch.nn.Parameter(torch.randn(1, 3, self.c_dim, self.resolution, self.resolution))
        elif init_way == "zeros":
            return torch.nn.Parameter(torch.zeros(1, 3, self.c_dim, self.resolution, self.resolution))

    def rendering(self, all_depths, all_colors, all_densities, rendering_kwargs):
        rgb_final, depth_final, weights = self.renderer.ray_marcher(all_colors, all_densities, all_depths, rendering_kwargs)
        return rgb_final, depth_final, weights.sum(2)

    def get_color_density(self, phi, planes, pts, depths_coarse, depths_fine, ray_directions, neural_rendering_resolution, rendering_options, return_raw_blendw=False):

        batch_size, num_rays, samples_per_ray, _ = depths_coarse.shape  # (N, H*W, 2*N_importance, 1)
        N_importance = rendering_options['depth_resolution_importance']
        
        self.renderer.plane_axes = self.renderer.plane_axes.to(planes.device)
        
        # # coarse pass
        pts_coarse = pts[:, :pts.shape[1]//2, :]
        pts_fine = pts[:, pts.shape[1]//2:, :]
        directions_coarse = ray_directions[:, :ray_directions.shape[1]//2, :]
        directions_fine = ray_directions[:, ray_directions.shape[1]//2:, :]

        out = self.renderer.run_model_extend(phi, planes, self.decoder, pts_coarse, directions_coarse, rendering_options)
        colors_coarse = out['rgb']
        blendings_coarse = out['blending']
        densities_coarse = out['sigma']

        colors_coarse = colors_coarse.reshape(batch_size, num_rays, samples_per_ray, colors_coarse.shape[-1])
        densities_coarse = densities_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        blendings_coarse = blendings_coarse.reshape(batch_size, num_rays, samples_per_ray, 1)
        if N_importance > 0:
            _, _, weights = self.renderer.ray_marcher(colors_coarse, densities_coarse, depths_coarse, rendering_options)
            depths_fine = self.renderer.sample_importance(depths_coarse, weights, N_importance)

            sample_directions = directions_coarse
            sample_coordinates = pts_fine

            out = self.renderer.run_model_extend(phi, planes, self.decoder, sample_coordinates, sample_directions, rendering_options)
            colors_fine = out['rgb']
            blendings_fine = out['blending']
            densities_fine = out['sigma']

            colors_fine = colors_fine.reshape(batch_size, num_rays, N_importance, colors_fine.shape[-1])
            densities_fine = densities_fine.reshape(batch_size, num_rays, N_importance, 1)
            blendings_fine = blendings_fine.reshape(batch_size, num_rays, N_importance, 1)
            all_depths, all_colors, all_densities, indices = self.renderer.unify_samples(depths_coarse, colors_coarse, densities_coarse,
                                                                    depths_fine, colors_fine, densities_fine, return_indices=True)

            all_blendings = torch.cat([blendings_coarse, blendings_fine], dim=-2) 
            all_blendings = torch.gather(all_blendings, -2, indices.expand(-1, -1, -1, 1))
            
        if not return_raw_blendw:
            return all_depths, all_colors, all_densities, all_blendings
        else:
            return all_depths, all_colors, all_densities, all_blendings, blendings_coarse, blendings_fine


    def forward(self, phi, pts, depths_coarse, depths_fine, ray_directions, neural_rendering_resolution, rendering_kwargs, return_raw_blendw=False):
        """
        Get rendered RGB images and output sorted colors and densities.
        Args:
        Output: 
        RGB map
        all_colors
        all_densities
        all_depths
        """
        
        # get triplanes
        planes = self.triplanes

        # get raw colors, densities, and blendings. Sorted by depths.
        if not return_raw_blendw:
            all_depths, all_colors, all_densities, all_blendings = self.get_color_density(phi, planes, pts, depths_coarse, depths_fine, ray_directions, neural_rendering_resolution, rendering_kwargs, return_raw_blendw)
            blendings_coarse = None
            blendings_fine = None
        else:
            all_depths, all_colors, all_densities, all_blendings, blendings_coarse, blendings_fine = self.get_color_density(phi, planes, pts, depths_coarse, depths_fine, ray_directions, neural_rendering_resolution, rendering_kwargs, return_raw_blendw)

        # rendering: raw color/density -> output RGB image
        feature_samples, depth_samples, weights_samples = self.rendering(all_depths, all_colors, all_densities, rendering_kwargs)
        
        return {'depths': all_depths, 'colors': all_colors, 'densities': all_densities, 'blendings': all_blendings,
                'rgb_final': feature_samples, 'depth_final': depth_samples, 'weights': weights_samples,
                'blendw_coarse': blendings_coarse, 'blendw_fine': blendings_fine,
            }


class TriplaneRadianceField(torch.nn.Module):
    def __init__(self, 
                rendering_kwargs,
                num_triplanes=1,
                num_channels=32,
                plane_resolution=256,
                neural_rendering_resolution=128,
                init_way='normal'):
        super().__init__()
        self.rendering_kwargs = rendering_kwargs
        self.num_triplanes = num_triplanes
        self.num_channels = num_channels
        self.plane_resolution = plane_resolution
        self.init_way = init_way
        self.triplane = self.init_triplane(num_triplanes, num_channels, plane_resolution, init_way)
        self.decoder = OSGDecoder(32, {'decoder_lr_mul': rendering_kwargs.get('decoder_lr_mul', 1), 'decoder_output_dim': 32})
        self.ray_sampler = RaySampler()
        self.renderer = ImportanceRenderer()
        self.neural_rendering_resolution = neural_rendering_resolution
    
    def init_triplane(self, batch_size=1, num_channels=32, resolution=256, init_way="normal"):
        if init_way == "normal":
            return torch.nn.Parameter(torch.randn(batch_size, 3, num_channels, resolution, resolution))
        elif init_way == "zeros":
            return torch.nn.Parameter(torch.zeros(batch_size, 3, num_channels, resolution, resolution))

    def forward(self, c, neural_rendering_resolution=None):
        """
        Get raw rgb and density, and
        render the output image.
        Input Args:
        c: camera parameteres [N, 25].
        neural_rendering_resolution: output resolution.
        """
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution

        # Create a batch of rays for volume rendering
        ray_origins, ray_directions = self.ray_sampler(cam2world_matrix, intrinsics, neural_rendering_resolution)  # [N, 128*128, 3], [N, 128*128, 3]

        # Create triplanes by running StyleGAN backbone
        N, M, _ = ray_origins.shape

        # Perform volume rendering, color/feature [B,128*128,32], density/opacity [B,128*128,1], weights [B,128*128,1]
        feature_samples, depth_samples, weights_samples = self.renderer(self.triplane, self.decoder, ray_origins, ray_directions, self.rendering_kwargs) # channels last

        # Reshape into 'raw' neural-rendered image
        H = W = neural_rendering_resolution
        feature_image = feature_samples.permute(0, 2, 1).reshape(N, feature_samples.shape[-1], H, W).contiguous()  # [B,32,128,128]
        depth_image = depth_samples.permute(0, 2, 1).reshape(N, 1, H, W)  # [B,1,128,128]

        # Run superresolution to get final image
        rgb_image = feature_image[:, :3]  # raw image

        return {'image_raw': rgb_image, 'image_depth': depth_image}
