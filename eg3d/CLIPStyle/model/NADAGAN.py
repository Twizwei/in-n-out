import sys
import os
sys.path.insert(0, os.path.abspath('../'))
import pdb

import torch
import torchvision.transforms as transforms

import numpy as np
import copy
import pickle

from functools import partial

from camera_utils import LookAtPoseSampler
from CLIPStyle.criteria.clip_loss import CLIPLoss       
import CLIPStyle.legacy as legacy

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

# class SG3Generator(torch.nn.Module):
#     def __init__(self, checkpoint_path):
#         super(SG3Generator, self).__init__()

#         with open(checkpoint_path, 'rb') as f:
#             self.generator = pickle.load(f)['G_ema'].cuda()

#     def get_all_layers(self):
#         return list(self.generator.synthesis.children())

#     def trainable_params(self):
#         params = []
#         for layer in self.get_training_layers():
#             params.extend(layer.parameters())

#         return params

#     def get_training_layers(self, phase=None):
#         return self.get_all_layers()[:11] + self.get_all_layers()[12:13] + self.get_all_layers()[14:]

#     def freeze_layers(self, layer_list=None):
#         '''
#         Disable training for all layers in list.
#         '''
#         if layer_list is None:
#             self.freeze_layers(self.generator.children())
#         else:
#             for layer in layer_list:
#                 requires_grad(layer, False)

#     def unfreeze_layers(self, layer_list=None):
#         '''
#         Enable training for all layers in list.
#         '''

#         if layer_list is None:
#             self.unfreeze_layers(self.get_all_layers())
#         else:
#             for layer in layer_list:
#                 requires_grad(layer, True)
                
#                 if hasattr(layer, "affine"):
#                     requires_grad(layer.affine, False)
#                 else:
#                     for child_layer in layer.children():
#                         requires_grad(child_layer.affine, False)
#                 if hasattr(layer, "torgb"):
#                     requires_grad(layer.torgb, False)

#     def style(self, z_codes, truncation=0.7):
#         return self.generator.mapping(z_codes[0], None, truncation_psi=truncation, truncation_cutoff=None)

#     def forward(self, styles, input_is_latent=None, truncation=None, randomize_noise=None): # unused args for compatibility with SG2 interface
#         return self.generator.synthesis(styles, noise_mode='random', force_fp32=True), None

# class SGXLGenerator(torch.nn.Module):
#     def __init__(self, checkpoint_path):
#         super(SGXLGenerator, self).__init__()

#         with open(checkpoint_path, 'rb') as f:
#             self.generator = legacy.load_network_pkl(f)['G_ema'].cuda() # type: ignore

#     def get_all_layers(self):
#         return list(self.generator.synthesis.children())

#     def get_training_layers(self, phase=None):
#         return self.get_all_layers()

#     def trainable_params(self):
#         params = []
#         for layer in self.get_training_layers():
#             params.extend(layer.parameters())

#         return params

#     def freeze_layers(self, layer_list=None):
#         '''
#         Disable training for all layers in list.
#         '''
#         if layer_list is None:
#             self.freeze_layers(self.get_all_layers())
#         else:
#             for layer in layer_list:
#                 requires_grad(layer, False)

#     def unfreeze_layers(self, layer_list=None):
#         '''
#         Enable training for all layers in list.
#         '''
#         if layer_list is None:
#             self.unfreeze_layers(self.get_all_layers())
#         else:
#             for layer in layer_list:
#                 requires_grad(layer, True)
#                 requires_grad(layer.affine, False)

#     def style(self, z_codes, truncation=1.0):
#         return self.generator.mapping(z_codes[0], None, truncation_psi=truncation, truncation_cutoff=8)

#     def forward(self, styles, input_is_latent=None, truncation=None, randomize_noise=None): # unused args for compatibility with SG2 interface
#         return self.generator.synthesis(styles, noise_mode='random', force_fp32=True), None

# class SG2Generator(torch.nn.Module):
#     def __init__(self, checkpoint_path, latent_size=512, map_layers=8, img_size=256, channel_multiplier=2, device='cuda:0'):
#         super(SG2Generator, self).__init__()

#         self.generator = Generator(
#             img_size, latent_size, map_layers, channel_multiplier=channel_multiplier
#         ).to(device)

#         checkpoint = torch.load(checkpoint_path, map_location=device)

#         self.generator.load_state_dict(checkpoint["g_ema"], strict=True)

#         with torch.no_grad():
#             self.mean_latent = self.generator.mean_latent(4096)

#     def get_all_layers(self):
#         return list(self.generator.children())

#     def get_training_layers(self, phase):

#         if phase == 'texture':
#             # learned constant + first convolution + layers 3-10
#             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
#         if phase == 'shape':
#             # layers 1-2
#              return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
#         if phase == 'no_fine':
#             # const + layers 1-10
#              return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
#         if phase == 'shape_expanded':
#             # const + layers 1-10
#              return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
#         if phase == 'all':
#             # everything, including mapping and ToRGB
#             return self.get_all_layers() 
#         else: 
#             # everything except mapping and ToRGB
#             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])  

#     def trainable_params(self):
#         params = []
#         for layer in self.get_training_layers():
#             params.extend(layer.parameters())

#         return params

#     def freeze_layers(self, layer_list=None):
#         '''
#         Disable training for all layers in list.
#         '''
#         if layer_list is None:
#             self.freeze_layers(self.get_all_layers())
#         else:
#             for layer in layer_list:
#                 requires_grad(layer, False)

#     def unfreeze_layers(self, layer_list=None):
#         '''
#         Enable training for all layers in list.
#         '''
#         if layer_list is None:
#             self.unfreeze_layers(self.get_all_layers())
#         else:
#             for layer in layer_list:
#                 requires_grad(layer, True)

#     def style(self, styles):
#         '''
#         Convert z codes to w codes.
#         '''
#         styles = [self.generator.style(s) for s in styles]
#         return styles

#     def get_s_code(self, styles, input_is_latent=False):
#         return self.generator.get_s_code(styles, input_is_latent)

#     def modulation_layers(self):
#         return self.generator.modulation_layers

#     #TODO Maybe convert to kwargs
#     def forward(self,
#         styles,
#         return_latents=False,
#         inject_index=None,
#         truncation=1,
#         truncation_latent=None,
#         input_is_latent=False,
#         input_is_s_code=False,
#         noise=None,
#         randomize_noise=True):
#         return self.generator(styles, return_latents=return_latents, truncation=truncation, truncation_latent=self.mean_latent, noise=noise, randomize_noise=randomize_noise, input_is_latent=input_is_latent, input_is_s_code=input_is_s_code)

# class SG2Discriminator(torch.nn.Module):
#     def __init__(self, checkpoint_path, img_size=256, channel_multiplier=2, device='cuda:0'):
#         super(SG2Discriminator, self).__init__()

#         self.discriminator = Discriminator(
#             img_size, channel_multiplier=channel_multiplier
#         ).to(device)

#         checkpoint = torch.load(checkpoint_path, map_location=device)

#         self.discriminator.load_state_dict(checkpoint["d"], strict=True)

#     def get_all_layers(self):
#         return list(self.discriminator.children())

#     def get_training_layers(self):
#         return self.get_all_layers() 

#     def freeze_layers(self, layer_list=None):
#         '''
#         Disable training for all layers in list.
#         '''
#         if layer_list is None:
#             self.freeze_layers(self.get_all_layers())
#         else:
#             for layer in layer_list:
#                 requires_grad(layer, False)

#     def unfreeze_layers(self, layer_list=None):
#         '''
#         Enable training for all layers in list.
#         '''
#         if layer_list is None:
#             self.unfreeze_layers(self.get_all_layers())
#         else:
#             for layer in layer_list:
#                 requires_grad(layer, True)

#     def forward(self, images):
#         return self.discriminator(images)

# def get_training_layers(self, phase):

#     if phase == 'texture':
#         # learned constant + first convolution + layers 3-10
#         return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
#     if phase == 'shape':
#         # layers 1-2
#             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
#     if phase == 'no_fine':
#         # const + layers 1-10
#             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
#     if phase == 'shape_expanded':
#         # const + layers 1-10
#             return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
#     if phase == 'all':
#         # everything, including mapping and ToRGB
#         return self.get_all_layers() 
#     else: 
#         # everything except mapping and ToRGB
#         return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:])  

class EG3DWrapper(torch.nn.Module):
    def __init__(self, eg3d_gen, device='cuda:0'):
        super(EG3DWrapper, self).__init__()
        self.eg3d_gen = eg3d_gen

    def get_all_layers(self):
        return list(self.eg3d_gen.children())

    def freeze_layers(self, layer_list=None):
        '''
        Disable training for all layers in list.
        '''
        if layer_list is None:
            self.freeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, False)

    def unfreeze_layers(self, layer_list=None):
        '''
        Enable training for all layers in list.
        '''
        if layer_list is None:
            self.unfreeze_layers(self.get_all_layers())
        else:
            for layer in layer_list:
                requires_grad(layer, True) 

    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.eg3d_gen.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        return self.eg3d_gen.backbone.mapping(z, c * self.eg3d_gen.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)

    def get_training_layers(self, phase):
        # if phase == 'texture':
        #     # learned constant + first convolution + layers 3-10
        #     return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][2:10])   
        # if phase == 'shape':
        #     # layers 1-2
        #         return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:2])
        # if phase == 'no_fine':
        #     # const + layers 1-10
        #         return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][:10])
        # if phase == 'shape_expanded':
        #     # const + layers 1-10
        #         return list(self.get_all_layers())[1:3] + list(self.get_all_layers()[4][0:3])
        if phase == 'all':
            # everything, including mapping and ToRGB
            return self.get_all_layers() 
        else: 
            # everything except mapping and ToRGB
            all_layers = self.get_all_layers()
            collect_layers = []
            collect_layers.append(all_layers[0])
            collect_layers.append(all_layers[1])

            # sg2/synthesis network
            sg2_conv_layers = []
            for res in all_layers[2].synthesis.block_resolutions:
                block = getattr(all_layers[2].synthesis, f'b{res}')
                if res == 4:
                    sg2_conv_layers.append(getattr(block, 'conv1'))
                else:
                    sg2_conv_layers.append(getattr(block, 'conv0'))
                    sg2_conv_layers.append(getattr(block, 'conv1'))
            collect_layers = collect_layers + sg2_conv_layers

            # SR network
            sr_conv_layers = [
                all_layers[3].block0.conv0,
                all_layers[3].block0.conv1,
                all_layers[3].block1.conv0,
                all_layers[3].block1.conv1
            ]
            collect_layers = collect_layers + sr_conv_layers

            collect_layers.append(all_layers[4])
            return collect_layers  

    def synthesis(self, ws, c, 
                update_emas=False, 
                neural_rendering_resolution=None, 
                cache_backbone=False, 
                use_cached_backbone=False, 
                **synthesis_kwargs
            ):
        return self.eg3d_gen.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

    def forward(self, z, c, 
                truncation_psi=1, truncation_cutoff=None,
                neural_rendering_resolution=None, 
                update_emas=False,
                cache_backbone=False, 
                use_cached_backbone=False, 
                **synthesis_kwargs
                ):
        return self.eg3d_gen(z, c, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)

class NADAGAN(torch.nn.Module):
    def __init__(self, args, generator_frozen, generator_trainable, device='cuda:0'):
        super(NADAGAN, self).__init__()

        self.args = args

        self.device = device

        # Set up twin generators
        self.generator_frozen = generator_frozen.to(self.device)
        self.generator_trainable = generator_trainable.to(self.device)

        # freeze relevant layers
        self.generator_frozen.freeze_layers()
        self.generator_frozen.eval()
        
        self.generator_trainable.freeze_layers()
        self.generator_trainable.unfreeze_layers(self.generator_trainable.get_training_layers(args.phase))
        self.generator_trainable.train()

        # Losses
        self.clip_loss_models = {model_name: CLIPLoss(self.device, 
                                                      lambda_direction=args.lambda_direction, 
                                                      lambda_patch=args.lambda_patch, 
                                                      lambda_global=args.lambda_global, 
                                                      lambda_manifold=args.lambda_manifold, 
                                                      lambda_texture=args.lambda_texture,
                                                      clip_model=model_name) 
                                for model_name in args.clip_models}

        self.clip_model_weights = {model_name: weight for model_name, weight in zip(args.clip_models, args.clip_model_weights)}

        self.mse_loss  = torch.nn.MSELoss()

        self.source_class = args.source_class
        self.target_class = args.target_class

        self.auto_layer_k     = args.auto_layer_k
        self.auto_layer_iters = args.auto_layer_iters
        
        if args.target_img_list is not None:
            self.set_img2img_direction()

    def set_img2img_direction(self):
        # with torch.no_grad():
        #     z_dim    = 64 if self.args.sgxl else 512
        #     sample_z = torch.randn(self.args.img2img_batch, z_dim, device=self.device)

        #     generated = self.generator_trainable(sample_z)[0]

        #     for _, model in self.clip_loss_models.items():
        #         direction = model.compute_img2img_direction(generated, self.args.target_img_list)

        #         model.target_direction = direction
        raise NotImplementedError

    def determine_opt_layers(self):
        z_dim    = 64 if self.args.sgxl else 512
        pitch_range = 0.25
        yaw_range = 0.35
        # sample z
        sample_z = torch.randn(self.args.auto_layer_batch, z_dim, device=self.device)
        # sample c
        sample_yaw = (yaw_range * torch.randn(self.args.auto_layer_batch, 1)).to(self.device)
        sample_pitch = (pitch_range * torch.randn(self.args.auto_layer_batch, 1)).to(self.device)
        camera_lookat_point = torch.tensor([0, 0, 0.2], device=self.device) 
        cam2world_pose = LookAtPoseSampler.sample(torch.pi/2 + sample_yaw, torch.pi/2 + sample_pitch, camera_lookat_point, radius=2.7, batch_size=self.args.auto_layer_batch, device=self.device)
        intrinsics = torch.tensor([[4.2647, 0, 0.5], [0, 4.2647, 0.5], [0, 0, 1]], device=self.device).unsqueeze(0).repeat(cam2world_pose.shape[0], 1, 1)
        sample_c = torch.cat([cam2world_pose.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1)

        initial_w_codes = self.generator_frozen.mapping(sample_z, sample_c, truncation_psi=1.0)  # (B, 14, 512)

        w_codes = torch.Tensor(initial_w_codes.cpu().detach().numpy()).to(self.device)

        w_codes.requires_grad = True

        w_optim = torch.optim.SGD([w_codes], lr=0.01)
        
        for _ in range(self.auto_layer_iters):
            w_codes_for_gen = w_codes
            generated_from_w = self.generator_trainable.synthesis(w_codes_for_gen, sample_c)['image']

            w_loss = [self.clip_model_weights[model_name] * self.clip_loss_models[model_name].global_clip_loss(generated_from_w, self.target_class) for model_name in self.clip_model_weights.keys()]
            w_loss = torch.sum(torch.stack(w_loss))
            
            w_optim.zero_grad()
            w_loss.backward()
            w_optim.step()
        
        layer_weights = torch.abs(w_codes - initial_w_codes).mean(dim=-1).mean(dim=0)
        chosen_layer_idx = torch.topk(layer_weights, self.auto_layer_k)[1].cpu().numpy()

        # assemble layers
        all_layers = list(self.generator_trainable.get_all_layers())
        # sg2/synthesis network
        sg2_conv_layers = []
        sg2_rgb_layers = []
        for res in all_layers[2].synthesis.block_resolutions:
            block = getattr(all_layers[2].synthesis, f'b{res}')
            if res == 4:
                sg2_conv_layers.append(getattr(block, 'conv1'))
            else:
                sg2_conv_layers.append(getattr(block, 'conv0'))
                sg2_conv_layers.append(getattr(block, 'conv1'))
            sg2_rgb_layers.append(getattr(block, 'torgb'))
        # SR network
        sr_conv_layers = [
            all_layers[3].block0.conv0,
            all_layers[3].block0.conv1,
            all_layers[3].block1.conv0,
            all_layers[3].block1.conv1
        ]
        sr_rgb_layers = [
            all_layers[3].block0.torgb,
            all_layers[3].block1.torgb
        ]

        conv_layers = sg2_conv_layers + sr_conv_layers
        rgb_layers = sg2_rgb_layers + sr_rgb_layers  # currently not optimized
        
        idx_to_layer = conv_layers # add initial convs to optimization

        chosen_layers = [idx_to_layer[idx] for idx in chosen_layer_idx] 

        # uncomment to add RGB layers to optimization.

        # for idx in chosen_layer_idx:
        #     if idx % 2 == 1 and idx >= 3 and idx < 14:
        #         chosen_layers.append(rgb_layers[(idx - 3) // 2])

        # uncomment to add learned constant to optimization
        # chosen_layers.append(all_layers[1])
                
        return chosen_layers

    def forward(
        self,
        styles,
        c,
        truncation_psi=1.0,
        input_is_latent=True,
    ):
        
        if self.training and self.auto_layer_iters > 0:
            self.generator_trainable.unfreeze_layers()
            train_layers = self.determine_opt_layers()

            if not isinstance(train_layers, list):
                train_layers = [train_layers]

            self.generator_trainable.freeze_layers()
            self.generator_trainable.unfreeze_layers(train_layers)

        with torch.no_grad():
            if input_is_latent:
                w_styles = styles
            else:
                w_styles = self.generator_frozen.mapping(styles, c, truncation_psi=truncation_psi)

            frozen_img = self.generator_frozen.synthesis(w_styles, c)['image']

            # if self.args.sg3 or self.args.sgxl:
            #     frozen_img = frozen_img + torch.randn_like(frozen_img) * 5e-4 # add random noise to add stochasticity in place of noise injections
        
        trainable_img = self.generator_trainable.synthesis(w_styles, c)['image']
        
        clip_loss = torch.sum(
            torch.stack(
                    [
                        self.clip_model_weights[model_name] * self.clip_loss_models[model_name](frozen_img, self.source_class, trainable_img, self.target_class) for model_name in self.clip_model_weights.keys()
                    ]
                )
            )

        return [frozen_img, trainable_img], clip_loss

    # def pivot(self):
    #     par_frozen = dict(self.generator_frozen.named_parameters())
    #     par_train  = dict(self.generator_trainable.named_parameters())

    #     for k in par_frozen.keys():
    #         par_frozen[k] = par_train[k]
