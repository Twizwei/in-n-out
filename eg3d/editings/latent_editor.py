"""
Example:
InterfaceGAN:
python latent_editor.py --checkpoints_dir ../full_models/ffhq1024x1024.pt --latents_path /home/yiranx/data/videogen_simple/person_00/latent_w.pt --saved_directions_path /home/yiranx/codes/StyleCLIP/interfacegan/saved/boundaries/stylesdf_smile/boundary.npy --results_dir ../evaluations/debug

GANspace:
python latent_editor.py --checkpoints_dir ../full_models/ffhq1024x1024.pt --latents_path /home/yiranx/data/videogen_simple/person_01/latent_w.pt --saved_directions_path /home/yiranx/codes/StyleSDF/evaluations/explore_directions/ganspace_80pca/cache/components/stylesdf-ffhq_style_ipca_c80_n1000000_w_render.npz --results_dir ../evaluations/debug
"""

import torch
import numpy as np
import cv2
import argparse
import os
import sys
sys.path.append(".")
sys.path.append("..")
from editings import ganspace
import legacy
import dnnlib
from camera_utils import LookAtPoseSampler

import pdb

class LatentEditor(object):
    def __init__(self, G):
        self.generator = G

    # def apply_ganspace(self, w_render_plus, w_render, ganspace_pca, edit_directions, w_gen_plus=None, output_latents=False, space='WRenderPlus'):
    #     edit_w_render, edit_w_render_plus = ganspace.edit(w_render, w_render_plus, ganspace_pca, edit_directions)
    #     if output_latents:
    #         return self._latents_to_image(edit_w_render_plus, edit_w_render, w_gen_plus, space=space), edit_w_render, edit_w_render_plus
    #     else:
    #         return self._latents_to_image(edit_w_render_plus, edit_w_render, w_gen_plus, space=space)

    def apply_interfacegan(self, ws, camera_params, direction, factor=1, factor_range=None, output_latents=False, space='WP', if_stack=True):
        edit_latents = []
        edit_latents_plus = []
        # import pdb; pdb.set_trace()
        if factor_range is not None:  # Apply a range of editing factors. for example, (-5, 5)
            # for f in range(*factor_range):
            for f in factor_range:
                edit_latent = ws + f * direction
                edit_latents.append(edit_latent)

            edit_latents = torch.cat(edit_latents)
        else:
            edit_latents = ws + factor * direction

        if output_latents:
            return self._latents_to_image(edit_latents, camera_params, space=space, if_stack=if_stack), edit_latents
        else:
            return self._latents_to_image(edit_latents, camera_params, space=space, if_stack=if_stack), None

    def _latents_to_image(self, ws, camera_params, space='WP', if_stack=True):
        with torch.no_grad():
            # get some camera parameters
            if camera_params.shape[0] == 1 and ws.shape[0] > 1:
                camera_params = camera_params.repeat(ws.shape[0], 1)
            if space == 'WP':
                images = self.generator.synthesis(ws, camera_params)['image']
            else:
                raise ValueError(f'Not implemented space-{space}!')
        if if_stack: # if horizontal concat
            horizontal_concat_image = torch.cat(list(images), 2)
            img_batch_torch = horizontal_concat_image.permute(1, 2, 0).detach()
        else:
            img_batch_torch = torch.stack(list(images), 0).permute(0, 2, 3, 1).detach()
        img_batch_torch = (img_batch_torch + 1) * (255/2)
        img_batch = img_batch_torch.clamp(0, 255).to(torch.uint8).cpu().numpy()

        return img_batch


if __name__ == '__main__':
    device = "cuda"
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_path", type=str, required=True, help="Where to save the results")
    parser.add_argument("--latents_path", type=str, required=True, help="Where to get the latent codes.")
    parser.add_argument("--camera_params_path", type=str, default=None, help="Where to get camera parameters.")
    parser.add_argument("--saved_directions_path", type=str, required=True, help="Path to the saved direction file.")
    parser.add_argument("--network_pkl", type=str, default='../pretrained_models/ffhqrebalanced512-128.pkl', help='Path to the pretrained checkpoint of G')
    parser.add_argument("--encode_space", type=str, default='WP', help="Path to the latents")
    parser.add_argument("--vis_idx", type=int, default=None, help="Which frame to visualize")
    parser.add_argument("--output_latents", action='store_true')

    args = parser.parse_args()

    # load a generator

    with dnnlib.util.open_url(args.network_pkl) as f:
        G = legacy.load_network_pkl(f, g_ema_only=True)['G_ema'].to(device) # type: ignore

    # wrap G into a editor
    editor = LatentEditor(G)

    # load latent codes
    if args.vis_idx is None:
        ws = torch.load(args.latents_path)['w_plus_cano'].to(device)
    else:
        ws = torch.load(args.latents_path)['w_plus'][args.vis_idx:args.vis_idx+1].to(device)

    # vis_idx = 10
    # # for encoder-based results...
    # w_render = w_render_plus[:, 0, :].clone().to(device)
    # w_render_plus = w_render_plus[vis_idx:vis_idx+1, :]
    # w_render = w_render[vis_idx:vis_idx+1, :]

    # get some cam parameters: --camera_params_path
    cams = torch.load(args.camera_params_path)
    intrinsics = cams['cam_intrinsics']
    if args.vis_idx is None:
        extrinsics = cams['cam_params'][0]
    else:
        extrinsics = cams['cam_params'][args.vis_idx]
    
    camera_params = torch.cat([extrinsics.reshape(-1, 16), intrinsics.reshape(-1, 9)], 1).to(device)

    ######### apply interfacegan ###########
    # load a pretrained interfacegan direction: --saved_directions_path
    direction = torch.from_numpy(np.load(args.saved_directions_path)).to(device)
    # edit!
    images, latents = editor.apply_interfacegan(ws, camera_params, direction, factor=1, factor_range=np.linspace(-2.0, 2.0, 7), output_latents=args.output_latents, space=args.encode_space)  # smile
    # images, latents = editor.apply_interfacegan(ws, camera_params, direction, factor=1, factor_range=np.linspace(-3.0, 3.0, 7), output_latents=args.output_latents, space=args.encode_space)  # age
    # images, latents = editor.apply_interfacegan(ws, camera_params, direction, factor=2.2, factor_range=None, output_latents=args.output_latents, space=args.encode_space)  # single eyeglasses
    ######### apply interfacegan ###########

    ######### apply ganspace ###########
    # direction = np.load(opt.inference.saved_directions_path)
    # # manually decide directions...
    # edit_directions = [
    #     (8, 0, 2, -2.0), # smile
    #     (8, 0, 2, -1.0),
    #     (8, 0, 2, 0.0),
    #     (8, 0, 2, 1.0),
    #     (8, 0, 2, 2.0),
    # ]
    # # edit_directions = [
    # #     (0, 0, 8, -1.0), # gender
    # #     (0, 0, 8, -1.0/4*3), 
    # #     (0, 0, 8, -1.0/4*2), 
    # #     (0, 0, 8, -1.0/4*1), 
    # #     (0, 0, 8, 0.0),
    # #     (0, 0, 8, 1.0/4*1),
    # #     (0, 0, 8, 1.0/4*2),
    # #     (0, 0, 8, 1.0/4*3),
    # #     (0, 0, 8, 1.0),
    # # ]
    # # # edit!
    # # images = editor.apply_ganspace(w_render_plus, w_render, direction, edit_directions)
    # images, latents, _ = editor.apply_ganspace(w_render_plus, w_render, direction, edit_directions, output_latents=True)
    ######### apply ganspace ###########

    # torch.save({'render_w':latents_out[0:1].cpu(), 'gen_w_plus':latents['gen_w_plus'][0:1].cpu()},'/home/yiranx/sensei-fs-symlink/users/yiranx/multiframe_static_w_cam_exps_realvideos/seoh_3_optim_project_lpips_init_w_plus_1l2_0.5lpips_lr1e-2_2000samples_600eposhs/pti_reg_2samples_use_w_plus/latents_angry.pt')
    # save: --results_path
    os.makedirs(args.results_path, exist_ok=True)
    cv2.imwrite(os.path.join(args.results_path, 'edited.png'), cv2.cvtColor(images, cv2.COLOR_BGR2RGB))