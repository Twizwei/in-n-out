"""
Example:
python mapper/inference_hfgi3d.py --exp_dir /fs/nexus-projects/video3dgan/HFGI3D/preprocessed_imgs/${p}/styleclip/eyeglasses --checkpoint_path /fs/nexus-scratch/yiranx/codes/eg3d/eg3d/CLIPStyle/mapper_results/eyeglasses/checkpoints/best_model.pt --stylegan_weights /fs/nexus-projects/video3dgan/HFGI3D/preprocessed_imgs/${p}/checkpoints --latents_test_path /fs/nexus-projects/video3dgan/HFGI3D/preprocessed_imgs/${p}/embeddings/${p}/PTI --camera_params_path /fs/nexus-projects/video3dgan/HFGI3D/preprocessed_imgs/${p}/frames --stylegan_weights /fs/nexus-projects/video3dgan/HFGI3D/preprocessed_imgs/${p}/checkpoints --no_fine_mapper --factor_step 0.08

"""

import os
from argparse import Namespace

import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import time

from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

import legacy
import dnnlib
import pdb
from mapper.datasets.latents_dataset import LatentsDataset
from mapper.options.test_options import TestOptions
from mapper.styleclip_mapper import StyleCLIPMapper


def run(test_opts):
    out_path_results = os.path.join(test_opts.exp_dir, 'inference_results')
    os.makedirs(out_path_results, exist_ok=True)

    # update test options with options used during training
    ckpt = torch.load(test_opts.checkpoint_path, map_location='cpu')
    opts = ckpt['opts']
    opts.update(vars(test_opts))
    opts = Namespace(**opts)

    latent_files = sorted(os.listdir(opts.latents_test_path))

    video_name = opts.latents_test_path.split('/')[-2]
    device = torch.device('cuda')

    camera_params_dict = torch.load(os.path.join(opts.camera_params_path, video_name, 'cam_params.pt'))
    
    for i, latent_file in enumerate(tqdm(latent_files)):

        network_ckpt = os.path.join(opts.stylegan_weights, 'model_' + video_name + '_' + latent_file + '.pt')
        G = torch.load(network_ckpt)
        net = StyleCLIPMapper(opts, G, load_G=False)
        net.eval()
        net.to(device)

        test_latents = torch.load(os.path.join(opts.latents_test_path, latent_file, '0.pt')).detach()
        # test_cam_params = torch.from_numpy(np.load(os.path.join(opts.camera_params_path, latent_file+'.npy'))).unsqueeze(0)
        # test_cam_params = {'cam_intrinsics': test_cam_params[:, 16:], 'cam_params': test_cam_params[:, :16]}
  
        # test_cam_params = torch.cat((camera_params_dict['cam_params'][i:i+1], camera_params_dict['cam_intrinsics'].view(1, 9)), dim=1).to(device)
        test_cam_params = {'cam_intrinsics': camera_params_dict['cam_intrinsics'], 'cam_params': camera_params_dict['cam_params'][i:i+1]}
        # test_cam_params = camera_params_dict
        # camera_params = torch.cat((camera_params_dict['cam_params'][idx:idx+1], camera_params_dict['cam_intrinsics'].view(1, 9)), dim=1).to(device)
        dataset = LatentsDataset(latents=test_latents.cpu(),
                                            opts=opts, cam_params=test_cam_params)
        dataloader = DataLoader(dataset,
                                batch_size=opts.test_batch_size,
                                shuffle=False,
                                num_workers=int(opts.test_workers),
                                drop_last=False)

        if opts.n_images is None:
            opts.n_images = len(dataset)
        
        global_i = 0
        global_time = []
        latents_edit = []
        for input_batch in dataloader:
            # if global_i >= opts.n_images:
            # 	break
            with torch.no_grad():
                ws, camera_params = input_batch
                ws = ws.to(device)
                camera_params = camera_params.to(device)
                input_cuda = (ws, camera_params)
                tic = time.time()
                result_batch = run_on_batch(input_cuda, net, test_opts.couple_outputs, test_opts.factor_step)
                toc = time.time()
                global_time.append(toc - tic)

            for i in range(opts.test_batch_size):
                if i >= len(result_batch[0]):
                    break
                im_path = str(global_i).zfill(5)
                if test_opts.couple_outputs:
                    couple_output = torch.cat([result_batch[2][i].unsqueeze(0), result_batch[0][i].unsqueeze(0)])
                    torchvision.utils.save_image(couple_output, os.path.join(out_path_results, latent_file+'.png'), normalize=True, range=(-1, 1))
                else:
                    torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, latent_file+'.png'), normalize=True, range=(-1, 1))
                # torch.save(result_batch[1][i].detach().cpu(), os.path.join(out_path_results, f"latent_{im_path}.pt"))
                latents_edit.append(result_batch[1][i].detach().cpu())

                global_i += 1
        if test_opts.save_latents:
            torch.save(torch.stack(latents_edit), os.path.join(os.path.join(test_opts.exp_dir, "latents.pt")))
        stats_path = os.path.join(opts.exp_dir, 'stats.txt')
        result_str = 'Runtime {:.4f}+-{:.4f}'.format(np.mean(global_time), np.std(global_time))
        print(result_str)

        with open(stats_path, 'w') as f:
            f.write(result_str)
        
        del net
        del G
        torch.cuda.empty_cache()


def run_on_batch(inputs, net, couple_outputs=False, factor_step=0.1):
	w, c = inputs
	with torch.no_grad():
		w_hat = w + factor_step * net.mapper(w)
		x_hat = net.decoder.synthesis(w_hat, c, noise_mode='const')['image']
		result_batch = (x_hat, w_hat)
		if couple_outputs:
			x = net.decoder.synthesis(w, c, noise_mode='const')['image']
			result_batch = (x_hat, w_hat, x)
	return result_batch


if __name__ == '__main__':
	test_opts = TestOptions().parse()
	run(test_opts)
