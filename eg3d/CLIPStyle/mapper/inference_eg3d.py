"""
Example:
python mapper/inference_eg3d.py --exp_dir /home/yiranx/sensei-fs-symlink/users/yiranx/optim_multi_dynamic/blunt1_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/styleclip/eyeglasses --checkpoint_path /home/yiranx/sensei-fs-symlink/users/yiranx/pretrained_mapper/eyeglasses.pt --latents_test_path /home/yiranx/sensei-fs-symlink/users/yiranx/optim_multi_dynamic/blunt1_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/latents.pt --camera_params_path /home/yiranx/sensei-fs-symlink/users/yiranx/optim_multi_dynamic/blunt1_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/cam_params.pt --no_fine_mapper --factor_step 0.08 --save_latents

PTI:
python mapper/inference_eg3d.py --exp_dir /home/yiranx/sensei-fs-symlink/users/yiranx/optim_multi_dynamic/blunt1_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/pti_lr3e-4_epochs200/styleclip/eyeglasses --checkpoint_path /home/yiranx/sensei-fs-symlink/users/yiranx/pretrained_mapper/eyeglasses.pt --latents_test_path /home/yiranx/sensei-fs-symlink/users/yiranx/optim_multi_dynamic/blunt1_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/latents.pt --camera_params_path /home/yiranx/sensei-fs-symlink/users/yiranx/optim_multi_dynamic/blunt1_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/cam_params.pt --stylegan_weights /home/yiranx/sensei-fs-symlink/users/yiranx/optim_multi_dynamic/blunt1_16f_1.0lpips_1.0l2_lr5e-3_deltanorm_1e-3_0.7res_nomask_prepose_noUpdatePose/pti_lr3e-4_epochs150/model_pti_refine.pkl --no_fine_mapper --factor_step 0.08
"""

import os
from argparse import Namespace

import torchvision
import numpy as np
import torch
from torch.utils.data import DataLoader
import sys
import time
import pickle
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
	device = 'cuda'
	# with dnnlib.util.open_url(opts.stylegan_weights) as f:
	# 	G = legacy.load_network_pkl(f, g_ema_only=True)['G_ema'] # type: ignore
	if opts.stylegan_weights.endswith("pkl"):
		try:
			with dnnlib.util.open_url(opts.stylegan_weights) as f:
				G = legacy.load_network_pkl(f, g_ema_only=True)['G_ema'] # type: ignore
		except:
			with open(opts.stylegan_weights, 'rb') as f:
				G = pickle.load(f).to(device)
	elif opts.stylegan_weights.endswith("pt"):
		G = torch.load(opts.stylegan_weights).to(device)
	net = StyleCLIPMapper(opts, G, load_G=False)
	net.eval()
	net.cuda()

	test_latents = torch.load(opts.latents_test_path)['w_plus'].detach()
	if test_latents.shape[1] == 1:
		test_latents = test_latents.repeat(1, 14, 1)
	test_cam_params = torch.load(opts.camera_params_path)
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
	for input_batch in tqdm(dataloader):
		# if global_i >= opts.n_images:
		# 	break
		with torch.no_grad():
			ws, camera_params = input_batch
			ws = ws.cuda()
			camera_params = camera_params.cuda()
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
				torchvision.utils.save_image(couple_output, os.path.join(out_path_results, f"{im_path}.png"), normalize=True, range=(-1, 1))
			else:
				torchvision.utils.save_image(result_batch[0][i], os.path.join(out_path_results, f"{im_path}.png"), normalize=True, range=(-1, 1))
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
