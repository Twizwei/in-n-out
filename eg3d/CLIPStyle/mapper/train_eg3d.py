"""
This file runs the main training/val loop
python mapper/train_eg3d.py --exp_dir ./mapper_results/eyeglasses --no_fine_mapper --description "face with eyeglasses" --factor_step 0.1 --latents_train_path /fs/nexus-projects/video3dgan/mapper_latents/train_faces.pt --latents_test_path /fs/nexus-projects/video3dgan/mapper_latents/test_faces.pt
"""
import os
import json
import sys
import pprint


sys.path.append("..")
sys.path.append(".")

import legacy
import dnnlib

from mapper.options.train_options import TrainOptions
from mapper.training_mapper.coach import Coach
# import training.volumetric_rendering

import pdb

def main(opts):
	# if os.path.exists(opts.exp_dir):
	# 	raise Exception('Oops... {} already exists'.format(opts.exp_dir))
	os.makedirs(opts.exp_dir, exist_ok=True)

	opts_dict = vars(opts)
	pprint.pprint(opts_dict)
	with open(os.path.join(opts.exp_dir, 'opt.json'), 'w') as f:
		json.dump(opts_dict, f, indent=4, sort_keys=True)
	with dnnlib.util.open_url(opts.stylegan_weights) as f:
		G = legacy.load_network_pkl(f, g_ema_only=True)['G_ema'] # type: ignore

	coach = Coach(opts, G)
	coach.train()


if __name__ == '__main__':
	opts = TrainOptions().parse()
	main(opts)
