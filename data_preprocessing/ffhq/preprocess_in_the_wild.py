#  SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
args = parser.parse_args()

# run mtcnn needed for Deep3DFaceRecon
command = "python batch_mtcnn.py --in_root " + args.indir
print(command)
os.system(command)

out_folder = args.indir.split("/")[-2] if args.indir.endswith("/") else args.indir.split("/")[-1]

# # smooth the landmarks
# command = "python smooth_video_lms.py -f " + os.path.join(args.indir, 'detections')
# print(command)
# os.system(command)

# 68 landmarks to 5 landmarks
# command = "python landmark68_5.py --landmark68_path=" + os.path.join(args.indir.replace('data/frames', 'data/landmarks_68'), 'landmarks.npy')  + " --frame_root=" + args.indir
# command = "python landmark68_5.py --landmark68_path=" + os.path.join(args.indir.replace('data/datasets', 'data/datasets/landmarks_68'), 'landmarks.npy')  + " --frame_root=" + args.indir

print(command)
os.system(command)

# run Deep3DFaceRecon
os.chdir('Deep3DFaceRecon_pytorch')
command = "python test.py --img_folder=" + args.indir + " --gpu_ids=0 --name=pretrained --epoch=20"
print(command)
os.system(command)
os.chdir('..')

# convert the pose to our format
command = "python 3dface2idr_mat.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/" + os.path.basename(args.indir)+"/epoch_20_000000"
print(command)
os.system(command)
# additional correction to match the submission version
command = "python preprocess_cameras.py --source Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/" + os.path.basename(args.indir)+"/epoch_20_000000 --mode orig"
print(command)
os.system(command)


# crop out the input image
command = "python crop_images_in_the_wild.py --indir=" + args.indir
print(command)
os.system(command)