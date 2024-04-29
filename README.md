## In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for Face Editing<br><sub>Official PyTorch implementation of the CVPR 2024 paper</sub>

![Teaser image](./assets/teaser.mp4)

**In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for Face Editing**<br>
Yiran Xu, Zhixin Shu, Cameron Smith, Seoung Wug Oh, and Jia-Bin Huang
<br>https://in-n-out-3d.github.io/<br>

Abstract: *3D-aware GANs offer new capabilities for view synthesis while preserving the editing functionalities of their 2D counterparts. GAN inversion is a crucial step that seeks the latent code to reconstruct input images or videos, subsequently enabling diverse editing tasks through manipulation of this latent code. However, a model pre-trained on a particular dataset (e.g., FFHQ) often has difficulty reconstructing images with out-of-distribution (OOD) objects such as faces with heavy make-up or occluding objects. We address this issue by explicitly modeling OOD objects from the input in 3D-aware GANs. Our core idea is to represent the image using two individual neural radiance fields: one for the in-distribution content and the other for the out-of-distribution object. The final reconstruction is achieved by optimizing the composition of these two radiance fields with carefully designed regularization. We demonstrate that our explicit decomposition alleviates the inherent trade-off between reconstruction fidelity and editability. We evaluate reconstruction accuracy and editability of our method on challenging real face images and videos and showcase favorable results against other baselines.*


## Requirements

* We recommend Linux for performance and compatibility reasons.
* The code is built upon NVIDIA's [eg3d repo](https://github.com/NVlabs/eg3d).
* 64-bit Python 3.8 and PyTorch 1.11.0 (or later). See https://pytorch.org for PyTorch install instructions. We tested our code on Python 3.9.13 and PyTorch 1.12.1.
* Python libraries: see [environment.yml](./eg3d/environment.yml) for exact library dependencies.  You can use the following commands with Miniconda3 to create and activate your Python environment:
  - `conda env create -f environment.yml`
  - `conda activate in-n-out`

## TODO
- [x] Test/inference
- [ ] Training
- [ ] Data preprocessing
- [ ] 


## Getting started
Please download a pre-trained [EG3D checkpoint](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/research/models/eg3d/files), put it at `./eg3d/pretrained_models`.
```
mkdir -p eg3d/pretrained_models
wget --content-disposition 'https://api.ngc.nvidia.com/v2/models/org/nvidia/team/research/eg3d/1/files?redirect=true&path=ffhqrebalanced512-128.pkl' -O ./eg3d/pretrained_models/ffhqrebalanced512-128.pkl
```

To test our code, we provide a pre-trained checkpoint [here](https://drive.google.com/file/d/18WzDoRXtstpG_IbUmbLURblr8YX3gvR-/view?usp=sharing).
Please download the checkpoint and place it at `eg3d/ckpts`.

We also provide all StyleCLIP checkpoints [here](https://drive.google.com/file/d/1IVm-IcKXkAHPu8_eMOZTuKdmxSAZjVU2/view?usp=sharing). Please download them and unzip them at `eg3d/CLIPStyle/mapper_results`. (e.g., `unzip mapper_results.zip -d ./eg3d/CLIPStyle`)

To edit a video, as an example, run
```
cd eg3d
bash scripts/run_test_styleclip.sh rednose2 eyeglasses ckpts/rednose2
```
The resutls will be saved at `eg3d/results/rednose2`.

## Preparing data[TODO]

<!-- 1. Ensure the [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) submodule is properly initialized
```.bash
git submodule update --init --recursive
```

2. Run the following commands
```.bash
cd dataset_preprocessing/ffhq
python runme.py
```

Optional: preprocessing in-the-wild portrait images. 
In case you want to crop in-the-wild face images and extract poses using [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21) in a way that align with the FFHQ data above and the checkpoint, run the following commands 
```.bash
cd dataset_preprocessing/ffhq
python preprocess_in_the_wild.py --indir=INPUT_IMAGE_FOLDER
``` -->

## Training

You can train new networks using `train.py`. For example:

```.bash
# Train with FFHQ from scratch with raw neural rendering resolution=64, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_512.zip \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True

# Second stage finetuning of FFHQ to 128 neural rendering resolution (optional).
python train.py --outdir=~/training-runs --cfg=ffhq --data=~/datasets/FFHQ_512.zip \
  --resume=~/training-runs/ffhq_experiment_dir/network-snapshot-025000.pkl \
  --gpus=8 --batch=32 --gamma=1 --gen_pose_cond=True --neural_rendering_resolution_final=128

# Train with Shapenet from scratch, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=shapenet --data=~/datasets/cars_train.zip \
  --gpus=8 --batch=32 --gamma=0.3

# Train with AFHQ, finetuning from FFHQ with ADA, using 8 GPUs.
python train.py --outdir=~/training-runs --cfg=afhq --data=~/datasets/afhq.zip \
  --gpus=8 --batch=32 --gamma=5 --aug=ada --neural_rendering_resolution_final=128 --gen_pose_cond=True --gpc_reg_prob=0.8
```

Please see the [Training Guide](./docs/training_guide.md) for a guide to setting up a training run on your own data.

Please see [Models](./docs/models.md) for recommended training configurations and download links for pre-trained checkpoints.


The results of each training run are saved to a newly created directory, for example `~/training-runs/00000-ffhq-ffhq512-gpus8-batch32-gamma1`. The training loop exports network pickles (`network-snapshot-<KIMG>.pkl`) and random image grids (`fakes<KIMG>.png`) at regular intervals (controlled by `--snap`). For each exported pickle, it evaluates FID (controlled by `--metrics`) and logs the result in `metric-fid50k_full.jsonl`. It also records various statistics in `training_stats.jsonl`, as well as `*.tfevents` if TensorBoard is installed.

References:
1. [EG3D](https://arxiv.org/abs/2112.07945), Chan et al. 2022
2. [Dynamic NeRF](https://arxiv.org/abs/2105.06468), Gao et al. 2021

## License
We use [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/). 
... are licensed under the [Adobe Research License](https://git.corp.adobe.com/sooyek/layered-depth-refinement/blob/main/LICENSE.md#adobe-research-license).
... are licensed from NVIDIA.
Some images were from [Upsplash](http://www.unsplash.com/) under the [standard Unsplash license](https://unsplash.com/license).

## Citation

```
@inproceedings{Xu2024in,
  author = {Xu, Yiran and Shu, Zhixin and Smith Cameron and Oh, Seoung Wug and Huang, Jia-Bin},
  title = {In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for Face Editings},
  booktitle = {CVPR},
  year = {2024}
}
```

## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## Acknowledgements

