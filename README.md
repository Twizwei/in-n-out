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
- [x] Training
- [ ] OOD removal
- [ ] Data preprocessing


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
The results will be saved at `eg3d/results/rednose2`.

## Preparing data[TODO]
1. Processed data. We provide a dataset of [preprocessed data](https://drive.google.com/file/d/1PRWZvLxtZexDG4PHTPiyp0WR0VFZoJFD/view?usp=sharing). Please download it and put it at `eg3d/data/wildvideos`
2. Your own data[TODO].

## Training
To train our model on a video, as an example, run
```
cd eg3d
bash scripts/run_train.sh rednose2 train
```
The results will be saved at `ckpts/rednose2/train`.

## References:
1. [EG3D](https://arxiv.org/abs/2112.07945), Chan et al. 2022
2. [Dynamic NeRF](https://arxiv.org/abs/2105.06468), Gao et al. 2021

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

## License
We use [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/). 
`eg3d/outdomain/*`, `eg3d/inversion/*`, `eg3d/frames2vid.py`, `eg3d/gen_3d_rgb.py`, `eg3d/vid2frames.py`, `w_avg.pt`, and other materials including the model checkpoints and shell scripts are licensed under the [Adobe Research License](https://git.corp.adobe.com/sooyek/layered-depth-refinement/blob/main/LICENSE.md#adobe-research-license).
`eg3d/configs/*`, `eg3d/criteria/*`, `eg3d/CLIPStyle/*` are under the [MIT License](https://github.com/danielroich/PTI?tab=MIT-1-ov-file).
Other files at `eg3d/dnnlib`, `eg3d/gui_utils`, `eg3d/torch_utils`, `eg3d/training`, `eg3d/gen_samples.py`, `eg3d/gen_videos.py`, `eg3d/legacy.py`, are licensed from NVIDIA.
Some images were from [Upsplash](http://www.unsplash.com/) under the [standard Unsplash license](https://unsplash.com/license).