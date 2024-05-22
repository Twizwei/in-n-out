## In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for Face Editing<br><sub>Official PyTorch implementation of the CVPR 2024 paper</sub>

![Teaser image](./assets/teaser.jpg)

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

## Preparing data
1. Processed data. We provide a dataset of [preprocessed data](https://drive.google.com/file/d/1PRWZvLxtZexDG4PHTPiyp0WR0VFZoJFD/view?usp=sharing). Please download it and put it at `eg3d/data/wildvideos`
2. Your own data. This includes human **face alignment** and will use part of the code from official EG3D repo.
First, follow EG3D's [instructions](https://github.com/NVlabs/eg3d#preparing-datasets) on setting up `Deep3DFaceRecon_pytorch`.
```
cd data_preprocessing/ffhq/
git clone https://github.com/sicxu/Deep3DFaceRecon_pytorch.git
```
Install `Deep3DFaceRecon_pytorch`following the their [instructions](https://github.com/sicxu/Deep3DFaceRecon_pytorch/tree/6ba3d22f84bf508f0dde002da8fff277196fef21).

Also make sure you have their checkpoint file `epoch_20.pth` and place it at `data_preprocessing/ffhq/Deep3DFaceRecon_pytorch/checkpoints/pretrained/epoch_20.pth`.

We provide a scrip `batch_preprocess_in_the_wild.sh` to preprocess your own data of human faces.
The script accepts following folder tree (either a video or an image):
```
InputRoot
├── VideoName1
│   ├── frame1
│   ├── frame2
...
│   ├── frameN
└── ImageName1
    └── image1
...
```
Run
```
bash batch_preprocess_in_the_wild.sh ${InputRoot} ${OutputRoot} ${VideoName}
bash batch_preprocess_in_the_wild.sh ${InputRoot} ${OutputRoot} ${ImageName}
```

## Training
To train our model on a video, as an example, run
```
cd eg3d
bash scripts/run_train.sh rednose2 train
```
The results will be saved at `ckpts/rednose2/train`.

To run your data, run
```
bash scripts/run_train.sh ${videoname} ${expname}
```

## OOD object removal
```
# change to eg3d
cd eg3d
# Here we try to use the pre-trained checkpoint. Suppose it has been placed at ./ckpts/rednose2/
# Remove the OOD object.
python outdomain/test_outdomain.py --remove_ood=true --smooth_out=true --network=pretrained_models/ffhqrebalanced512-128.pkl --ckpt_path=./ckpts/rednose2/triplanes.pt --target_path Path-to-rednose2 --latents_path ./ckpts/rednose2/triplanes.pt --outdir ./results/rednose2/eval/ood_removal_smoothed
# Please replace `Path-to-rednose2` with your own path.

# Save it as a video.
python frames2vid.py --frames_path ./results/rednose2/eval/ood_removal_smoothed/frames/projected_sr  --output_dir ./results/rednose2/eval/ood_removal_smoothed/frames/projected_sr.mp4
```

## References:
1. [EG3D](https://arxiv.org/abs/2112.07945), Chan et al. 2022
2. [Dynamic NeRF](https://arxiv.org/abs/2105.06468), Gao et al. 2021

## Citation

```
@inproceedings{Xu2024inNout,
  author = {Xu, Yiran and Shu, Zhixin and Smith Cameron and Oh, Seoung Wug and Huang, Jia-Bin},
  title = {In-N-Out: Faithful 3D GAN Inversion with Volumetric Decomposition for Face Editings},
  booktitle = {CVPR},
  year = {2024}
}
```

## Development

This is a research reference implementation and is treated as a one-time code drop. As such, we do not accept outside code contributions in the form of pull requests.

## License
`data_preprocessing/ffhq/3dface2idr_mat.py`, `data_preprocessing/ffhq/batch_preprocess_in_the_wild.sh`, `data_preprocessing/ffhq/draw_images_in_the_wild.py`, `data_preprocessing/ffhq/smooth_video_lms.py`, `data_preprocessing/ffhq/landmark68_5.py`,`eg3d/outdomain/*`, `eg3d/inversion/*`, `eg3d/frames2vid.py`, `eg3d/gen_3d_rgb.py`, `eg3d/vid2frames.py`, `eg3d/scripts/*`, `eg3d/frames2vid.py`, `eg3d/vid2frames.py`, `w_avg.pt`, and other materials including the model checkpoints and shell scripts are licensed under the [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/).

Files at `eg3d/CLIPStyle/*` are from [StyleCLIP](https://github.com/orpatashnik/StyleCLIP/blob/main/LICENSE).

Files at `eg3d/configs/*`, `eg3d/criteria` are from [PTI](https://github.com/danielroich/PTI/blob/main/LICENSE).

Other files at `dataset_preprocessing`, `eg3d/dnnlib`, `eg3d/gui_utils`, `eg3d/torch_utils`, `eg3d/training`, and `eg3d/camera_utils.py`, `eg3d/cammat2json.py`, `eg3d/gen_3d_rgb.py`, `eg3d/gen_samples.py`, `eg3d/gen_videos.py`, `eg3d/legacy.py`, are licensed from [NVIDIA](https://github.com/NVlabs/eg3d/blob/main/LICENSE.txt).

Some images were from [Upsplash](http://www.unsplash.com/) under the [standard Unsplash license](https://unsplash.com/license).