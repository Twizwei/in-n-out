"""
Inverse processing of alignment.
"""


import argparse
import os
import json
from preprocess import align_img
from PIL import Image, ImageFilter, ImageDraw
import cv2
import numpy as np
import sys
sys.path.append('Deep3DFaceRecon_pytorch')
from Deep3DFaceRecon_pytorch.util.load_mats import load_lm3d
import pdb

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--indir', type=str, required=True, help='path to cropped/aligned frames')
    parser.add_argument('--origin_indir', type=str, required=True, help='path to the original (uncropped) frames')
    parser.add_argument('--ldm_dir', type=str, help='path to landmarks')
    parser.add_argument('--output_dir', type=str, required=True, help='output path.')
    args = parser.parse_args()

    lm_dir = os.path.join(args.origin_indir, "detections") if args.ldm_dir is None else args.ldm_dir
    img_files = sorted([x for x in os.listdir(args.indir) if x.lower().endswith(".png") or x.lower().endswith(".jpg")])
    img_origin_files = sorted([x for x in os.listdir(args.origin_indir) if x.lower().endswith(".png") or x.lower().endswith(".jpg")])
    lm_files = sorted([x for x in os.listdir(lm_dir) if x.endswith(".txt")])

    lm3d_std = load_lm3d("Deep3DFaceRecon_pytorch/BFM/") 

    out_dir = args.output_dir
    out_dir_bbox_coords = os.path.join(out_dir, 'bbox_coords')
    out_dir_bbox_frames = os.path.join(out_dir, 'frames')
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(out_dir_bbox_coords, exist_ok=True)
    os.makedirs(out_dir_bbox_frames, exist_ok=True)
    coords = []

    target_size = 1024.
    rescale_factor = 300
    center_crop_size = 700
    output_size = 512

    for n_iter, img_file in enumerate(img_files):
        # lm_file = img_file.replace('png', 'txt')
        lm_file = lm_files[n_iter]
        if os.path.exists(os.path.join(lm_dir, lm_file)):
            print(n_iter, img_file, lm_file)
            img_path = os.path.join(args.indir, img_file)
            lm_path = os.path.join(lm_dir, lm_file)
            im = Image.open(img_path).convert('RGB')

            # read original input image 
            im_origin = Image.open(os.path.join(args.origin_indir, img_origin_files[n_iter])).convert('RGB')
            
            _, H = im_origin.size
            lm = np.loadtxt(lm_path).astype(np.float32)
            lm = lm.reshape([-1, 2])
            lm[:, -1] = H - 1 - lm[:, -1]


            trans_params, im_high, _, _, = align_img(im_origin, lm, lm3d_std, target_size=target_size, rescale_factor=rescale_factor)
            t = np.zeros((2, 1))
            w0, h0, s, t[0], t[1] = trans_params

            # Unalignment
            # Rescale 512x512 -> center_crop_size x center_crop_size
            im_rescale_center_crop_size = im.resize((center_crop_size, center_crop_size), resample=Image.LANCZOS)

            # Un-center crop, paste it back
            left = int(im_high.size[0]/2 - center_crop_size/2)
            upper = int(im_high.size[1]/2 - center_crop_size/2)
            right = left + center_crop_size
            lower = upper + center_crop_size
            back_im = im_high.copy()
            
            back_im.paste(im_rescale_center_crop_size, (left, upper))  # ((162, 162), (162+700, 162+700)) in (1024, 1024)
            back_im_draw = ImageDraw.Draw(back_im)  
            back_im_draw.rectangle(((left, upper), (left+im_rescale_center_crop_size.size[0], upper+im_rescale_center_crop_size.size[0])), outline="yellow", width=5)
            left_back = left
            up_back = upper

            white_canvas = Image.new('RGB', back_im.size, (255, 255, 255))
            white_canvas.paste(im_rescale_center_crop_size, (left, upper))
            alpha = Image.new('RGB', back_im.size, (0, 0, 0))
            alpha.paste(Image.new('RGB', im_rescale_center_crop_size.size, (255, 255, 255)), (left, upper))
            # alpha = alpha.filter(ImageFilter.GaussianBlur(radius=50)).convert('L')
            # alpha = np.array(alpha)[:, :, np.newaxis]/255.

            alpha = np.array(alpha)/255.
            radius = 10
            kernel_size = (radius * 2 + 1, radius * 2 + 1)
            kernel = np.ones(kernel_size)
            eroded = cv2.erode(alpha, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            alpha = cv2.GaussianBlur(eroded, kernel_size, sigmaX=5.0)
            
            output_img = np.array(back_im) * (1 - alpha) + np.array(white_canvas) * alpha
            back_im = Image.fromarray(output_img.astype(np.uint8))


            # Un alignment crop
            w = (w0*s).astype(np.int32)
            h = (h0*s).astype(np.int32)
            left = (w/2 - target_size/2 + float((t[0] - w0/2)*s)).astype(np.int32)
            right = left + target_size
            up = (h/2 - target_size/2 + float((h0/2 - t[1])*s)).astype(np.int32)
            below = up + target_size
            im_origin_copy = im_origin.copy()
            im_origin_rescale = im_origin_copy.resize((w, h), resample=Image.BICUBIC)
            im_origin_rescale.paste(back_im, (left, up))  # ((162, 162), (162+700, 162+700)) in (1024, 1024) -> ((162+left, 162+up), (162+700+left, 162+700+up))
            

            # smooth boundary
            # white_canvas = Image.new('RGB', im_origin_rescale.size, (255, 255, 255))
            # white_canvas.paste(back_im, (left, up))
            # alpha = Image.new('RGB', im_origin_rescale.size, (0, 0, 0))
            # alpha.paste(Image.new('RGB', back_im.size, (255, 255, 255)), (left, up))
            # # alpha = alpha.filter(ImageFilter.GaussianBlur(radius=50)).convert('L')
            # # alpha = np.array(alpha)[:, :, np.newaxis]/255.
            # alpha = np.array(alpha)/255.
            # radius = 50
            # kernel_size = (radius * 2 + 1, radius * 2 + 1)
            # kernel = np.ones(kernel_size)
            # eroded = cv2.erode(alpha, kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0)
            # alpha = cv2.GaussianBlur(eroded, kernel_size, sigmaX=0.0)
            
            # output_img = np.array(im_origin_rescale) * (1 - alpha) + np.array(white_canvas) * alpha
            # im_origin_rescale = Image.fromarray(output_img.astype(np.uint8))
            
            
            # Unscale (t & s)
            im_origin_copy_ = im_origin.copy()
            img_draw = ImageDraw.Draw(im_rescale_center_crop_size)
            img_draw.rectangle(((5,5), (695,695)), outline="yellow", width=5)

            im_origin_copy_.paste(im_rescale_center_crop_size.resize((int(700/s), int(700/s))),   (int((left_back+left)/s), int((up_back+up)/s)))
            # (int((162+left)/s), int((162+up)/s)), (int((left_back+left)/s), int((up_back+up)/s))
            coords.append(np.array((int((left_back+left)/s), int((up_back+up)/s), int(700/s), int(700/s) ) ))
            out_path = os.path.join(out_dir_bbox_frames, img_file.split(".")[0] + ".png")
            # out_im_origin = im_origin_rescale.resize(im_origin.size)
            # out_im_origin.save(out_path)
            im_origin_copy_.save(out_path)

    coords = np.stack(coords)
    np.save(os.path.join(out_dir_bbox_coords, 'bbox_coords.npy'), coords)