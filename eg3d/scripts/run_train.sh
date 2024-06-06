DatasetRoot=./data/wildvideos
OutputRoot=./ckpts
VideoName=$1
ExpName=$2

# Training w/ masks. This is expected to have better performance.
python outdomain/train_outdomain.py --network=pretrained_models/ffhqrebalanced512-128.pkl \
    --target_path=${DatasetRoot}/${VideoName} \
    --latents_path=${DatasetRoot}/${VideoName}/latents.pt \
    --outdir=${OutputRoot}/${VideoName}/${ExpName} \
    --num_epochs_raw 200000 --max_num_iters_raw 10000 --num_epochs_sr 100 \
    --lr 5e-3 --lr_sr 1e-3 --batch_size 1 --save_intermediates=True \
    --use_raw_rgb_loss=true --use_mask=True --lamb_l2_raw_full 1.0 --lamb_l2_raw_o 1.0 \
    --lamb_lpips_raw_full 1.0 --lamb_lpips_raw_o 0.0 --lamb_l2_sr_full 1.0 \
    --lamb_lpips_sr_full 1.0 --lamb_l2_sr_o 0.0 --vis_step 2000 --return_raw_blendw=True  \
    --lamb_blendw_loss 1.0 --lamb_blendw_area_loss 1.0 --lamb_blendw_sparse_loss 0.0 \
    --train_raw=true --train_sr=true

# Training w/o masks
# python outdomain/train_outdomain_auto.py --network=pretrained_models/ffhqrebalanced512-128.pkl \
#     --target_path=${DatasetRoot}/${VideoName} \
#     --latents_path=${DatasetRoot}/${VideoName}/latents.pt \
#     --outdir=${OutputRoot}/${VideoName}/${ExpName} \
#     --num_epochs_raw 200000 --max_num_iters_raw 10000 --num_epochs_sr 400 \
#     --lr 5e-3 --lr_sr 1e-3 --batch_size 1 --save_intermediates=True \
#     --use_raw_rgb_loss=true --use_mask=False --lamb_l2_raw_full 1.0 --lamb_l2_raw_o 1.0 \
#     --lamb_lpips_raw_full 1.0 --lamb_lpips_raw_o 0.0 --lamb_l2_sr_full 1.0 \
#     --lamb_lpips_sr_full 1.0 --lamb_l2_sr_o 0.0 --vis_step 2000 --return_raw_blendw=True  \
#     --lamb_blendw_loss 1.0 --lamb_blendw_area_loss 1.0 --lamb_blendw_sparse_loss 0.0 \
#     --train_raw=true --train_sr=true


