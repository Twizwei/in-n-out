VideoName=$1
EditDirection=$2
ComposeInvRoot=$3
ComposeInvCkpt=${ComposeInvRoot}/triplanes.pt
echo ${VideoName}
TargetPath=./data/wildvideos/${VideoName}
OutDir=./results/${VideoName}/eval/${EditDirection}_smoothed

StyleClipCkpt=./mapper_results/${EditDirection}/checkpoints/best_model.pt
LatentPath=${ComposeInvCkpt}
CamParaPath=${ComposeInvRoot}/cam_params.pt
ExpDir=./results/${VideoName}/styleclip/${EditDirection}


# get new latent codes
cd ./CLIPStyle
printf "Getting new latent codes..."
python mapper/inference_eg3d.py --exp_dir=../${ExpDir} --checkpoint_path=${StyleClipCkpt} --latents_test_path=../${LatentPath} --camera_params_path=../${CamParaPath} --factor_step 0.06 --save_latents
python mapper/inference_eg3d.py --exp_dir=../${ExpDir} --checkpoint_path=${StyleClipCkpt} --latents_test_path=../${LatentPath} --camera_params_path=../${CamParaPath} --factor_step 0.06 --save_latents --no_fine_mapper
# generate results
cd ../
printf "Rendering edited results..."
python outdomain/test_outdomain.py --network=pretrained_models/ffhqrebalanced512-128.pkl --ckpt_path=${ComposeInvCkpt} --target_path=${TargetPath} --latents_path=${ExpDir}/latents.pt --outdir=${OutDir} --smooth_out=True

# get videos
printf "Getting video..."
python frames2vid.py --frames_path=${OutDir}/frames/projected_sr --output_dir=${OutDir}/frames/projected_sr_${EditDirection}.mp4 --fps 30

