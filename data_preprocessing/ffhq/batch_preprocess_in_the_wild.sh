InputRoot=$1
OutputRoot=$2
VideoName=$3
p=${VideoName}
mkdir -p ${OutputRoot}

printf "Aligning frames..."
python preprocess_in_the_wild.py --indir ${InputRoot}/${p}

mv ${InputRoot}/${p}/crop ${OutputRoot}/${p}
mv ${OutputRoot}/${p}/crop ${OutputRoot}/${p}/frames
rm -r ${OutputRoot}/${p}/crop
mv Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/${p}/epoch_20_000000/dataset.json ${OutputRoot}/${p}