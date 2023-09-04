#!/bin/sh
# qsub -q qgpu -A OPEN-28-60 -l select=1,walltime=01:00:00 -I
# conda activate invsfm

data=cambridge

. ./scripts/export_path.sh

# cpu mode (!! DO NOT EDIT !!)
export CUDA_VISIBLE_DEVICES=""

# TODO
invsfm_mode=depth
invsfm_mode=depth_rgb
invsfm_mode=depth_sift_rgb
#invsfm_mode=depth_sift

# TODO
scene=ShopFacade # for debug
#scene=GreatCourt # for later
#scene=KingsCollege # for later
scene=OldHospital # for debug
#scene=StMarysChurch # for debug

# TODO
feat_name=rootsift+nn

# TODO
rec_mode=lines

######################################

#for invsfm_mode in depth depth_rgb depth_sift_rgb depth_sift
#for invsfm_mode in depth_rgb depth_sift_rgb
for invsfm_mode in depth_sift_rgb
do
#for scene in ShopFacade GreatCourt KingsCollege OldHospital StMarysChurch
#for scene in ShopFacade OldHospital StMarysChurch
#for scene in StMarysChurch
#for scene in OldHospital
for scene in ShopFacade
do

img_dir="$CAMBRIDGE_DIR"/"$scene"/

colmap_model_path="$WS_DIR"/tf/hloc/res/cambridge/num-covis-20_num-loc-10/"$scene"/sfm_rootsift+nn/

if [ "$rec_mode" = original ]; then
rec_colmap_model_path="$colmap_model_path"

out_dir="$RUN_DIR"/viz/invsfm-original/"$invsfm_mode"/"$data"/"$scene"/
#out_dir="$RUN_DIR"/viz/cambridge/"$scene"/original/

else
    #line_rec_dir="$WS_DIR"/tools/pcl_from_lcl_nn/res/cambridge_"$scene"/pcl_from_lcl/lcl_id1/trial1/
    line_rec_dir="$WS_DIR"/tools/pcl_from_lcl_nn/res/cambridge_"$scene"/pcl_from_lcl/lcl_id1/trial0/
    xp_name=L0-PI0-C0-K25-No1.0-O0
    #xp_name=L0-PI0-C0-K100-No0.9-O0
    rec_colmap_model_path="$line_rec_dir""$xp_name"/lcl_iter0/sparse_recovered/

    if ! [ -d "$colmap_model_path" ]; then
        echo "Error: no colmap model at "$colmap_model_path""
        exit 1
    fi

    out_dir="$RUN_DIR"/viz/"$xp_name"/"$invsfm_mode"/"$data"/"$scene"/
fi

mkdir -p "$out_dir"
echo "$out_dir"

#python3 demo_colmap_cambridge.py \
python3 run_colmap_cambridge.py \
  --colmap_model_path "$colmap_model_path" \
  --rec_colmap_model_path "$rec_colmap_model_path" \
  --input_attr "$invsfm_mode" \
  --num_samples 5 \
  --scene "$scene" \
  --image_path "$img_dir" \
  --output_path "$out_dir" \
  --mode "$rec_mode"

if [ "$?" -ne 0 ]; then
    echo "Error on "$invsfm_mode" "$scene""
    exit 1
fi

done
done
