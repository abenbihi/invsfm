#!/bin/sh
mode=depth
mode=depth_rgb
#mode=depth_sift_rgb
#mode=depth_sift

WS_DIR=/scratch/project/open-28-60/ws/
CAMBRIDGE_DIR="$WS_DIR"/datasets/cambridge/

# Add descriptors to the database with 
# python -m tests.tests_desc_in_colmap_db
# in /scratch/project/open-28-60/ws/tf/code_inv_sfm

#python3 demo_5k_assia.py \
#  --input_attr "$mode" \
#  --num_samples 1


# cpu mode
export CUDA_VISIBLE_DEVICES=""

#python3 demo_5k_cambridge_sift.py \
#  --input_attr "$mode" \
#  --num_samples 1

colmap_model_path=/scratch/project/open-28-60/ws/tf/hloc/res/cambridge/num-covis-20_num-loc-10/ShopFacade/sfm_rootsift+nn/

scene=ShopFacade
scene=GreatCourt
#scene=KingsCollege
#scene=OldHospital
#scene=StMarysChurch
img_dir="$CAMBRIDGE_DIR"/"$scene"/

colmap_model_path=/scratch/project/open-28-60/ws/tf/hloc/res/cambridge/num-covis-20_num-loc-10/"$scene"/sfm_rootsift+nn/

rec_mode=original
#rec_mode=rec
colmap_model_path="$WS_DIR"/tf/hloc/res/cambridge/num-covis-20_num-loc-10/"$scene"/sfm_rootsift+nn/

if [ "$rec_mode" = original ]; then
rec_colmap_model_path="$colmap_model_path"
out_dir=./viz/cambridge/"$scene"/original/

else

line_rec_dir="$WS_DIR"/tools/pcl_from_lcl_nn/res/cambridge_"$scene"/pcl_from_lcl/lcl_id1/trial1/
xp_name=L0-PI0-C0-K25-No1.0-O0
#xp_name=L0-PI0-C0-K100-No0.9-O0
rec_colmap_model_path="$line_rec_dir""$xp_name"/lcl_iter0/sparse_recovered/

if ! [ -d "$rec_colmap_model_path" ]; then
    echo "Error: no colmap model at "$rec_colmap_model_path""
    exit 1
fi

out_dir=./viz/cambridge/"$scene"/from_lines/"$xp_name"/
fi
mkdir -p "$out_dir"


python3 demo_colmap_cambridge.py \
  --colmap_model_path "$colmap_model_path" \
  --rec_colmap_model_path "$rec_colmap_model_path" \
  --input_attr "$mode" \
  --num_samples 5 \
  --scene "$scene" \
  --image_path "$img_dir" \
  --output_path "$out_dir" \
  --mode "$rec_mode"
