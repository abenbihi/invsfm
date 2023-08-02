#!/bin/sh

#MACHINE=2
MACHINE=3
if [ "$MACHINE" -eq 2 ]; then # ciirc cluster
  WS_DIR=/home/benbiass/ws/
elif [ "$MACHINE" -eq 3 ]; then # karolina cluster
  WS_DIR=/scratch/project/open-28-60/ws/
else
  echo "Error: Unknown MACHINE="$MACHINE""
  exit 1
fi


container_path="$WS_DIR"/tf/invsfm/invsfm_pittaluga.sif

singularity shell --nv --bind "$WS_DIR":"$WS_DIR" "$container_path" 
