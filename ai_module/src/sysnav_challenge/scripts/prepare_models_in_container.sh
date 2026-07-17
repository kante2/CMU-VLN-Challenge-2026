#!/usr/bin/env bash
set -eo pipefail

cd /home/docker/ai_module/src/SysNav

external_dir=src/semantic_mapping/semantic_mapping/external
sam_checkpoint="$external_dir/sam2/checkpoints/sam2.1_hiera_base_plus.pt"
object_file=/home/docker/ai_module/src/sysnav_challenge/config/challenge_objects.yaml
object_checksum_file="$external_dir/challenge_objects.sha256"

current_object_checksum="$(sha256sum "$object_file" | cut -d' ' -f1)"
stored_object_checksum=""
if [[ -f "$object_checksum_file" ]]; then
  stored_object_checksum="$(cat "$object_checksum_file")"
fi

if [[ "$current_object_checksum" != "$stored_object_checksum" ]]; then
  echo "Challenge object vocabulary changed; detector engines will be regenerated."
  rm -f "$external_dir/yolov8x-worldv2_cus.engine" \
    "$external_dir/yoloe-26x-seg.engine"
fi

export SYSNAV_OBJECT_FILE="$object_file"

if [[ ! -s "$sam_checkpoint" ]]; then
  echo "Downloading the SAM2.1 base-plus checkpoint used by SysNav..."
  checkpoint_url=https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
  if command -v curl >/dev/null 2>&1; then
    curl -fL "$checkpoint_url" -o "$sam_checkpoint"
  else
    wget -O "$sam_checkpoint" "$checkpoint_url"
  fi
fi

if [[ ! -f "$external_dir/yolov8x-worldv2_cus.engine" ]]; then
  echo "Generating YOLO-World TensorRT engine (GPU required)..."
  python3 set_yolo_world.py
fi

if [[ ! -f "$external_dir/yoloe-26x-seg.engine" ]]; then
  echo "Generating YOLOE TensorRT engine (GPU required, may take a while)..."
  python3 set_yolo_e.py
fi

echo "$current_object_checksum" > "$object_checksum_file"

echo "SysNav detector engines are ready."
