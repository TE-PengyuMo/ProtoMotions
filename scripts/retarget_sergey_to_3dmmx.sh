#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Convenience script to retarget Sergey FBX mocap onto the 3dmmx humanoid (full pipeline)
#
# IMPORTANT: ProtoMotions and PyRoki require separate Python environments.
# You must provide paths to both Python interpreters (plus a Blender binary for FBX import).
#
# Usage: ./scripts/retarget_sergey_to_3dmmx.sh <blender> <proto_python> <pyroki_python> <fbx_root> <output_dir> [skip_freq] [trim_thresh] [gpu]
#
# Example:
#   ./scripts/retarget_sergey_to_3dmmx.sh \
#       /usr/bin/blender \
#       ~/miniconda3/envs/protomotions/bin/python \
#       ~/miniconda3/envs/pyroki/bin/python \
#       data/Sony/sie_data data/Sony/run1 1
#
# Arguments:
#   blender        Path to Blender binary (for FBX -> NPZ export)
#   proto_python   Path to Python interpreter with ProtoMotions installed
#   pyroki_python  Path to Python interpreter with PyRoki installed
#   fbx_root       Directory of source Sergey FBX mocap files
#   output_dir     Directory for all intermediate + final outputs
#   skip_freq      (Optional) Skip every N motions (default: 1 = all motions)
#   trim_thresh    (Optional) Idle-trim activity threshold in m/s, 0 disables (default: 0.05)
#   gpu            (Optional) CUDA device for PyRoki/JAX retargeting (default: 0)

set -e  # Exit on error

# Parse arguments
if [ $# -lt 5 ]; then
    echo "Usage: $0 <blender> <proto_python> <pyroki_python> <fbx_root> <output_dir> [skip_freq] [trim_thresh] [gpu]"
    echo ""
    echo "Arguments:"
    echo "  blender        Path to Blender binary (for FBX -> NPZ export)"
    echo "  proto_python   Path to Python interpreter with ProtoMotions installed"
    echo "  pyroki_python  Path to Python interpreter with PyRoki installed"
    echo "  fbx_root       Directory of source Sergey FBX mocap files"
    echo "  output_dir     Directory for all intermediate + final outputs"
    echo "  skip_freq      (Optional) Skip every N motions (default: 1 = all motions)"
    echo "  trim_thresh    (Optional) Idle-trim threshold in m/s, 0 disables (default: 0.05)"
    echo "  gpu            (Optional) CUDA device for PyRoki/JAX (default: 0)"
    echo ""
    echo "Example:"
    echo "  $0 /usr/bin/blender ~/miniconda3/envs/protomotions/bin/python ~/miniconda3/envs/pyroki/bin/python data/Sony/sie_data data/Sony/run1 1"
    exit 1
fi

BLENDER="$1"
PROTO_PYTHON="$2"
PYROKI_PYTHON="$3"
FBX_ROOT="$4"
OUTPUT_DIR="$5"
SKIP_FREQ="${6:-1}"
TRIM_THRESH="${7:-0.05}"
GPU="${8:-0}"

# Validate Blender binary exists
if [ ! -f "$BLENDER" ]; then
    echo "Error: Blender binary not found: $BLENDER"
    exit 1
fi

# Validate Python interpreters exist
if [ ! -f "$PROTO_PYTHON" ]; then
    echo "Error: ProtoMotions Python not found: $PROTO_PYTHON"
    exit 1
fi

if [ ! -f "$PYROKI_PYTHON" ]; then
    echo "Error: PyRoki Python not found: $PYROKI_PYTHON"
    exit 1
fi

# Validate input FBX directory exists
if [ ! -d "$FBX_ROOT" ]; then
    echo "Error: FBX root directory not found: $FBX_ROOT"
    exit 1
fi

# Run from the repo root so relative script paths and imports resolve
REPO="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="$REPO"
cd "$REPO"

# Output directories (all under the provided output_dir)
NPZ_DIR="${OUTPUT_DIR}/npz"
PROTO_SRC_DIR="${OUTPUT_DIR}/proto-sergey"
PROTO_SRC_TRIM_DIR="${OUTPUT_DIR}/proto-sergey-trimmed"
SOURCE_PT="${OUTPUT_DIR}/source-sergey.pt"
KEYPOINTS_DIR="${OUTPUT_DIR}/keypoints"
RETARGETED_DIR="${OUTPUT_DIR}/pyroki-3dmmx"
CONTACTS_DIR="${OUTPUT_DIR}/contacts"
PROTO_DIR="${OUTPUT_DIR}/proto-3dmmx"
FINAL_PT="${OUTPUT_DIR}/robot-3dmmx.pt"
mkdir -p "$OUTPUT_DIR"

echo "=============================================="
echo "Retargeting Sergey to 3DMMX"
echo "=============================================="
echo "Blender:             $BLENDER"
echo "ProtoMotions Python: $PROTO_PYTHON"
echo "PyRoki Python:       $PYROKI_PYTHON"
echo "Input FBX:           $FBX_ROOT"
echo "Output dir:          $OUTPUT_DIR"
echo "Skip freq:           $SKIP_FREQ (1 = all motions)"
echo "Trim thresh:         $TRIM_THRESH m/s (0 = disabled)"
echo "GPU:                 $GPU"
echo "=============================================="

# Step 1: Export FBX to NPZ (uses Blender)
echo ""
echo "[Step 1/9] Exporting FBX to NPZ (Blender)..."
"$BLENDER" --background --python data/scripts/batch_export_fbx.py -- \
    "$FBX_ROOT" \
    "$NPZ_DIR" \
    "$SKIP_FREQ"

# Step 2: Convert NPZ to ProtoMotions .motion (uses ProtoMotions)
echo ""
echo "[Step 2/9] Converting NPZ to ProtoMotions .motion..."
$PROTO_PYTHON data/scripts/convert_fbx_npz_to_proto.py \
    --input-dir "$NPZ_DIR" \
    --output-dir "$PROTO_SRC_DIR" \
    --mjcf data/Sony/sergey_humanoid.xml \
    --input-fps 120 \
    --output-fps 30

# Step 3: Trim idle head/tail (uses ProtoMotions; optional)
echo ""
if [ "$TRIM_THRESH" != "0" ]; then
    echo "[Step 3/9] Trimming idle head/tail (thresh=$TRIM_THRESH m/s)..."
    $PROTO_PYTHON data/scripts/trim_motion.py \
        --in-dir "$PROTO_SRC_DIR" \
        --out-dir "$PROTO_SRC_TRIM_DIR" \
        --thresh "$TRIM_THRESH" \
        --margin 5 \
        --force
    PACK_SRC_DIR="$PROTO_SRC_TRIM_DIR"
else
    echo "[Step 3/9] Skipping idle trim (trim_thresh=0)..."
    PACK_SRC_DIR="$PROTO_SRC_DIR"
fi

# Step 4: Package source MotionLib (uses ProtoMotions)
echo ""
echo "[Step 4/9] Packaging source MotionLib..."
$PROTO_PYTHON protomotions/components/motion_lib.py \
    --motion-path "$PACK_SRC_DIR" \
    --output-file "$SOURCE_PT"

# Step 5: Extract keypoints from source motions (uses ProtoMotions)
echo ""
echo "[Step 5/9] Extracting keypoints from Sergey motions..."
$PROTO_PYTHON data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
    "$SOURCE_PT" \
    --output-path "$KEYPOINTS_DIR" \
    --skeleton-format sergey \
    --start-idx 0 \
    --skip-freq 1 \
    --force-remake

# Step 6: Run PyRoki retargeting to 3DMMX (uses PyRoki)
echo ""
echo "[Step 6/9] Running PyRoki retargeting to 3DMMX (GPU $GPU)..."
$PYROKI_PYTHON pyroki/batch_retarget_to_3dmmx_from_keypoints.py \
    --gpu "$GPU" \
    --config pyroki/3dmmx_retarget_config.yaml \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --source-type sergey \
    --subsample-factor 1 \
    --output-dir "$RETARGETED_DIR" \
    --no-visualize \
    --skip-existing

# Step 7: Extract foot contact labels from source motions (uses PyRoki)
echo ""
echo "[Step 7/9] Extracting foot contact labels from source motions..."
$PYROKI_PYTHON pyroki/batch_retarget_to_3dmmx_from_keypoints.py \
    --gpu "$GPU" \
    --keypoints-folder-path "$KEYPOINTS_DIR" \
    --source-type sergey \
    --subsample-factor 1 \
    --save-contacts-only \
    --contacts-dir "$CONTACTS_DIR" \
    --skip-existing

# Step 8: Convert to ProtoMotions format with contact labels (uses ProtoMotions)
echo ""
echo "[Step 8/9] Converting to ProtoMotions format..."
$PROTO_PYTHON data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py \
    --retargeted-motion-dir "$RETARGETED_DIR" \
    --output-dir "$PROTO_DIR" \
    --robot-type 3dmmx \
    --contact-labels-dir "$CONTACTS_DIR" \
    --apply-motion-filter \
    --force-remake

# Step 9: Package into MotionLib (uses ProtoMotions)
echo ""
echo "[Step 9/9] Packaging into MotionLib..."
$PROTO_PYTHON protomotions/components/motion_lib.py \
    --motion-path "$PROTO_DIR" \
    --output-file "$FINAL_PT"

echo ""
echo "=============================================="
echo "Retargeting complete!"
echo "=============================================="
echo "Output MotionLib: $FINAL_PT"
echo ""
echo "To verify the result:"
echo "  python examples/motion_libs_visualizer.py --motion_files $FINAL_PT --robot 3dmmx --simulator isaacgym"
echo ""
