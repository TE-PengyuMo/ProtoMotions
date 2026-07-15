# SPDX-FileCopyrightText: Copyright (c) 2025-2026 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Batch-export every FBX under a folder to per-motion NPZ (one Blender process).

Recurses <fbx_root>/**/*.fbx, writes <out_dir>/<session>_<name>.npz with the same
RAW Blender world-space data as export_fbx_motion.py (z-up, meters; Hips=joint 0).

Run:
    blender --background --python data/Sony/tools/batch_export_fbx.py -- \
        data/Sony/sie_data data/Sony/sie_data_npz [skip_freq]
"""
import bpy
import sys
import os
import glob
import numpy as np

argv = sys.argv
args = argv[argv.index("--") + 1:] if "--" in argv else []
fbx_root = args[0]
out_dir = args[1]
skip_freq = int(args[2]) if len(args) > 2 else 1
os.makedirs(out_dir, exist_ok=True)

fbx_files = sorted(glob.glob(os.path.join(fbx_root, "**", "*.fbx"), recursive=True))
fbx_files = fbx_files[::skip_freq]
print(f"found {len(fbx_files)} FBX files under {fbx_root} (skip_freq={skip_freq})")


def export_one(fbx_path, out_path):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.fbx(filepath=fbx_path)
    arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]

    ROOT = arm.name
    data_bones = list(arm.data.bones)
    names = [ROOT] + [b.name for b in data_bones]
    name_to_idx = {n: i for i, n in enumerate(names)}
    parent_idx = [-1]
    for b in data_bones:
        parent_idx.append(name_to_idx[b.parent.name] if b.parent else 0)

    sc = bpy.context.scene
    # Use the real action keyframe range (scene range defaults to 1-250 and
    # would truncate each clip to the first ~2s static intro).
    fr_min, fr_max = None, None
    for a in bpy.data.actions:
        r = a.frame_range
        fr_min = r[0] if fr_min is None else min(fr_min, r[0])
        fr_max = r[1] if fr_max is None else max(fr_max, r[1])
    if fr_min is not None:
        f0, f1 = int(round(fr_min)), int(round(fr_max))
    else:
        f0, f1 = sc.frame_start, sc.frame_end
    T, N = f1 - f0 + 1, len(names)
    root_pos = np.zeros((T, 3))
    gmat = np.zeros((T, N, 3, 3))
    gpos = np.zeros((T, N, 3))

    for ti, f in enumerate(range(f0, f1 + 1)):
        sc.frame_set(f)
        M = arm.matrix_world
        gpos[ti, 0] = np.array(M.translation)
        gmat[ti, 0] = np.array(M.to_3x3())
        for b in arm.pose.bones:
            Mw = M @ b.matrix
            bi = name_to_idx[b.name]
            gpos[ti, bi] = np.array(Mw.translation)
            gmat[ti, bi] = np.array(Mw.to_3x3())
        root_pos[ti] = gpos[ti, 0]

    np.savez_compressed(
        out_path,
        bone_names=np.array(names),
        parent_indices=np.array(parent_idx),
        root_positions=root_pos,
        global_rot_mats=gmat,
        global_positions=gpos,
        fps=sc.render.fps,
    )
    return T


for i, fbx in enumerate(fbx_files):
    rel = os.path.relpath(fbx, fbx_root)
    ident = rel.replace(os.sep, "_")[:-4]  # drop .fbx, flatten path
    out_path = os.path.join(out_dir, ident + ".npz")
    if os.path.exists(out_path):
        print(f"[{i+1}/{len(fbx_files)}] skip (exists) {ident}")
        continue
    try:
        T = export_one(fbx, out_path)
        print(f"[{i+1}/{len(fbx_files)}] {ident}  ({T} frames)")
    except Exception as e:
        print(f"[{i+1}/{len(fbx_files)}] FAILED {ident}: {e}")

print("done.")
