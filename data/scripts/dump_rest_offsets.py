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
"""Dump rest-pose, parent-relative bone offsets (meters, z-up) for building an MJCF.

Run:
    blender --background --python dump_rest_offsets.py -- data/sie_data/BaseSkeleton.fbx
"""
import bpy
import sys
import numpy as np

argv = sys.argv
fbx_path = argv[argv.index("--") + 1] if "--" in argv else None

bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.import_scene.fbx(filepath=fbx_path)

arm = [o for o in bpy.data.objects if o.type == "ARMATURE"][0]
ROOT = arm.name
data_bones = list(arm.data.bones)
names = [ROOT] + [b.name for b in data_bones]
name_to_idx = {n: i for i, n in enumerate(names)}
parent = {ROOT: None}
for b in data_bones:
    parent[b.name] = b.parent.name if b.parent else ROOT

# rest-pose world positions (meters, z-up): use edit-bone head via armature world matrix
sc = bpy.context.scene
sc.frame_set(sc.frame_start)
M = arm.matrix_world

world_head = {}
# Hips = armature object origin
world_head[ROOT] = np.array((M).translation)
for b in arm.pose.bones:
    # at rest the pose matrix equals the rest matrix; head in world
    world_head[b.name] = np.array((M @ b.matrix).translation)

print("\n===== REST OFFSETS (meters, z-up, parent-relative) =====")
print(f"file: {fbx_path}")
print(f"root (Hips) world: {world_head[ROOT]}")
print(f"{'body':<24}{'parent':<24}local_pos (x y z)")
for n in names:
    p = parent[n]
    if p is None:
        off = np.array([0.0, 0.0, 0.0])
    else:
        off = world_head[n] - world_head[p]
    print(f"{n:<24}{str(p):<24}{off[0]:+.5f} {off[1]:+.5f} {off[2]:+.5f}")
print("===== END =====")
