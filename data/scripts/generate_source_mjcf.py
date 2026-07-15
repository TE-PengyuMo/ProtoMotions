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
"""Generate a MuJoCo MJCF for the Sergey FBX skeleton (21 joints).

Offsets are parent-relative rest positions (meters, z-up) measured from
BaseSkeleton.fbx via dump_rest_offsets.py.

Run: python data/scripts/generate_source_mjcf.py
Writes: protomotions/data/assets/mjcf/sergey_humanoid.xml
"""

# name: (parent, (x, y, z) parent-relative offset, list of children)
# Tree is defined via parent pointers; children order = MJCF/body order.
TREE = [
    # name,                 parent,                 offset
    ("Sergey_Hips",         None,                   (0.0, 0.0, 0.0)),
    ("Sergey_Spine",        "Sergey_Hips",          (0.0, 0.0, 0.08619)),
    ("Sergey_Spine1",       "Sergey_Spine",         (0.0, 0.0, 0.23182)),
    ("Sergey_Neck",         "Sergey_Spine1",        (0.0, 0.0, 0.26693)),
    ("Sergey_Head",         "Sergey_Neck",          (0.0, -0.02102, 0.16292)),
    ("Sergey_LeftShoulder", "Sergey_Spine1",        (0.04134, 0.00147, 0.19982)),
    ("Sergey_LeftArm",      "Sergey_LeftShoulder",  (0.20427, 0.0, 0.0)),
    ("Sergey_LeftForeArm",  "Sergey_LeftArm",       (0.25641, 0.0, 0.0)),
    ("Sergey_LeftHand",     "Sergey_LeftForeArm",   (0.23419, -0.00153, 0.00115)),
    ("Sergey_RightShoulder","Sergey_Spine1",        (-0.04227, 0.00147, 0.19982)),
    ("Sergey_RightArm",     "Sergey_RightShoulder", (-0.20427, 0.0, 0.0)),
    ("Sergey_RightForeArm", "Sergey_RightArm",      (-0.25641, 0.0, 0.0)),
    ("Sergey_RightHand",    "Sergey_RightForeArm",  (-0.27439, 0.00044, -0.00023)),
    ("Sergey_LeftUpLeg",    "Sergey_Hips",          (0.10511, 0.0, 0.0)),
    ("Sergey_LeftLeg",      "Sergey_LeftUpLeg",     (0.0, 0.0, -0.39167)),
    ("Sergey_LeftFoot",     "Sergey_LeftLeg",       (-0.00128, -0.00055, -0.44295)),
    ("Sergey_LeftToeBase",  "Sergey_LeftFoot",      (0.0, -0.15767, -0.06832)),
    ("Sergey_RightUpLeg",   "Sergey_Hips",          (-0.10511, 0.0, 0.0)),
    ("Sergey_RightLeg",     "Sergey_RightUpLeg",    (0.0, 0.0, -0.39167)),
    ("Sergey_RightFoot",    "Sergey_RightLeg",      (0.00130, -0.00019, -0.44717)),
    ("Sergey_RightToeBase", "Sergey_RightFoot",     (0.0, -0.15767, -0.06832)),
]

ROOT_Z = 0.92260  # rest hip height (meters)

children = {n: [] for n, _, _ in TREE}
offset = {}
parent = {}
for name, par, off in TREE:
    parent[name] = par
    offset[name] = off
    if par is not None:
        children[par].append(name)


def emit_body(name, indent):
    pad = "  " * indent
    x, y, z = offset[name]
    lines = [f'{pad}<body name="{name}" pos="{x:.5f} {y:.5f} {z:.5f}">']
    if parent[name] is None:
        # root free joint
        lines.append(f'{pad}  <freejoint name="{name}"/>')
    else:
        for ax, vec in (("x", "1 0 0"), ("y", "0 1 0"), ("z", "0 0 1")):
            lines.append(
                f'{pad}  <joint name="{name}_{ax}" type="hinge" pos="0 0 0" '
                f'axis="{vec}" range="-180 180" limited="true" '
                f'stiffness="200" damping="20" armature="0.02"/>'
            )
    # tiny visual geom so the model is valid
    lines.append(
        f'{pad}  <geom type="sphere" size="0.03" pos="0 0 0" density="1000" '
        f'rgba="0.8 0.6 0.4 1" contype="1" conaffinity="1"/>'
    )
    for ch in children[name]:
        lines.extend(emit_body(ch, indent + 1))
    lines.append(f"{pad}</body>")
    return lines


root = TREE[0][0]
body_xml = "\n".join(emit_body(root, 3))

xml = f"""<mujoco model="sergey_humanoid">
  <compiler coordinate="local" angle="degree"/>
  <worldbody>
{body_xml}
  </worldbody>
</mujoco>
"""

out = "data/Sony/sergey_humanoid.xml"
with open(out, "w") as f:
    f.write(xml)
print(f"wrote {out} with {len(TREE)} bodies")
