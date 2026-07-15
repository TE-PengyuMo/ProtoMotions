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
"""Convert a 3-hinge-per-body humanoid MJCF into a URDF for PyRoki.

Each MJCF body with 3 hinge joints (x/y/z) becomes, in URDF:
    parent_link --(rev x)--> <body>_rx --(rev y)--> <body>_ry --(rev z)--> <body>
The named link (<body>) ends up at the body origin, so its FK position matches
the MJCF body position -- this is what PyRoki maps keypoints to.

Run:
    python mjcf_to_urdf.py data/3dmmx_neutral_no_fingers_eyes_light.xml \
        protomotions/data/assets/urdf/for_retargeting/3dmmx.urdf
"""
import sys
import math
import xml.etree.ElementTree as ET

mjcf_path = sys.argv[1]
urdf_path = sys.argv[2]

tree = ET.parse(mjcf_path)
root = tree.getroot()
worldbody = root.find("worldbody")

# MuJoCo default angle is "degree" unless compiler angle="radian"
compiler = root.find("compiler")
angle_unit = compiler.get("angle", "degree") if compiler is not None else "degree"


def to_rad(v):
    return math.radians(v) if angle_unit == "degree" else v


links = []   # (name, is_dummy)
joints = []  # (name, type, parent, child, origin_xyz, axis, lower, upper)


def find_root_body(wb):
    for b in wb.findall("body"):
        return b  # first top-level body = root
    return None


def walk(body, parent_link):
    name = body.get("name")
    pos = [float(x) for x in body.get("pos", "0 0 0").split()]
    hinges = [j for j in body.findall("joint") if j.get("type") == "hinge"]
    freej = body.findall("freejoint") + [
        j for j in body.findall("joint") if j.get("type") == "free"
    ]

    if parent_link is None:
        # root link: no joint, it's the URDF base
        links.append((name, False))
        cur = name
    elif freej:
        # shouldn't happen for non-root, but handle: fixed to parent
        links.append((name, False))
        joints.append((f"{name}_fixed", "fixed", parent_link, name, pos,
                       None, None, None))
        cur = name
    else:
        # chain 3 revolute joints with 2 dummy links, then the named link
        cur = parent_link
        chain = []
        for k, h in enumerate(hinges):
            axis = [float(x) for x in h.get("axis", "0 0 1").split()]
            rng = h.get("range")
            if rng:
                lo, hi = [to_rad(float(x)) for x in rng.split()]
            else:
                lo, hi = -math.pi, math.pi
            jname = h.get("name", f"{name}_{k}")
            if k < len(hinges) - 1:
                child = f"{name}_r{k}"
                links.append((child, True))
            else:
                child = name
                links.append((child, False))
            origin = pos if k == 0 else [0.0, 0.0, 0.0]
            chain.append((jname, "revolute", cur, child, origin, axis, lo, hi))
            cur = child
        joints.extend(chain)
        cur = name

    for child_body in body.findall("body"):
        walk(child_body, cur)


root_body = find_root_body(worldbody)
walk(root_body, None)

# ---- emit URDF ----
out = ['<?xml version="1.0"?>', '<robot name="3dmmx">']
INERTIAL = (
    '    <inertial><mass value="{m}"/>'
    '<inertia ixx="1e-4" iyy="1e-4" izz="1e-4" ixy="0" ixz="0" iyz="0"/>'
    '</inertial>'
)
for lname, is_dummy in links:
    out.append(f'  <link name="{lname}">')
    out.append(INERTIAL.format(m=0.001 if is_dummy else 1.0))
    out.append("  </link>")

for jname, jtype, parent, child, origin, axis, lo, hi in joints:
    out.append(f'  <joint name="{jname}" type="{jtype}">')
    out.append(f'    <parent link="{parent}"/>')
    out.append(f'    <child link="{child}"/>')
    ox, oy, oz = origin
    out.append(f'    <origin xyz="{ox:.6f} {oy:.6f} {oz:.6f}" rpy="0 0 0"/>')
    if jtype == "revolute":
        ax, ay, az = axis
        out.append(f'    <axis xyz="{ax:.0f} {ay:.0f} {az:.0f}"/>')
        out.append(f'    <limit lower="{lo:.4f}" upper="{hi:.4f}" '
                   f'effort="200" velocity="20"/>')
    out.append("  </joint>")

out.append("</robot>")

with open(urdf_path, "w") as f:
    f.write("\n".join(out) + "\n")

named_links = [n for n, d in links if not d]
print(f"wrote {urdf_path}")
print(f"  links: {len(links)} ({len(named_links)} named + "
      f"{len(links)-len(named_links)} dummy), joints: {len(joints)}")
print(f"  named links: {named_links}")
