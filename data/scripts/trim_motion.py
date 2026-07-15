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
"""Auto-trim idle (near-static) head/tail from a directory of .motion files.

Per-frame "activity" = mean over bodies of ||rigid_body_vel||. The first/last
frames whose activity exceeds --thresh mark the active span; everything outside
(minus a small margin) is trimmed. Velocities are recomputed after trimming so
the new boundary frames are correct.

Best run BEFORE packaging / pyroki (on the source proto-sergey .motion files),
so pyroki's 15s budget isn't wasted on standing. Also works on retargeted
.motion files.

Run (protomotions env, PYTHONPATH=repo root):
    python data/Sony/tools/trim_motion.py \
        --in-dir data/Sony/run1/proto-sergey \
        --out-dir data/Sony/run1/proto-sergey-trimmed \
        --thresh 0.05 --margin 5
"""
import argparse
import glob
import os
from pathlib import Path

import torch

from protomotions.utils.rotations import quaternion_to_matrix
from protomotions.components.pose_lib import compute_kinematics_velocities
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument(
        "--thresh",
        type=float,
        default=0.05,
        help="activity threshold (m/s, mean body speed) for 'moving'",
    )
    ap.add_argument(
        "--margin", type=int, default=5, help="keep this many frames before/after"
    )
    ap.add_argument(
        "--min-frames", type=int, default=10, help="skip clip if active span shorter"
    )
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.in_dir, "*.motion")))
    print(f"found {len(files)} .motion files")

    kept, skipped = 0, 0
    for f in files:
        name = Path(f).name
        outp = os.path.join(args.out_dir, name)
        if os.path.exists(outp) and not args.force:
            continue
        d = torch.load(f, weights_only=False)
        m = RobotState.from_dict(d, state_conversion=StateConversion.COMMON)

        vel = m.rigid_body_vel  # (T, N, 3)
        activity = torch.linalg.norm(vel, dim=-1).mean(dim=-1)  # (T,)
        moving = activity > args.thresh
        idx = torch.nonzero(moving).flatten()
        T = vel.shape[0]
        if idx.numel() == 0:
            print(f"  {name}: ENTIRELY static (max act {activity.max():.3f}) -> skip")
            skipped += 1
            continue
        s = max(0, int(idx[0]) - args.margin)
        e = min(T, int(idx[-1]) + 1 + args.margin)
        if e - s < args.min_frames:
            print(f"  {name}: active span {e-s} < min_frames -> skip")
            skipped += 1
            continue

        def sl(x):
            return x[s:e] if x is not None else None

        m.rigid_body_pos = sl(m.rigid_body_pos)
        m.rigid_body_rot = sl(m.rigid_body_rot)
        m.rigid_body_contacts = sl(m.rigid_body_contacts)
        m.dof_pos = sl(m.dof_pos)

        # recompute velocities on the trimmed window
        rot_mats = quaternion_to_matrix(m.rigid_body_rot, w_last=True)
        lin, ang = compute_kinematics_velocities(
            m.rigid_body_pos, rot_mats, fps=int(m.fps), velocity_max_horizon=3
        )
        m.rigid_body_vel = lin
        m.rigid_body_ang_vel = ang
        if m.dof_vel is not None:
            m.dof_vel = sl(m.dof_vel)  # source dof_vel is zeros; fine

        torch.save(m.to_dict(), outp)
        print(f"  {name}: {T} -> {e-s} frames  (kept {s}:{e})")
        kept += 1

    print(f"done. kept {kept}, skipped {skipped}, out -> {args.out_dir}")


if __name__ == "__main__":
    with torch.no_grad():
        main()
