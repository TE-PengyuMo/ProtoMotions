"""
Convert Blender-exported FBX skeleton NPZ files to ProtoMotions format.

NPZ files come from batch_export_fbx.py and contain RAW Blender world-space data
(already z-up, meters):
    bone_names        — (N,)          str
    global_positions  — (T, N, 3)     world positions of each bone
    global_rot_mats   — (T, N, 3, 3)  world rotation matrices of each bone
    fps               — scalar        (unused; --input-fps is authoritative)

Bodies are reindexed by NAME to match the source MJCF body order, so the export
bone order does not need to match the MJCF. Rotation matrices carry the baked
armature scale and are snapped to proper rotations (SVD); velocities are
finite-differenced; foot contacts are detected from height+velocity. dof_pos/
dof_vel are set to zeros — the source .pt is only used for keypoint extraction,
which reads gts/grs/contacts, not the DOFs.

Usage:
    python data/scripts/convert_fbx_npz_to_proto.py \
        --input-dir data/Sony/sie_data_npz/ \
        --output-dir data/Sony/proto-sergey/ \
        --mjcf data/Sony/sergey_humanoid.xml \
        --input-fps 120 --output-fps 30
"""
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer

from protomotions.components.pose_lib import (
    extract_kinematic_info,
    compute_kinematics_velocities,
)
from protomotions.utils.rotations import matrix_to_quaternion
from protomotions.simulator.base_simulator.simulator_state import (
    RobotState,
    StateConversion,
)

from contact_detection import compute_contact_labels_from_pos_and_vel

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def main(
    input_dir: Path = typer.Option(..., help="Directory containing .npz files."),
    output_dir: Path = typer.Option(..., help="Directory to save .motion files."),
    mjcf: Path = typer.Option(
        Path("data/Sony/sergey_humanoid.xml"), help="Source skeleton MJCF."
    ),
    input_fps: int = typer.Option(120, help="Input motion fps"),
    output_fps: int = typer.Option(30, help="Output motion fps"),
    vel_thres: float = typer.Option(0.15, help="Contact velocity threshold (m/s)."),
    height_thresh: float = typer.Option(0.10, help="Contact height threshold (m)."),
    force_remake: bool = False,
    yaml_output_name: Optional[str] = None,
):
    """Convert Blender FBX-exported NPZ motion files to ProtoMotions format."""
    device = torch.device("cpu")
    dtype = torch.float32

    kinematic_info = extract_kinematic_info(str(mjcf))
    print("kinematic_info:", kinematic_info)
    mjcf_names = list(kinematic_info.body_names)
    num_dofs = kinematic_info.num_dofs

    output_dir.mkdir(parents=True, exist_ok=True)

    if input_fps % output_fps != 0:
        raise ValueError(
            f"input_fps ({input_fps}) must be divisible by output_fps ({output_fps})"
        )
    downsample_factor = input_fps // output_fps

    npz_files = sorted(input_dir.glob("*.npz"))
    print(f"Found {len(npz_files)} npz files in {input_dir}")

    output_motions_yaml = []

    for npz_file in npz_files:
        motion_filename = npz_file.stem + ".motion"
        output_file = output_dir / motion_filename

        if not force_remake and output_file.exists():
            print(f"Skipping {motion_filename} (already exists)")
            continue

        print(f"Processing {npz_file}")

        try:
            data = np.load(npz_file, allow_pickle=True)

            missing = [
                k
                for k in ("bone_names", "global_positions", "global_rot_mats")
                if k not in data
            ]
            if missing:
                print(f"Skipping {npz_file}: missing {missing}")
                continue

            bone_names = [str(b) for b in data["bone_names"]]
            gpos = data["global_positions"]  # (T, N, 3)
            gmat = data["global_rot_mats"]  # (T, N, 3, 3)

            # Reindex bodies to MJCF order, by name.
            not_in_npz = [n for n in mjcf_names if n not in bone_names]
            if not_in_npz:
                print(f"Skipping {npz_file}: MJCF bodies not in NPZ: {not_in_npz}")
                continue
            order = [bone_names.index(n) for n in mjcf_names]
            gpos = gpos[:, order]  # (T, num_bodies, 3)
            gmat = gmat[:, order]  # (T, num_bodies, 3, 3)

            # Downsample
            gpos = gpos[::downsample_factor]
            gmat = gmat[::downsample_factor]

            pos = torch.from_numpy(gpos).to(device, dtype)
            rot_mats = torch.from_numpy(gmat).to(device, dtype)
            T = pos.shape[0]

            # The Blender/FBX export bakes the armature scale (~0.01, cm->m) into
            # global_rot_mats, so they are rotation*scale (row norms != 1), NOT pure
            # rotations. matrix_to_quaternion assumes orthonormal input and returns
            # sign-flipping quaternions on scaled matrices -- this corrupts the pelvis
            # heading (~180 deg/frame flips) and makes the retarget root spin. Snap
            # each matrix to its nearest proper rotation (SVD / Procrustes) first.
            U, _, Vh = torch.linalg.svd(rot_mats)
            det = torch.linalg.det(torch.matmul(U, Vh))  # (T, num_bodies)
            U = U.clone()
            U[..., :, -1] = U[..., :, -1] * det.sign().unsqueeze(-1)  # fix reflections
            rot_mats = torch.matmul(U, Vh)

            rot_quat = matrix_to_quaternion(rot_mats, w_last=True)  # xyzw

            lin_vel, ang_vel = compute_kinematics_velocities(
                pos, rot_mats, fps=output_fps, velocity_max_horizon=3
            )

            contacts = compute_contact_labels_from_pos_and_vel(
                pos, lin_vel, vel_thres=vel_thres, height_thresh=height_thresh
            ).to(torch.bool)

            # dof_pos/dof_vel are zeros: the NPZ only carries global body transforms
            # (no local joint rotations), and this source .pt is consumed only by
            # keypoint extraction (gts/grs/contacts), not the DOFs.
            motion = RobotState(state_conversion=StateConversion.COMMON)
            motion.rigid_body_pos = pos
            motion.rigid_body_rot = rot_quat
            motion.rigid_body_vel = lin_vel
            motion.rigid_body_ang_vel = ang_vel
            motion.rigid_body_contacts = contacts
            motion.dof_pos = torch.zeros(T, num_dofs, device=device, dtype=dtype)
            motion.dof_vel = torch.zeros(T, num_dofs, device=device, dtype=dtype)
            motion.local_rigid_body_rot = None
            motion.fps = float(output_fps)

            print(f"  rigid_body_pos:  {motion.rigid_body_pos.shape}")
            print(f"  contacts_any:    {bool(contacts.any())}")
            print(f"  Saving to {output_file}")
            torch.save(motion.to_dict(), str(output_file))

            if yaml_output_name is not None:
                output_motions_yaml.append(
                    {"file": motion_filename, "fps": output_fps}
                )

        except Exception as e:
            print(f"Error processing {npz_file}: {e}")
            import traceback

            traceback.print_exc()
            continue

    if yaml_output_name is not None:
        import yaml

        yaml_output = output_dir / yaml_output_name
        with open(yaml_output, "w") as f:
            yaml.dump({"motions": output_motions_yaml}, f)
        print(f"Saved motions list to {yaml_output}")


if __name__ == "__main__":
    with torch.no_grad():
        app()
