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
import time
from typing import Tuple, TypedDict
import glob
import os
import sys
import argparse
from pathlib import Path

if "--gpu" in sys.argv:
    _gi = sys.argv.index("--gpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = sys.argv[_gi + 1]
    del sys.argv[_gi : _gi + 2]

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
import yourdfpy

THREEDMMX_LINK_NAMES = None
N_retarget = 21
N_AUX = 0

# for the local bones alignment cost
direct_pairs = [
    # ("pelvis", "left_shoulder", 1.0),
    # ("pelvis", "right_shoulder", 1.0),
    ("left_shoulder", "left_elbow", 1.0),
    ("right_shoulder", "right_elbow", 1.0),
    ("left_elbow", "left_wrist", 1.0),
    ("right_elbow", "right_wrist", 1.0),
    ("left_hip", "left_knee", 1.0),
    ("right_hip", "right_knee", 1.0),
    ("left_knee", "left_ankle", 1.0),
    ("right_knee", "right_ankle", 1.0),
    ("left_ankle", "left_foot", 1.0),
    ("right_ankle", "right_foot", 1.0),
]


def get_humanoid_retarget_indices() -> jnp.ndarray:
    human_retarget_names = []
    threedmmx_joint_retarget_indices = []

    # NOTE: the order matters here.
    # Source keypoint name -> 3dmmx URDF link name.
    for human_name, threedmmx_link_name in [
        ("pelvis", "hip"),
        ("left_hip", "LeftUpLeg"),
        ("right_hip", "RightUpLeg"),
        ("left_knee", "LeftLeg"),
        ("right_knee", "RightLeg"),
        ("left_ankle", "LeftFoot"),
        ("right_ankle", "RightFoot"),
        ("left_foot", "LeftToeBase"),
        ("right_foot", "RightToeBase"),
        ("left_shoulder", "LeftArm"),
        ("right_shoulder", "RightArm"),
        ("left_elbow", "LeftForeArm"),
        ("right_elbow", "RightForeArm"),
        ("left_wrist", "LeftHand"),
        ("right_wrist", "RightHand"),
        ("left_clavicle", "LeftShoulder"),
        ("right_clavicle", "RightShoulder"),
        ("spine", "spine"),
        ("chest", "chest2"),
        ("neck", "neck"),
        ("head", "head"),
    ]:
        human_retarget_names.append(human_name)
        threedmmx_joint_retarget_indices.append(THREEDMMX_LINK_NAMES.index(threedmmx_link_name))

    threedmmx_joint_retarget_indices = jnp.array(threedmmx_joint_retarget_indices)

    return human_retarget_names, threedmmx_joint_retarget_indices


human_retarget_names, threedmmx_joint_retarget_indices = None, None


def load_motion_data(motion_path, source_type, subsample_factor, target_raw_frames):
    """Load and process motion data from a keypoints file.

    Args:
        motion_path: Path to the motion file
        source_type: Source type ('smpl' or 'rigv1')
        subsample_factor: Subsampling factor
        target_raw_frames: Target number of raw frames before subsampling

    Returns:
        Tuple of (simplified_keypoints, keypoint_orientations, left_foot_contact, right_foot_contact, num_timesteps)
    """
    print(f"Loading motion from: {motion_path}")
    if str(motion_path).endswith(".npz"):
        _z = onp.load(motion_path)  # portable plain-array format
        motion_data = {k: _z[k] for k in _z.files}
    else:
        motion_data = onp.load(motion_path, allow_pickle=True).item()

    # Compute target subsampled frames from raw frames and subsample factor
    target_subsampled_frames = len(list(range(0, target_raw_frames, subsample_factor)))

    raw_positions = motion_data["positions"]
    raw_orientations = motion_data["orientations"]
    raw_left_foot_contacts = motion_data[
        "left_foot_contacts"
    ]  # [T, 2] - ankle, toebase
    raw_right_foot_contacts = motion_data[
        "right_foot_contacts"
    ]  # [T, 2] - ankle, toebase
    original_raw_frames = raw_positions.shape[0]

    print(f"Original motion length: {original_raw_frames} frames.")

    # Calculate the original number of frames after subsampling for display purposes
    assert original_raw_frames > 0
    original_subsampled_display_count = raw_positions[::subsample_factor].shape[0]
    # Display frames are the minimum of original's useful frames and the buffer's capacity
    num_timesteps = min(original_subsampled_display_count, target_subsampled_frames)

    print(
        f"Motion will be displayed for {num_timesteps} subsampled frames (original subsampled count: {original_subsampled_display_count})."
    )

    # Pad or trim raw data to target_raw_frames for the solver's fixed-size input
    if original_raw_frames >= target_raw_frames:
        processed_positions = raw_positions[:target_raw_frames]
        processed_orientations = raw_orientations[:target_raw_frames]
        processed_left_foot_contacts = raw_left_foot_contacts[:target_raw_frames]
        processed_right_foot_contacts = raw_right_foot_contacts[:target_raw_frames]
    else:
        padding_count = target_raw_frames - original_raw_frames

        last_pos_frame = raw_positions[-1:]
        pos_padding = onp.repeat(last_pos_frame, padding_count, axis=0)
        processed_positions = onp.concatenate((raw_positions, pos_padding), axis=0)

        last_orient_frame = raw_orientations[-1:]
        orient_padding = onp.repeat(last_orient_frame, padding_count, axis=0)
        processed_orientations = onp.concatenate(
            (raw_orientations, orient_padding), axis=0
        )

        last_left_contact_frame = raw_left_foot_contacts[-1:]
        left_contact_padding = onp.repeat(
            last_left_contact_frame, padding_count, axis=0
        )
        processed_left_foot_contacts = onp.concatenate(
            (raw_left_foot_contacts, left_contact_padding), axis=0
        )

        last_right_contact_frame = raw_right_foot_contacts[-1:]
        right_contact_padding = onp.repeat(
            last_right_contact_frame, padding_count, axis=0
        )
        processed_right_foot_contacts = onp.concatenate(
            (raw_right_foot_contacts, right_contact_padding), axis=0
        )

    # Process contact labels BEFORE subsampling for better smoothing
    # Replace OR with average of ankle and toe contacts
    left_foot_contacts_avg = onp.mean(
        processed_left_foot_contacts.astype(float), axis=1
    )[:, None]
    right_foot_contacts_avg = onp.mean(
        processed_right_foot_contacts.astype(float), axis=1
    )[:, None]

    # Apply cross-fade (sliding window average) to smooth contact transitions
    window_size = 5

    def apply_crossfade(contact_flags):
        smoothed = onp.zeros_like(contact_flags)
        for i in range(len(contact_flags)):
            start_idx = max(0, i - window_size // 2)
            end_idx = min(len(contact_flags), i + window_size // 2 + 1)
            smoothed[i] = onp.mean(contact_flags[start_idx:end_idx])
        return smoothed

    left_foot_contacts_smoothed = apply_crossfade(left_foot_contacts_avg)
    right_foot_contacts_smoothed = apply_crossfade(right_foot_contacts_avg)

    # Subsample the processed (padded/trimmed) data for the solver's buffer
    simplified_keypoints = processed_positions[::subsample_factor]

    # Scale keypoints to roughly match the robot's size
    if source_type == "smpl":
        simplified_keypoints_root = simplified_keypoints[:, 0, :]
        simplified_keypoints_local = (
            simplified_keypoints - simplified_keypoints_root[:, None, :]
        )
        simplified_keypoints_lower_body_local = simplified_keypoints_local[:, 1:9, :]
        simplified_keypoints_lower_body_local = (
            simplified_keypoints_lower_body_local
            * onp.array([0.9, 0.9, 0.85])[None, None, :]
        )

        simplified_keypoints_upper_body_local = simplified_keypoints_local[
            :, 9 : N_retarget + N_AUX, :
        ]
        simplified_keypoints_upper_body_local = (
            simplified_keypoints_upper_body_local
            * onp.array([0.9, 0.9, 0.8])[None, None, :]
        )

        simplified_keypoints_local = onp.concatenate(
            [
                simplified_keypoints_lower_body_local,
                simplified_keypoints_upper_body_local,
            ],
            axis=1,
        )

        simplified_keypoints_root = (
            simplified_keypoints_root * onp.array([0.9, 0.9, 0.85])[None, :]
        )
        simplified_keypoints = (
            simplified_keypoints_root[:, None, :] + simplified_keypoints_local
        )
        simplified_keypoints = onp.concatenate(
            [simplified_keypoints_root[:, None, :], simplified_keypoints], axis=1
        )

    elif source_type == "rigv1":
        simplified_keypoints_root = simplified_keypoints[:, 0, :]
        simplified_keypoints_local = (
            simplified_keypoints - simplified_keypoints_root[:, None, :]
        )
        simplified_keypoints_lower_body_local = simplified_keypoints_local[:, 1:9, :]
        simplified_keypoints_lower_body_local = (
            simplified_keypoints_lower_body_local
            * onp.array([0.8, 0.8, 0.75])[None, None, :]
        )

        simplified_keypoints_upper_body_local = simplified_keypoints_local[
            :, 9 : N_retarget + N_AUX, :
        ]
        simplified_keypoints_upper_body_local = (
            simplified_keypoints_upper_body_local
            * onp.array([0.8, 0.8, 0.7])[None, None, :]
        )

        simplified_keypoints_local = onp.concatenate(
            [
                simplified_keypoints_lower_body_local,
                simplified_keypoints_upper_body_local,
            ],
            axis=1,
        )

        simplified_keypoints_root = (
            simplified_keypoints_root * onp.array([0.8, 0.8, 0.75])[None, :]
        )
        simplified_keypoints = (
            simplified_keypoints_root[:, None, :] + simplified_keypoints_local
        )
        simplified_keypoints = onp.concatenate(
            [simplified_keypoints_root[:, None, :], simplified_keypoints], axis=1
        )
    elif source_type == "sergey":
        # Sergey FBX and 3dmmx are both ~human-sized with similar limb lengths,
        # so keypoints need little scaling. TUNE these if limbs over/undershoot.
        lower_scale = onp.array([1.0, 1.0, 1.0])
        upper_scale = onp.array([1.0, 1.0, 1.0])
        root_scale = onp.array([1.0, 1.0, 1.0])

        # Keypoints-first: no shoulder-width narrowing. The per-joint scale
        # variable absorbs the small Sergey/3dmmx limb-width difference, so the
        # arms track the real shoulder keypoints directly.
        #
        # Optional arm de-abduction: 3dmmx shoulders are narrower, so hitting the
        # source's wide elbow/wrist forces the upper arm to splay out (elbow rides
        # ~10deg high). Shrink the *lateral* (body-sideways) offset of elbow/wrist
        # relative to their shoulder so the arm hangs more naturally. Only the
        # sideways component is scaled -- forward/back pumping is preserved.
        if _ARM_NARROW != 1.0:
            _lat = simplified_keypoints[:, 9, :] - simplified_keypoints[:, 10, :]
            _lat = _lat / (onp.linalg.norm(_lat, axis=1, keepdims=True) + 1e-8)
            for _sh, _js in [(9, [11, 13]), (10, [12, 14])]:
                for _j in _js:
                    _off = simplified_keypoints[:, _j, :] - simplified_keypoints[:, _sh, :]
                    _latc = onp.sum(_off * _lat, axis=1, keepdims=True) * _lat
                    simplified_keypoints[:, _j, :] = (
                        simplified_keypoints[:, _j, :] - (1.0 - _ARM_NARROW) * _latc
                    )

        # Toe-out: 3dmmx's ToeBase bone sits ~17deg lateral to the ankle-forward
        # line, so retargeting a straight source toe onto it makes the solver yaw
        # the whole foot inward (pigeon-toe) to reach the straight target. Splay
        # the source toe keypoints outward by the same angle so the neutral foot
        # already matches -- no compensating yaw. Rotate (toe - ankle) about the
        # vertical (world z) axis: +angle for the left toe (7 about ankle 5),
        # -angle for the right (8 about ankle 6). Rotation about z is heading-
        # independent, so the splay stays "outward" through turns.
        if _TOE_OUT != 0.0:
            _a = onp.deg2rad(_TOE_OUT)
            for _toe, _ank, _sgn in [(7, 5, +1.0), (8, 6, -1.0)]:
                _v = simplified_keypoints[:, _toe, :] - simplified_keypoints[:, _ank, :]
                _c, _s = onp.cos(_sgn * _a), onp.sin(_sgn * _a)
                _vx = _c * _v[:, 0] - _s * _v[:, 1]
                _vy = _s * _v[:, 0] + _c * _v[:, 1]
                simplified_keypoints[:, _toe, 0] = simplified_keypoints[:, _ank, 0] + _vx
                simplified_keypoints[:, _toe, 1] = simplified_keypoints[:, _ank, 1] + _vy

        simplified_keypoints_root = simplified_keypoints[:, 0, :]
        simplified_keypoints_local = (
            simplified_keypoints - simplified_keypoints_root[:, None, :]
        )
        simplified_keypoints_lower_body_local = (
            simplified_keypoints_local[:, 1:9, :] * lower_scale[None, None, :]
        )
        simplified_keypoints_upper_body_local = (
            simplified_keypoints_local[:, 9 : N_retarget + N_AUX, :]
            * upper_scale[None, None, :]
        )
        simplified_keypoints_local = onp.concatenate(
            [
                simplified_keypoints_lower_body_local,
                simplified_keypoints_upper_body_local,
            ],
            axis=1,
        )
        simplified_keypoints_root = simplified_keypoints_root * root_scale[None, :]
        simplified_keypoints = (
            simplified_keypoints_root[:, None, :] + simplified_keypoints_local
        )
        simplified_keypoints = onp.concatenate(
            [simplified_keypoints_root[:, None, :], simplified_keypoints], axis=1
        )
    else:
        raise ValueError(f"Invalid source type: {source_type}")

    keypoint_orientations = processed_orientations[::subsample_factor]

    # Subsample the smoothed contact flags
    left_foot_contact = left_foot_contacts_smoothed[::subsample_factor]
    right_foot_contact = right_foot_contacts_smoothed[::subsample_factor]

    expected_pos_shape = (target_subsampled_frames, N_retarget + N_AUX, 3)
    expected_orient_shape = (target_subsampled_frames, N_retarget + N_AUX, 3, 3)
    expected_contact_or_shape = (target_subsampled_frames, 1)
    assert (
        simplified_keypoints.shape == expected_pos_shape
    ), f"Expected positions shape {expected_pos_shape}, got {simplified_keypoints.shape}"
    assert (
        keypoint_orientations.shape == expected_orient_shape
    ), f"Expected orientations shape {expected_orient_shape}, got {keypoint_orientations.shape}"
    assert (
        left_foot_contact.shape == expected_contact_or_shape
    ), f"Expected left foot contacts OR shape {expected_contact_or_shape}, got {left_foot_contact.shape}"
    assert (
        right_foot_contact.shape == expected_contact_or_shape
    ), f"Expected right foot contacts OR shape {expected_contact_or_shape}, got {right_foot_contact.shape}"

    return (
        simplified_keypoints,
        keypoint_orientations,
        left_foot_contact,
        right_foot_contact,
        num_timesteps,
    )


def save_contact_labels(
    output_path, left_foot_contact, right_foot_contact, num_timesteps
):
    """Save processed foot contact labels to disk.

    Args:
        output_path: Path to save the contact labels
        left_foot_contact: Left foot contact array [T, 1]
        right_foot_contact: Right foot contact array [T, 1]
        num_timesteps: Number of actual timesteps (to trim padding)
    """
    # Extract contact labels (already smoothed from load_motion_data), trim to actual length
    left_contacts = left_foot_contact[:num_timesteps].squeeze(-1)  # [K]
    right_contacts = right_foot_contact[:num_timesteps].squeeze(-1)  # [K]

    # Stack into [K, 2] format (raw smoothed values, not binarized)
    foot_contacts = onp.stack([left_contacts, right_contacts], axis=-1)  # [K, 2]

    # Save contact labels
    onp.savez_compressed(output_path, foot_contacts=foot_contacts)
    print(f"Saved contact labels to {output_path} with shape {foot_contacts.shape}")


class RetargetingWeights(TypedDict):
    local_alignment: float
    """Local alignment weight, by matching the relative joint/keypoint positions and angles."""
    global_alignment: float
    """Global alignment weight, by matching the keypoint positions to the robot."""
    root_smoothness: float
    """Root smoothness weight, to penalize the robot's root from jittering too much."""
    joint_smoothness: float
    """Joint smoothness weight, to penalize the robot's joints from jittering too much."""
    self_collision: float
    """Self collision weight, to prevent the robot from colliding with itself."""
    joint_rest_penalty: float
    """Joint rest penalty weight, to penalize certain joints from moving too much."""
    joint_vel_limit: float
    """Joint velocity limit weight, to prevent joint velocities from exceeding limits."""
    foot_contact: float
    """Foot contact weight, to penalize foot movement when in contact."""
    foot_tilt: float
    """Foot tilt weight, to prevent excessive foot tilting when in contact."""
    foot_flat: float
    """Always-on (weak) foot-flat prior: keep each foot's z-axis near world-up so
    it doesn't tilt when the contact-gated foot_tilt cost is inactive. Weak enough
    that the swing foot can still lift."""


def main():
    """Main function for simplified humanoid retargeting."""
    # Get the directory containing this script for script-relative default paths
    SCRIPT_DIR = Path(__file__).parent.resolve()

    parser = argparse.ArgumentParser(description="Simplified Humanoid Retargeting")
    parser.add_argument(
        "--no-visualize",
        action="store_false",
        dest="visualize",
        help="Run retargeting without visualization and save results to disk.",
    )
    parser.add_argument(
        "--keypoints-folder-path",
        type=str,
        required=True,
        help="Path to the folder containing the keypoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./retargeted_output_motions",
        help="Directory to save retargeted motions in non-visualize mode.",
    )
    parser.add_argument(
        "--urdf-path",
        type=str,
        default=str(
            SCRIPT_DIR
            / "../data/Sony/3dmmx_neutral_no_fingers_eyes_light.urdf"
        ),
        help="Path to the URDF file for the robot. Default is script-relative.",
    )
    parser.add_argument(
        "--mesh-dir",
        type=str,
        default=str(SCRIPT_DIR / "../data/Sony"),
        help="Mesh dir (unused: 3dmmx URDF uses primitive geoms, no meshes).",
    )
    parser.add_argument(
        "--subsample-factor",
        type=int,
        default=1,
        help="Subsample factor for the keypoints. Adjust this to control memory usage and solve speed",
    )
    parser.add_argument(
        "--target-raw-frames",
        type=int,
        default=450,
        help="Target raw frames before subsampling.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing motions that already have retargeted output files (useful for resuming interrupted runs).",
    )
    parser.add_argument(
        "--source-type",
        type=str,
        default="sergey",
        help="Source type for the retargeting.",
    )
    parser.add_argument(
        "--save-contacts-only",
        action="store_true",
        help="Skip retargeting and only save processed foot contact labels from source motions.",
    )
    parser.add_argument(
        "--contacts-dir",
        type=str,
        default=None,
        help="Directory to save contact labels. Defaults to {keypoints_folder_path}/contacts",
    )
    parser.add_argument(
        "--input-fps",
        type=float,
        default=30.0,
        help="FPS of the input keypoint data (before subsampling). Used for velocity limit cost.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(SCRIPT_DIR / "3dmmx_retarget_config.yaml"),
        help="Retarget cost-weight config (flat `key: value` list). ALL cost "
        "weights are read from here -- edit the file to retune. See "
        "3dmmx_retarget_config.yaml for the keys.",
    )

    args = parser.parse_args()

    # All retarget cost weights are read straight from the config file (single
    # source of truth). Parsed without a yaml dependency (pyroki env has no
    # PyYAML): a flat `key: float  # comment` list.
    _REQUIRED_W = {
        "foot_flat", "foot_tilt", "joint_smooth", "toe_out", "foot_twist_rate",
        "foot_sagittal", "ee_weight", "arm_narrow", "global_align", "local_align",
        "foot_contact", "root_smooth", "joint_rest", "joint_vel_limit",
        "self_collision",
    }
    W = {}
    with open(args.config) as _f:
        for _line in _f:
            _line = _line.split("#", 1)[0].strip()
            if not _line or ":" not in _line:
                continue
            _k, _v = _line.split(":", 1)
            W[_k.strip()] = float(_v.strip())
    _missing = _REQUIRED_W - W.keys()
    assert not _missing, f"config {args.config} missing weights: {sorted(_missing)}"

    global _EE_W, _ARM_NARROW, _TOE_OUT, _FOOT_TWIST_RATE, _FOOT_SAGITTAL
    _EE_W = W["ee_weight"]
    _ARM_NARROW = W["arm_narrow"]
    _TOE_OUT = W["toe_out"]
    _FOOT_TWIST_RATE = W["foot_twist_rate"]
    _FOOT_SAGITTAL = W["foot_sagittal"]

    # Directory containing motion data files
    keypoints_folder_path = args.keypoints_folder_path

    # Dynamically populate the list of motion data paths (.npz preferred, .npy legacy)
    test_keypoints_paths = sorted(
        glob.glob(os.path.join(keypoints_folder_path, "*.npz"))
        + glob.glob(os.path.join(keypoints_folder_path, "*.npy"))
    )

    if not test_keypoints_paths:
        print(f"No .npy files found in {keypoints_folder_path}. Exiting.")
        return

    # Subsample factor
    subsample_factor = args.subsample_factor
    TARGET_RAW_FRAMES = args.target_raw_frames  # Target frames before subsampling

    # Early exit for save-contacts-only mode (skip robot/JAX initialization)
    if args.save_contacts_only:
        print(
            "Running in save-contacts-only mode. Extracting foot contact labels from source motions."
        )

        contacts_dir = (
            args.contacts_dir
            if args.contacts_dir
            else os.path.join(args.keypoints_folder_path, "contacts")
        )
        os.makedirs(contacts_dir, exist_ok=True)

        for i, motion_path in enumerate(test_keypoints_paths):
            print(
                f"Processing motion {i+1}/{len(test_keypoints_paths)}: {os.path.basename(motion_path)}"
            )

            # Check if output already exists and skip if requested
            base_filename = os.path.splitext(os.path.basename(motion_path))[0]
            output_filename = f"{base_filename}_contacts.npz"
            output_path = os.path.join(contacts_dir, output_filename)

            if args.skip_existing and os.path.exists(output_path):
                print(f"Output file {output_filename} already exists, skipping...")
                continue

            _, _, left_foot_contact, right_foot_contact, num_timesteps = (
                load_motion_data(
                    motion_path, args.source_type, subsample_factor, TARGET_RAW_FRAMES
                )
            )
            save_contact_labels(
                output_path, left_foot_contact, right_foot_contact, num_timesteps
            )
        return

    # Initialize robot and retargeting infrastructure (only needed for visualization and retargeting)
    global THREEDMMX_LINK_NAMES

    urdf_path = args.urdf_path
    urdf_mesh_dir = args.mesh_dir
    urdf = yourdfpy.URDF.load(urdf_path, mesh_dir=urdf_mesh_dir)

    robot = pk.Robot.from_urdf(urdf)
    # robot_coll = pk.collision.RobotCollision.from_urdf(urdf)
    robot_coll = None

    if THREEDMMX_LINK_NAMES is None:
        THREEDMMX_LINK_NAMES = list(robot.links.names)

    global human_retarget_names, threedmmx_joint_retarget_indices
    human_retarget_names, threedmmx_joint_retarget_indices = get_humanoid_retarget_indices()

    current_motion_index = 0

    # Create connectivity matrix for the 3dmmx joints
    n_retarget = len(threedmmx_joint_retarget_indices)
    threedmmx_retarget_mask = jnp.zeros((n_retarget, n_retarget))
    for link_a, link_b, weight in direct_pairs:
        retarget_idx_a = human_retarget_names.index(link_a)
        retarget_idx_b = human_retarget_names.index(link_b)
        threedmmx_retarget_mask = threedmmx_retarget_mask.at[retarget_idx_a, retarget_idx_b].set(
            weight
        )
        threedmmx_retarget_mask = threedmmx_retarget_mask.at[retarget_idx_b, retarget_idx_a].set(
            weight
        )

    weights_dict = RetargetingWeights(
        local_alignment=W["local_align"],
        global_alignment=W["global_align"],
        root_smoothness=W["root_smooth"],
        joint_smoothness=W["joint_smooth"],
        self_collision=W["self_collision"],  # 0.0 -> cost stays commented out
        joint_rest_penalty=W["joint_rest"],
        joint_vel_limit=W["joint_vel_limit"],
        foot_contact=W["foot_contact"],
        foot_tilt=W["foot_tilt"],
        foot_flat=W["foot_flat"],
    )

    if args.visualize:
        # Import GUI packages only when visualization is enabled
        import viser
        from viser.extras import ViserUrdf

        # Load initial motion data
        (
            simplified_keypoints,
            keypoint_orientations,
            left_foot_contact,
            right_foot_contact,
            num_timesteps,
        ) = load_motion_data(
            test_keypoints_paths[current_motion_index],
            args.source_type,
            subsample_factor,
            TARGET_RAW_FRAMES,
        )
        server = viser.ViserServer()
        base_frame = server.scene.add_frame("/base", show_axes=False)
        urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
        playing = server.gui.add_checkbox("playing", True)
        # Initialize slider with displayable frames for the first loaded motion
        timestep_slider = server.gui.add_slider(
            "timestep", 0, num_timesteps - 1 if num_timesteps > 0 else 0, 1, 0
        )

        def reset_timeline_callback(_: viser.GuiEvent):
            timestep_slider.value = 0

        reset_timeline_button = server.gui.add_button("Reset Timeline")
        reset_timeline_button.on_click(reset_timeline_callback)

        weights = pk.viewer.WeightTuner(
            server,
            weights_dict,  # type: ignore
        )

        Ts_world_root, joints = None, None

        def generate_trajectory():
            nonlocal Ts_world_root, joints
            gen_button.disabled = True
            retarget_next_button.disabled = True  # Disable while generating
            Ts_world_root, joints = solve_retargeting(
                robot=robot,
                robot_coll=robot_coll,
                target_keypoints=simplified_keypoints,  # Use current motion data
                target_orientations=keypoint_orientations,  # Use current motion data
                left_foot_contact=left_foot_contact,  # Use current motion data
                right_foot_contact=right_foot_contact,  # Use current motion data
                threedmmx_joint_retarget_indices=threedmmx_joint_retarget_indices,
                threedmmx_retarget_mask=threedmmx_retarget_mask,
                weights=weights.get_weights(),  # type: ignore
                subsample_factor=subsample_factor,
                input_fps=args.input_fps,
            )
            gen_button.disabled = False
            retarget_next_button.disabled = False  # Re-enable after generating

        gen_button = server.gui.add_button("Retarget!")
        gen_button.on_click(lambda _: generate_trajectory())

        def retarget_next_motion(_: viser.GuiEvent):
            nonlocal current_motion_index, Ts_world_root, joints, num_timesteps
            nonlocal \
                simplified_keypoints, \
                keypoint_orientations, \
                left_foot_contact, \
                right_foot_contact
            current_motion_index = (current_motion_index + 1) % len(
                test_keypoints_paths
            )
            (
                simplified_keypoints,
                keypoint_orientations,
                left_foot_contact,
                right_foot_contact,
                num_timesteps,
            ) = load_motion_data(
                test_keypoints_paths[current_motion_index],
                args.source_type,
                subsample_factor,
                TARGET_RAW_FRAMES,
            )

            # Update UI elements that depend on num_timesteps (displayable frames)
            timestep_slider.max = num_timesteps - 1 if num_timesteps > 0 else 0
            timestep_slider.value = 0

            # Clear previous trajectory visualization if any
            Ts_world_root, joints = None, None
            # server.scene.remove("/target_keypoints") # Optional: clear previous keypoints immediately

            generate_trajectory()

        retarget_next_button = server.gui.add_button("Retarget Next")
        retarget_next_button.on_click(retarget_next_motion)

        generate_trajectory()
        assert Ts_world_root is not None and joints is not None

        while True:
            with server.atomic():
                if playing.value and num_timesteps > 0:
                    timestep_slider.value = (timestep_slider.value + 1) % num_timesteps
                tstep = timestep_slider.value

            try:
                base_frame.wxyz = onp.array(Ts_world_root.wxyz_xyz[tstep][:4])
                base_frame.position = onp.array(Ts_world_root.wxyz_xyz[tstep][4:])
                urdf_vis.update_cfg(onp.array(joints[tstep]))

                server.scene.add_point_cloud(
                    "/target_keypoints",
                    onp.array(simplified_keypoints[tstep]),
                    onp.array((0, 0, 255))[None].repeat(
                        simplified_keypoints.shape[1], axis=0
                    ),
                    point_size=0.01,
                )
            except Exception as _:
                pass

            time.sleep(subsample_factor / args.input_fps)
    else:
        print(
            "Running in non-visualize mode. Retargeting all motions and saving to disk."
        )

        output_dir = args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        for i, motion_path in enumerate(test_keypoints_paths):
            print(
                f"Processing motion {i+1}/{len(test_keypoints_paths)}: {os.path.basename(motion_path)}"
            )

            # Check if output already exists and skip if requested
            base_filename = os.path.splitext(os.path.basename(motion_path))[0]
            output_filename = f"{base_filename}_retargeted.npz"
            output_path = os.path.join(output_dir, output_filename)

            if args.skip_existing and os.path.exists(output_path):
                print(f"Output file {output_filename} already exists, skipping...")
                continue

            (
                simplified_keypoints,
                keypoint_orientations,
                left_foot_contact,
                right_foot_contact,
                num_timesteps,
            ) = load_motion_data(
                motion_path, args.source_type, subsample_factor, TARGET_RAW_FRAMES
            )

            Ts_world_root, joints = solve_retargeting(
                robot=robot,
                robot_coll=robot_coll,
                target_keypoints=simplified_keypoints,
                target_orientations=keypoint_orientations,
                left_foot_contact=left_foot_contact,
                right_foot_contact=right_foot_contact,
                threedmmx_joint_retarget_indices=threedmmx_joint_retarget_indices,
                threedmmx_retarget_mask=threedmmx_retarget_mask,
                weights=weights_dict,
                subsample_factor=subsample_factor,
                input_fps=args.input_fps,
            )

            # Save results, sliced to the actual motion length
            results_to_save = {
                "base_frame_pos": onp.array(Ts_world_root.wxyz_xyz[:num_timesteps, 4:]),
                "base_frame_wxyz": onp.array(
                    Ts_world_root.wxyz_xyz[:num_timesteps, :4]
                ),
                "joint_angles": onp.array(joints[:num_timesteps]),
            }

            onp.savez_compressed(output_path, **results_to_save)
            print(f"Saved retargeted motion to {output_path}")


@jaxls.Cost.create_factory
def joint_vel_limit_cost(
    var_values: jaxls.VarValues,
    var_joints_curr: jaxls.Var[jnp.ndarray],
    var_joints_prev: jaxls.Var[jnp.ndarray],
    max_vel: float,
    dt: float,
    weight: float,
) -> jax.Array:
    """Joint velocity limit cost to prevent excessive joint velocities."""
    joints_curr = var_values[var_joints_curr]
    joints_prev = var_values[var_joints_prev]

    # Calculate joint velocity
    joint_vel = (joints_curr - joints_prev) / dt

    # Apply penalty when velocity exceeds limit
    excess_vel = jnp.maximum(jnp.abs(joint_vel) - max_vel, 0.0)

    return excess_vel.flatten() * weight


@jaxls.Cost.create_factory
def foot_contact_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root_curr: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
    var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,  # [1] - average of ankle and toebase contact (cross-faded)
    right_foot_contact: jnp.ndarray,  # [1] - average of ankle and toebase contact (cross-faded)
    threedmmx_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,  # [4] - left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx
    weight: float,
) -> jax.Array:
    """When either ankle or toe is in contact, penalize velocity of both ankle and toe,
    and also penalize ankle and toe being at different z heights."""
    T_world_root_curr = var_values[var_Ts_world_root_curr]
    T_world_root_prev = var_values[var_Ts_world_root_prev]
    robot_cfg_curr = var_values[var_robot_cfg_curr]
    robot_cfg_prev = var_values[var_robot_cfg_prev]

    # Get current and previous link positions
    T_root_link_curr = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_curr))
    T_root_link_prev = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg_prev))
    T_world_link_curr = T_world_root_curr @ T_root_link_curr
    T_world_link_prev = T_world_root_prev @ T_root_link_prev

    # Unpack foot indices
    left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx = foot_indices

    left_ankle_robot_idx = threedmmx_joint_retarget_indices[left_ankle_idx]
    right_ankle_robot_idx = threedmmx_joint_retarget_indices[right_ankle_idx]
    left_foot_robot_idx = threedmmx_joint_retarget_indices[left_foot_idx]
    right_foot_robot_idx = threedmmx_joint_retarget_indices[right_foot_idx]

    # Get current and previous robot foot positions
    robot_positions_curr = T_world_link_curr.translation()
    robot_positions_prev = T_world_link_prev.translation()

    left_ankle_curr = robot_positions_curr[left_ankle_robot_idx]
    right_ankle_curr = robot_positions_curr[right_ankle_robot_idx]
    left_foot_curr = robot_positions_curr[left_foot_robot_idx]
    right_foot_curr = robot_positions_curr[right_foot_robot_idx]

    left_ankle_prev = robot_positions_prev[left_ankle_robot_idx]
    right_ankle_prev = robot_positions_prev[right_ankle_robot_idx]
    left_foot_prev = robot_positions_prev[left_foot_robot_idx]
    right_foot_prev = robot_positions_prev[right_foot_robot_idx]

    # Calculate velocities (position differences between timesteps)
    left_ankle_vel = left_ankle_curr - left_ankle_prev
    right_ankle_vel = right_ankle_curr - right_ankle_prev
    left_foot_vel = left_foot_curr - left_foot_prev
    right_foot_vel = right_foot_curr - right_foot_prev

    # Calculate z-height differences between ankle and toe (current timestep).
    # NOTE: the ToeBase link origin sits ~4.6 cm BELOW the ankle origin when the foot
    # is flat (URDF LeftToeBase xyz z=-0.046395, RightToeBase z=-0.044313). Driving the
    # raw origin z-diff to 0 therefore forces ~20deg toe-up dorsiflexion every stance
    # frame. Subtract the structural offset so this cost enforces a FLAT sole instead.
    left_ankle_toe_z_diff = left_ankle_curr[2] - left_foot_curr[2] - 0.046395
    right_ankle_toe_z_diff = right_ankle_curr[2] - right_foot_curr[2] - 0.044313

    # Apply contact weighting - when either ankle or toe is in contact, penalize velocities
    left_contact_weight = left_foot_contact[0]  # OR of ankle and toe contact
    right_contact_weight = right_foot_contact[0]  # OR of ankle and toe contact

    # Velocity penalty costs (penalize movement when in contact)
    left_ankle_vel_cost = left_contact_weight * left_ankle_vel
    right_ankle_vel_cost = right_contact_weight * right_ankle_vel
    left_foot_vel_cost = left_contact_weight * left_foot_vel
    right_foot_vel_cost = right_contact_weight * right_foot_vel

    # Z-height consistency costs (ankle and toe should be at similar z when in contact)
    left_z_consistency_cost = left_contact_weight * left_ankle_toe_z_diff
    right_z_consistency_cost = right_contact_weight * right_ankle_toe_z_diff

    return (
        jnp.concatenate(
            [
                left_ankle_vel_cost.flatten(),
                right_ankle_vel_cost.flatten(),
                left_foot_vel_cost.flatten(),
                right_foot_vel_cost.flatten(),
                jnp.array([left_z_consistency_cost]),  # scalar, so wrap in array
                jnp.array([right_z_consistency_cost]),  # scalar, so wrap in array
            ]
        )
        * weight
    )


@jaxls.Cost.create_factory
def foot_tilt_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    left_foot_contact: jnp.ndarray,  # [1] - average of ankle and toebase contact (cross-faded)
    right_foot_contact: jnp.ndarray,  # [1] - average of ankle and toebase contact (cross-faded)
    threedmmx_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,  # [4] - left_ankle_idx, right_ankle_idx, left_foot_idx, right_foot_idx
    weight: float,
) -> jax.Array:
    """Cost to penalize foot tilting when in contact - keep z axis up."""
    T_world_root = var_values[var_Ts_world_root]
    robot_cfg = var_values[var_robot_cfg]
    T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
    T_world_link = T_world_root @ T_root_link

    # Unpack foot indices - we need the ankle indices for orientation
    left_ankle_idx, right_ankle_idx, _, _ = foot_indices

    left_ankle_robot_idx = threedmmx_joint_retarget_indices[left_ankle_idx]
    right_ankle_robot_idx = threedmmx_joint_retarget_indices[right_ankle_idx]

    # Get foot orientations (rotation matrices)
    left_foot_ori = T_world_link.rotation().as_matrix()[left_ankle_robot_idx]
    right_foot_ori = T_world_link.rotation().as_matrix()[right_ankle_robot_idx]

    # Penalize tilting: z-axis should point up (rotation_matrix[2,2] should be close to 1)
    left_contact_weight = left_foot_contact[0]  # OR of ankle and toe contact
    right_contact_weight = right_foot_contact[0]  # OR of ankle and toe contact

    left_tilt_residual = left_contact_weight * (left_foot_ori[2, 2] - 1.0)
    right_tilt_residual = right_contact_weight * (right_foot_ori[2, 2] - 1.0)

    return (
        jnp.concatenate(
            [
                jnp.array([left_tilt_residual]),  # scalar, so wrap in array
                jnp.array([right_tilt_residual]),  # scalar, so wrap in array
            ]
        )
        * weight
    )


# Ankle->toe vector in the FOOT LINK frame (URDF ToeBase origin offsets, normalized).
# This is the true gauge axis: the two foot keypoints (ankle origin, toe origin) never
# constrain rotation ABOUT this line, so the swing foot is free to twist around it.
# Note it is ~ -y with a downward droop (NOT local-x).
_TWIST_AXIS_L = jnp.array([0.038283, -0.122615, -0.046395])
_TWIST_AXIS_L = _TWIST_AXIS_L / jnp.linalg.norm(_TWIST_AXIS_L)
_TWIST_AXIS_R = jnp.array([-0.038743, -0.121504, -0.044313])
_TWIST_AXIS_R = _TWIST_AXIS_R / jnp.linalg.norm(_TWIST_AXIS_R)


@jaxls.Cost.create_factory
def foot_twist_rate_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root_curr: jaxls.SE3Var,
    var_Ts_world_root_prev: jaxls.SE3Var,
    var_robot_cfg_curr: jaxls.Var[jnp.ndarray],
    var_robot_cfg_prev: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    gate_l: jnp.ndarray,  # [1] precomputed source-data constant: airborne * near-vertical
    gate_r: jnp.ndarray,  # [1]
    threedmmx_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    """Gauge-DOF regularizer for the running swing foot. The two foot keypoints
    (ankle, toe) never constrain rotation about the ankle->toe line, so during the
    airborne deep-plantarflexion phase the solver picks arbitrary per-frame twist and
    jitters. Penalize the RATE of that twist only: the body-frame SO3-log of the
    frame-to-frame foot rotation, projected on the material ankle->toe axis. Zero
    preferred pose -> cannot fight plantarflexion pitch (unlike the absolute-roll
    anchor, which failed exactly because 'roll=0' conflicts with the demanded pitch).
    The gate is a source-data constant, ~0 during walking and during all contact, so
    the term is provably inert outside running swing. Well-conditioned at every pitch
    (log of a small dR never degenerates), unlike the arctan2 roll metric."""
    Rc = (
        var_values[var_Ts_world_root_curr]
        @ jaxlie.SE3(robot.forward_kinematics(cfg=var_values[var_robot_cfg_curr]))
    ).rotation().as_matrix()
    Rp = (
        var_values[var_Ts_world_root_prev]
        @ jaxlie.SE3(robot.forward_kinematics(cfg=var_values[var_robot_cfg_prev]))
    ).rotation().as_matrix()
    li = threedmmx_joint_retarget_indices[foot_indices[0]]
    ri = threedmmx_joint_retarget_indices[foot_indices[1]]
    w_l = (
        jaxlie.SO3.from_matrix(Rp[li]).inverse() @ jaxlie.SO3.from_matrix(Rc[li])
    ).log()
    w_r = (
        jaxlie.SO3.from_matrix(Rp[ri]).inverse() @ jaxlie.SO3.from_matrix(Rc[ri])
    ).log()
    return (
        jnp.array([gate_l[0] * (w_l @ _TWIST_AXIS_L),
                   gate_r[0] * (w_r @ _TWIST_AXIS_R)])
        * weight
    )


@jaxls.Cost.create_factory
def foot_sole_sagittal_cost(
    var_values: jaxls.VarValues,
    var_Ts_world_root: jaxls.SE3Var,
    var_robot_cfg: jaxls.Var[jnp.ndarray],
    robot: pk.Robot,
    gate_l: jnp.ndarray,  # [1] airborne * near-vertical constant (per frame)
    gate_r: jnp.ndarray,  # [1]
    threedmmx_joint_retarget_indices: jnp.ndarray,
    foot_indices: jnp.ndarray,
    weight: float,
) -> jax.Array:
    """Airborne sole-sagittal anchor. The foot roll about its long axis is a gauge DOF
    (two foot keypoints don't constrain it); with only a rate penalty it stays smooth
    but settles at an arbitrary sideways lean, so the swing sole visibly leans to one
    side. Penalize the LATERAL component of the sole normal = (foot local-z) . (root
    local-x, the body's lateral axis) -> pushes the sole back into the sagittal plane.
    Well-conditioned at every pitch (sole normal always defined) and ORTHOGONAL to
    plantarflexion: pitch tilts the sole fore-aft (in the sagittal plane), contributing
    zero lateral component, so deep plantarflexion is untouched. Gated to airborne +
    near-vertical (running swing); ~0 on flat walking swing and all contact."""
    T_root = var_values[var_Ts_world_root]
    lat = T_root.rotation().as_matrix()[:, 0]  # body lateral axis (root local-x) in world
    R = (
        T_root @ jaxlie.SE3(robot.forward_kinematics(cfg=var_values[var_robot_cfg]))
    ).rotation().as_matrix()
    li = threedmmx_joint_retarget_indices[foot_indices[0]]
    ri = threedmmx_joint_retarget_indices[foot_indices[1]]
    sole_l = R[li][:, 2]  # left sole normal (foot local-z) in world
    sole_r = R[ri][:, 2]
    return jnp.array([gate_l[0] * (sole_l @ lat),
                      gate_r[0] * (sole_r @ lat)]) * weight


@jdc.jit
def solve_retargeting(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision | None,
    target_keypoints: jnp.ndarray,
    target_orientations: jnp.ndarray,
    left_foot_contact: jnp.ndarray,
    right_foot_contact: jnp.ndarray,
    threedmmx_joint_retarget_indices: jnp.ndarray,
    threedmmx_retarget_mask: jnp.ndarray,
    weights: RetargetingWeights,
    subsample_factor: int = 1,
    input_fps: float = 30.0,
) -> Tuple[jaxlie.SE3, jnp.ndarray]:
    """Solve the simplified retargeting problem."""

    n_retarget = len(threedmmx_joint_retarget_indices)
    timesteps = target_keypoints.shape[0]

    # Robot properties.
    # - Joints that should move less for natural humanoid motion.
    #   With the 21-keypoint set we now have BOTH neck and head keypoints, so the
    #   neck->head column is directly constrained and no longer drifts. Rely on
    #   the keypoints (keypoints-first): no extra rest pin here. If the redundant
    #   3-segment neck starts contorting, re-add a small pin on neck*/head joints.
    joints_to_move_less = jnp.array([], dtype=jnp.int32)

    # Per-keypoint weight for the world-position match (pc_alignment_cost).
    # Prioritize the end-effectors (wrist/ankle/toe) so they track the source
    # tightly, and de-emphasize the clavicle (its ~7cm offset is a structural
    # torso-proportion difference, not worth fighting the arm chain over).
    _kp_w = {n: 1.0 for n in human_retarget_names}
    for _n in ["left_wrist", "right_wrist", "left_ankle", "right_ankle",
               "left_foot", "right_foot"]:
        _kp_w[_n] = _EE_W
    kp_pos_weights = jnp.asarray(
        [_kp_w[n] for n in human_retarget_names], dtype=jnp.float32
    )[:, None]  # (N_retarget, 1)

    # Per-timestep keypoint position weights (T, N, 1), passed to pc_alignment_cost as a
    # batched arg so jaxls vmaps it per-frame. Currently just the uniform weights
    # broadcast over time (per-frame weighting hook kept for future use).
    kp_pos_weights_t = jnp.broadcast_to(
        kp_pos_weights[None, :, :], (timesteps, n_retarget, 1)
    )

    # Compute foot indices for contact cost
    foot_indices = jnp.array(
        [
            human_retarget_names.index("left_ankle"),
            human_retarget_names.index("right_ankle"),
            human_retarget_names.index("left_foot"),
            human_retarget_names.index("right_foot"),
        ]
    )

    # Twist-rate gate (source-data CONSTANT, not a decision variable): active only when
    # the foot is airborne AND the source foot is near-vertical -- the exact degenerate
    # regime of the running swing. Verticality = downward component of the source
    # ankle->toe unit vector (structural toe droop already puts flat-foot baseline ~0.33,
    # so the 0.60 onset sits well above walking's toe-off range). Smooth in t (contacts
    # are cross-faded, verticality is smooth in the keypoints) -> introduces no new
    # discontinuity. No gradient flows through it, so the solver cannot game the gate.
    _dl3 = target_keypoints[:, 7, :] - target_keypoints[:, 5, :]  # left  ankle->toe (3D)
    _dr3 = target_keypoints[:, 8, :] - target_keypoints[:, 6, :]  # right ankle->toe (3D)
    _vert_l = -_dl3[:, 2] / (jnp.linalg.norm(_dl3, axis=-1) + 1e-8)
    _vert_r = -_dr3[:, 2] / (jnp.linalg.norm(_dr3, axis=-1) + 1e-8)

    def _smoothstep(x, lo, hi):
        s = jnp.clip((x - lo) / (hi - lo), 0.0, 1.0)
        return s * s * (3.0 - 2.0 * s)

    _air_l = 1.0 - jnp.clip(left_foot_contact.reshape(-1), 0.0, 1.0)
    _air_r = 1.0 - jnp.clip(right_foot_contact.reshape(-1), 0.0, 1.0)
    twist_gate_l = _air_l * _smoothstep(_vert_l, 0.60, 0.85)  # ~37deg..58deg down
    twist_gate_r = _air_r * _smoothstep(_vert_r, 0.60, 0.85)
    # Pair gate for the (t-1, t) factor: min of endpoints (continuous in t).
    twist_gate_l_pair = jnp.minimum(twist_gate_l[:-1], twist_gate_l[1:])
    twist_gate_r_pair = jnp.minimum(twist_gate_r[:-1], twist_gate_r[1:])

    # Variables.
    class SimplifiedJointsScaleVarThreeDmmx(
        jaxls.Var[jax.Array], default_factory=lambda: jnp.ones((n_retarget, n_retarget))
    ): ...

    var_joints = robot.joint_var_cls(jnp.arange(timesteps))
    var_Ts_world_root = jaxls.SE3Var(jnp.arange(timesteps))
    var_joints_scale = SimplifiedJointsScaleVarThreeDmmx(jnp.zeros(timesteps))

    # Initialize root position and orientation using source root at each timestep t
    root_init_se3_list = []
    for t in range(timesteps):
        root_pos_t = target_keypoints[t, 0, :]  # Root position at timestep t
        root_rot_t = target_orientations[t, 0, :, :]  # Root orientation at timestep t

        # Create SE3 transformation for this timestep
        root_se3_t = jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3.from_matrix(root_rot_t), root_pos_t
        )
        root_init_se3_list.append(root_se3_t)

    # Stack all timesteps into a single array for batch initialization
    root_init_values = jaxlie.SE3(
        jnp.stack([se3.wxyz_xyz for se3 in root_init_se3_list])
    )

    # Costs.
    costs: list[jaxls.Cost] = []

    # local bones alignment cost
    @jaxls.Cost.create_factory
    def retargeting_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarThreeDmmx,
        keypoints: jnp.ndarray,
    ) -> jax.Array:
        """Retargeting factor, with a focus on:
        - matching the relative joint/keypoint positions (vectors).
        - and matching the relative angles between the vectors.
        """
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_root = var_values[var_Ts_world_root]
        T_world_link = T_world_root @ T_root_link

        # Input keypoints are already in the right format
        target_pos = keypoints[:N_retarget, :]  # (N_retarget, 3)
        robot_pos = T_world_link.translation()[jnp.array(threedmmx_joint_retarget_indices)]

        # NxN grid of relative positions.
        delta_target = target_pos[:, None] - target_pos[None, :]
        delta_robot = robot_pos[:, None] - robot_pos[None, :]

        # Vector regularization.
        position_scale = var_values[var_joints_scale][..., None]
        residual_position_delta = (
            (delta_target - delta_robot * position_scale)
            * (1 - jnp.eye(delta_target.shape[0])[..., None])
            * threedmmx_retarget_mask[..., None]
        )

        # Vector angle regularization.
        delta_target_normalized = delta_target / jnp.linalg.norm(
            delta_target + 1e-6, axis=-1, keepdims=True
        )
        delta_robot_normalized = delta_robot / jnp.linalg.norm(
            delta_robot + 1e-6, axis=-1, keepdims=True
        )
        residual_angle_delta = 1 - (
            delta_target_normalized * delta_robot_normalized
        ).sum(axis=-1)
        residual_angle_delta = (
            residual_angle_delta
            * (1 - jnp.eye(residual_angle_delta.shape[0]))
            * threedmmx_retarget_mask
        )

        residual = (
            jnp.concatenate(
                [residual_position_delta.flatten(), residual_angle_delta.flatten()]
            )
            * weights["local_alignment"]
        )
        return residual

    @jaxls.Cost.create_factory
    def scale_regularization(
        var_values: jaxls.VarValues,
        var_joints_scale: SimplifiedJointsScaleVarThreeDmmx,
    ) -> jax.Array:
        """Regularize the scale of the retargeted joints."""
        # Close to 1.
        res_0 = (var_values[var_joints_scale] - 1.0).flatten() * 1.0
        # Symmetric.
        res_1 = (
            var_values[var_joints_scale] - var_values[var_joints_scale].T
        ).flatten() * 100.0
        # Non-negative.
        res_2 = jnp.clip(-var_values[var_joints_scale], min=0).flatten() * 100.0
        return jnp.concatenate([res_0, res_1, res_2])

    @jaxls.Cost.create_factory
    def pc_alignment_cost(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
        var_joints_scale: SimplifiedJointsScaleVarThreeDmmx,
        keypoints: jnp.ndarray,
        kp_w: jnp.ndarray,
    ) -> jax.Array:
        """Soft cost to align the target keypoints to the robot, in the world frame."""
        # position_scale = var_values[var_joints_scale][..., None]

        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        link_pos = T_world_link.translation()[threedmmx_joint_retarget_indices]

        # 21 direct keypoint correspondences (no aux points) -- match every
        # robot link position to its source keypoint in the world frame.
        return (
            (link_pos - keypoints) * kp_w
        ).flatten() * weights["global_alignment"]

    @jaxls.Cost.create_factory
    def root_smoothness(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_Ts_world_root_prev: jaxls.SE3Var,
    ) -> jax.Array:
        """Smoothness cost for the robot root pose."""
        return (
            var_values[var_Ts_world_root].inverse() @ var_values[var_Ts_world_root_prev]
        ).log().flatten() * weights["root_smoothness"]

    @jaxls.Cost.create_factory
    def foot_flat_prior(
        var_values: jaxls.VarValues,
        var_Ts_world_root: jaxls.SE3Var,
        var_robot_cfg: jaxls.Var[jnp.ndarray],
    ) -> jax.Array:
        """Weak always-on prior keeping both feet flat (z-axis near world-up),
        so they don't tilt when the contact-gated foot_tilt cost is inactive."""
        T_world_root = var_values[var_Ts_world_root]
        robot_cfg = var_values[var_robot_cfg]
        T_root_link = jaxlie.SE3(robot.forward_kinematics(cfg=robot_cfg))
        T_world_link = T_world_root @ T_root_link
        Rlink = T_world_link.rotation().as_matrix()
        lf = threedmmx_joint_retarget_indices[human_retarget_names.index("left_ankle")]
        rf = threedmmx_joint_retarget_indices[human_retarget_names.index("right_ankle")]
        up = jnp.array([0.0, 0.0, 1.0])
        res_l = Rlink[lf][:, 2] - up
        res_r = Rlink[rf][:, 2] - up
        return jnp.concatenate([res_l, res_r]) * weights["foot_flat"]

    costs = [
        # Costs that are relatively self-contained to the robot.
        retargeting_cost(
            var_Ts_world_root,
            var_joints,
            var_joints_scale,
            target_keypoints,
        ),
        scale_regularization(var_joints_scale),
        pk.costs.limit_cost(
            jax.tree.map(lambda x: x[None], robot),
            var_joints,
            100.0,
        ),
        pk.costs.smoothness_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            weights["joint_smoothness"],
        ),
        root_smoothness(
            jaxls.SE3Var(jnp.arange(1, timesteps)),
            jaxls.SE3Var(jnp.arange(0, timesteps - 1)),
        ),
        # root_upright / head_upright priors removed: the pelvis, spine, neck and
        # head are now directly constrained by keypoints (21-point set), so these
        # orientation priors are unnecessary (and would fight the source pose).
        foot_flat_prior(
            jaxls.SE3Var(jnp.arange(timesteps)),
            robot.joint_var_cls(jnp.arange(timesteps)),
        ),
        pc_alignment_cost(
            var_Ts_world_root,
            var_joints,
            var_joints_scale,
            target_keypoints,
            kp_pos_weights_t,
        ),
        pk.costs.rest_cost(
            var_joints,
            var_joints.default_factory()[None],
            jnp.full(
                var_joints.default_factory().shape, 0.02
            )  # small rest cost for all joints
            .at[joints_to_move_less]
            .set(weights["joint_rest_penalty"])[
                None
            ],  # large rest cost for joints that should move less
        ),
        joint_vel_limit_cost(
            robot.joint_var_cls(jnp.arange(1, timesteps)),
            robot.joint_var_cls(jnp.arange(0, timesteps - 1)),
            20.0,  # max velocity in rad/s
            subsample_factor / input_fps,  # dt in seconds (accounting for subsampling)
            weights["joint_vel_limit"],
        ),
        # pk.costs.self_collision_cost(
        #     jax.tree.map(lambda x: x[None], robot),
        #     jax.tree.map(lambda x: x[None], robot_coll),
        #     var_joints,
        #     margin=0.01,
        #     weight=weights["self_collision"],
        # ),
    ]

    # Add foot contact costs for each timestep (using v2) - start from t=1 since we need previous timestep
    for t in range(1, timesteps):
        costs.append(
            foot_contact_cost(
                jaxls.SE3Var(t),  # current
                jaxls.SE3Var(t - 1),  # previous
                robot.joint_var_cls(t),  # current cfg
                robot.joint_var_cls(t - 1),  # previous cfg
                robot,
                left_foot_contact[t],
                right_foot_contact[t],
                threedmmx_joint_retarget_indices,
                foot_indices,
                weights["foot_contact"],
            )
        )

    # Add foot tilt costs for each timestep - doesn't need previous timestep
    for t in range(timesteps):
        costs.append(
            foot_tilt_cost(
                jaxls.SE3Var(t),
                robot.joint_var_cls(t),
                robot,
                left_foot_contact[t],
                right_foot_contact[t],
                threedmmx_joint_retarget_indices,
                foot_indices,
                weights["foot_tilt"],  # Reusing foot_tilt weight for tilt cost
            )
        )

    if _FOOT_TWIST_RATE > 0.0:
        for t in range(1, timesteps):
            costs.append(
                foot_twist_rate_cost(
                    jaxls.SE3Var(t),  # current
                    jaxls.SE3Var(t - 1),  # previous
                    robot.joint_var_cls(t),  # current cfg
                    robot.joint_var_cls(t - 1),  # previous cfg
                    robot,
                    twist_gate_l_pair[t - 1 : t],
                    twist_gate_r_pair[t - 1 : t],
                    threedmmx_joint_retarget_indices,
                    foot_indices,
                    _FOOT_TWIST_RATE,
                )
            )

    if _FOOT_SAGITTAL > 0.0:
        for t in range(timesteps):
            costs.append(
                foot_sole_sagittal_cost(
                    jaxls.SE3Var(t),
                    robot.joint_var_cls(t),
                    robot,
                    twist_gate_l[t : t + 1],
                    twist_gate_r[t : t + 1],
                    threedmmx_joint_retarget_indices,
                    foot_indices,
                    _FOOT_SAGITTAL,
                )
            )

    solution = (
        jaxls.LeastSquaresProblem(
            costs, [var_joints, var_Ts_world_root, var_joints_scale]
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make(
                [
                    var_joints,  # Use default initialization for joints
                    var_Ts_world_root.with_value(
                        root_init_values
                    ),  # Use source root initialization
                    var_joints_scale,  # Use default initialization for joint scale
                ]
            ),
            termination=jaxls.TerminationConfig(max_iterations=800),
        )
    )

    return solution[var_Ts_world_root], solution[var_joints]


if __name__ == "__main__":
    main()
