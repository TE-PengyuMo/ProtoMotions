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
from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import IsaacGymSimParams
from protomotions.simulator.isaaclab.config import IsaacLabSimParams
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field


@dataclass
class ThreeDmmxRobotConfig(RobotConfig):
    """
    3dmmx humanoid (26 bodies, 3-DOF hinge joints, primitive geoms).

    Target morphology for retargeting Sergey FBX mocap onto. The asset lives
    under data/Sony/ (single source of truth); geoms are primitives, so no
    external mesh files are needed.

    See data/Sony/3dmmx_neutral_no_fingers_eyes_light.xml for definition.
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["LeftFoot", "LeftToeBase"],
            "all_right_foot_bodies": ["RightFoot", "RightToeBase"],
            "all_left_hand_bodies": ["LeftHand"],
            "all_right_hand_bodies": ["RightHand"],
            "head_body_name": ["head"],
            "torso_body_name": ["chest2"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "chest2",
            "head",
            "LeftFoot",
            "RightFoot",
            "LeftHand",
            "RightHand",
        ]
    )

    default_root_height: float = 0.90

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_root="data/Sony",
            asset_file_name="3dmmx_neutral_no_fingers_eyes_light.xml",
            # IsaacLab uses the USD (converted from the MJCF via usd_convert/).
            usd_asset_file_name="usd_3dmmx/threedmmx_flat.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/hip/",
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={
                "(spine|spine1|spine2|chest|chest2|neck|neck1|neck2|head).*": ControlInfo(
                    effort_limit=300, velocity_limit=100
                ),
                ".*(Shoulder|Arm|ForeArm|Hand)_.*": ControlInfo(
                    effort_limit=150, velocity_limit=100
                ),
                ".*(UpLeg|Leg|Foot|ToeBase)_.*": ControlInfo(
                    effort_limit=300, velocity_limit=100
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            isaaclab=IsaacLabSimParams(
                fps=120,
                decimation=4,
            ),
            genesis=GenesisSimParams(
                fps=60,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=120,
                decimation=4,
            ),
        )
    )
