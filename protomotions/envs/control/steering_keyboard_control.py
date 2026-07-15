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
"""Real-time keyboard steering for the AMP steering policy (line A, inference).

The steering task is already driven by three signals — tar_dir (movement
direction), tar_face_dir (body heading) and tar_speed (speed). Normally they are
sampled randomly/periodically. This component instead sets them from live
keyboard input, so a user drives the trained steering policy directly.

Key bindings (held keys, integrated per step):
    Arrow keys  : world-frame MOVEMENT direction (tar_dir)
    A / D       : rotate body HEADING left / right (tar_face_dir)
    W / S       : increase / decrease SPEED (tar_speed)

Needs headed mode. Reuses _KeyboardSteer from masked_mimic_keyboard_control.
"""

import math
from dataclasses import dataclass

from protomotions.envs.control.steering_control import (
    SteeringControl,
    SteeringControlConfig,
)
from protomotions.envs.control.masked_mimic_keyboard_control import _KeyboardSteer


@dataclass
class SteeringKeyboardControlConfig(SteeringControlConfig):
    """Config for keyboard-driven steering (inference only)."""

    _target_: str = (
        "protomotions.envs.control.steering_keyboard_control.SteeringKeyboardControl"
    )
    turn_rate: float = 1.5  # rad/s for A/D heading rotation
    accel: float = 1.5  # m/s^2 for W/S speed change
    max_speed: float = 2.0  # speed clamp (m/s)


class SteeringKeyboardControl(SteeringControl):
    """Steering control whose targets come from the keyboard, not random sampling."""

    def __init__(self, config, env):
        super().__init__(config, env)
        self._kb = _KeyboardSteer(
            turn_rate=config.turn_rate,
            accel=config.accel,
            max_speed=config.max_speed,
        )

    def step(self):
        # Keep the root-position double buffer up to date (reward/obs need prev/curr).
        self._prev_root_pos[:] = self._curr_root_pos
        self._curr_root_pos[:] = self.env.simulator.get_root_state().root_pos

        # Read keyboard and overwrite the steering targets (no random reset).
        self._kb.update(self.env.dt)
        mt = self._kb.move_theta
        ft = self._kb.facing_theta
        sp = self._kb.speed

        self._tar_dir_theta[:] = mt
        self._tar_dir[:, 0] = math.cos(mt)
        self._tar_dir[:, 1] = math.sin(mt)
        self._tar_face_dir[:, 0] = math.cos(ft)
        self._tar_face_dir[:, 1] = math.sin(ft)
        self._tar_speed[:] = sp

        # Live terminal HUD of the current steering command (single refreshing line).
        move_deg = math.degrees(mt) % 360.0
        face_deg = math.degrees(ft) % 360.0
        print(
            f"\r[steer] speed={sp:4.2f} m/s | move_dir(RED)={move_deg:5.1f}deg | "
            f"heading(BLUE)={face_deg:5.1f}deg    ",
            end="",
            flush=True,
        )
