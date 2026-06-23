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
"""Real-time keyboard steering for a trained MaskedMimic policy (Stage 3).

At inference (headed IsaacLab only), a user drives a single character via the
keyboard. Instead of reading targets from the motion library, this component
synthesizes a virtual ROOT goal each step and injects it into the MaskedMimic
target, conditioning ONLY on the root — the transformer prior + decoder then
generate the full-body motion that follows it.

Key bindings (held keys, integrated per step):
    Arrow keys (UP/DOWN/LEFT/RIGHT) : world-frame MOVEMENT direction
                                      UP=+X, DOWN=-X, LEFT=+Y, RIGHT=-Y
    A / D                           : rotate body HEADING (facing) left / right
    W / S                           : increase / decrease SPEED magnitude

Movement direction, heading and speed are decoupled (strafing / backpedaling
while facing elsewhere is possible), matching the steering task's three signals.

NOTE: the keyboard subscription needs a headed IsaacLab app (run inference with
--headless False). In headless mode the controller stays at defaults (speed 0,
i.e. standing) and the policy just tracks a stationary root goal. The carb
keyboard wiring must be validated in a headed run; the injection math below is
sim-independent.
"""

import math
from dataclasses import dataclass

import torch

from protomotions.envs.control.masked_mimic_control import (
    MaskedMimicControl,
    MaskedMimicControlConfig,
    FixedBodyCondition,
)


@dataclass
class MaskedMimicKeyboardControlConfig(MaskedMimicControlConfig):
    """Config for keyboard-driven MaskedMimic steering (inference only)."""

    _target_: str = (
        "protomotions.envs.control.masked_mimic_keyboard_control.MaskedMimicKeyboardControl"
    )

    # Keyboard dynamics
    turn_rate: float = 1.5  # rad/s for A/D heading rotation
    accel: float = 1.5  # m/s^2 for W/S speed change
    max_speed: float = 2.0  # speed clamp (m/s)
    root_body_name: str = "Pelvis"  # body to condition on (the root)


class _KeyboardSteer:
    """Subscribes to the carb keyboard and tracks held keys.

    Maintains persistent (move_theta, facing_theta, speed). Arrow keys set the
    world-frame movement direction; A/D integrate heading; W/S integrate speed.
    Falls back to inert defaults if no headed app / carb input is available.
    """

    def __init__(self, turn_rate: float, accel: float, max_speed: float):
        self.turn_rate = turn_rate
        self.accel = accel
        self.max_speed = max_speed

        self.move_theta = 0.0  # world-frame movement direction (rad)
        self.facing_theta = 0.0  # body heading (rad)
        self.speed = 0.0  # speed magnitude (m/s)

        self._active = set()
        self._carb = None
        self._sub = None
        self._setup()

    def _setup(self):
        try:
            import carb
            import omni.appwindow

            self._carb = carb
            win = omni.appwindow.get_default_app_window()
            self._input = carb.input.acquire_input_interface()
            self._kbd = win.get_keyboard()
            self._sub = self._input.subscribe_to_keyboard_events(
                self._kbd, self._on_event
            )
            print(
                "[KeyboardSteer] subscribed to keyboard. "
                "Arrows=move dir, A/D=heading, W/S=speed."
            )
        except Exception as e:  # headless / no app / no carb
            print(
                f"[KeyboardSteer] keyboard unavailable ({e}); "
                "steering stays at defaults (standing)."
            )

    def _on_event(self, event, *args):
        carb = self._carb
        et = event.type
        if et in (
            carb.input.KeyboardEventType.KEY_PRESS,
            carb.input.KeyboardEventType.KEY_REPEAT,
        ):
            self._active.add(event.input)
        elif et == carb.input.KeyboardEventType.KEY_RELEASE:
            self._active.discard(event.input)
        return True

    def update(self, dt: float):
        """Integrate held keys into move/facing/speed. No-op without keyboard."""
        if self._carb is None:
            return
        K = self._carb.input.KeyboardInput
        a = self._active

        # Heading: A/D
        if K.A in a:
            self.facing_theta += self.turn_rate * dt
        if K.D in a:
            self.facing_theta -= self.turn_rate * dt

        # Speed magnitude: W/S
        if K.W in a:
            self.speed = min(self.max_speed, self.speed + self.accel * dt)
        if K.S in a:
            self.speed = max(0.0, self.speed - self.accel * dt)

        # Movement direction: arrow keys -> world-frame vector (held = update)
        mx = (1.0 if K.UP in a else 0.0) - (1.0 if K.DOWN in a else 0.0)
        my = (1.0 if K.LEFT in a else 0.0) - (1.0 if K.RIGHT in a else 0.0)
        if mx != 0.0 or my != 0.0:
            self.move_theta = math.atan2(my, mx)


class MaskedMimicKeyboardControl(MaskedMimicControl):
    """MaskedMimic control that overrides the root target with keyboard input."""

    def __init__(self, config, env):
        super().__init__(config, env)
        # Force root-only conditioning regardless of what config carried over.
        self.config.fixed_conditioning = [
            FixedBodyCondition(body_name=config.root_body_name, constraint_state=1)
        ]
        self.config.visible_target_pose_prob = 1.0
        self.config.repeat_mask_probability = 1.0

        self._kb = _KeyboardSteer(
            turn_rate=config.turn_rate,
            accel=config.accel,
            max_speed=config.max_speed,
        )
        self._root_idx = self.env.robot_config.kinematic_info.body_names.index(
            config.root_body_name
        )

    def populate_context(self, ctx) -> None:
        # Build the normal masked-mimic context (mask is root-only via fixed_conditioning).
        super().populate_context(ctx)

        # Advance keyboard state.
        self._kb.update(self.env.dt)

        device = self.env.device
        # Current root position [num_envs, 3].
        cur_root = self.env.simulator.get_root_state().root_pos
        cur_xy = cur_root[:, :2]
        cur_z = cur_root[:, 2:3]

        # World-frame velocity from (move_theta, speed).
        vx = math.cos(self._kb.move_theta) * self._kb.speed
        vy = math.sin(self._kb.move_theta) * self._kb.speed
        vel_xy = torch.tensor([vx, vy], device=device, dtype=cur_xy.dtype)

        # Future root xy goal at each conditioned time offset: cur + vel * t.
        time_offsets = ctx.masked_mimic.time_offsets  # [num_envs, num_future_steps]
        fut_xy = (
            cur_xy[:, None, :] + vel_xy[None, None, :] * time_offsets[..., None]
        )  # [num_envs, num_future_steps, 2]

        # Heading quaternion (xyzw) about world Z.
        h = self._kb.facing_theta
        face_q = torch.tensor(
            [0.0, 0.0, math.sin(h / 2.0), math.cos(h / 2.0)],
            device=device,
            dtype=ctx.masked_mimic.ref_rot.dtype,
        )

        # Overwrite ONLY the root body's future target (pos xy + z, rot).
        r = self._root_idx
        ctx.masked_mimic.ref_pos[:, :, r, :2] = fut_xy
        ctx.masked_mimic.ref_pos[:, :, r, 2] = cur_z  # keep current height
        ctx.masked_mimic.ref_rot[:, :, r, :] = face_q
