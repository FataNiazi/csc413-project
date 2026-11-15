#!/usr/bin/env python3
"""
Teleoperation script for the Panda robot in the lift task using Cartesian control.
"""

import argparse
import numpy as np
import torch
from pynput import keyboard
import time

from surgical_lab.tasks.lift.config.panda.env_cfg import panda_lift_cube_cartesian_env_cfg
from mjlab.envs import ManagerBasedRlEnv


class Se3KeyboardController:
    """Keyboard controller for SE(3) Cartesian control."""
    def __init__(self, translation_speed=0.5, rotation_speed=1.0, modifier: str = "none", debug: bool = False):
        self.translation_speed = translation_speed
        self.rotation_speed = rotation_speed
        self.velocity = np.zeros(6)  # vx, vy, vz, wx, wy, wz
        self.gripper_cmd = -1.0  # -1 for open, 1 for close
        self._pressed: set = set()
        self._modifiers: set = set()
        self._last_input_time = time.time()
        self._required_modifier = modifier  # 'ctrl', 'alt', 'shift', or 'none'
        self._debug = debug

        self.listener = keyboard.Listener(
            on_press=self._on_press, on_release=self._on_release
        )
        self.listener.start()

        print("\nSE3 Teleop Controls:")
        print("  W / S : +X / -X (Forward/Backward)")
        print("  A / D : +Y / -Y (Left/Right)")
        print("  Q / E : +Z / -Z (Up/Down)")
        print("  U / J : +Roll / -Roll")
        print("  I / K : +Pitch / -Pitch")
        print("  O / L : +Yaw / -Yaw")
        print("  Space : Toggle Gripper")
        print("  Z     : Open Gripper")
        print("  X     : Close Gripper")
        print("  Esc   : Quit")

    def _on_press(self, key):
        try:
            # Track modifier keys
            if key in (
                keyboard.Key.ctrl,
                keyboard.Key.ctrl_l,
                keyboard.Key.ctrl_r,
                keyboard.Key.alt,
                keyboard.Key.alt_l,
                keyboard.Key.alt_r,
                keyboard.Key.shift,
            ):
                self._modifiers.add(key)
                return

            # Check required modifier (if any)
            if self._required_modifier != "none":
                if self._required_modifier == "ctrl":
                    mod_ok = any(m in self._modifiers for m in (keyboard.Key.ctrl, keyboard.Key.ctrl_l, keyboard.Key.ctrl_r))
                elif self._required_modifier == "alt":
                    mod_ok = any(m in self._modifiers for m in (keyboard.Key.alt, keyboard.Key.alt_l, keyboard.Key.alt_r))
                elif self._required_modifier == "shift":
                    mod_ok = any(m in self._modifiers for m in (keyboard.Key.shift,))
                else:
                    mod_ok = False
                if not mod_ok:
                    return
            # track pressed character keys so multiple keys can be held
            ch = None
            try:
                ch = key.char
            except Exception:
                ch = None

            if ch is not None:
                # add to pressed set
                self._pressed.add(ch)
                self._last_input_time = time.time()

            elif key == keyboard.Key.space:
                self.gripper_cmd = -self.gripper_cmd
                print(f"Gripper command: {'Close' if self.gripper_cmd > 0 else 'Open'}")
                self._last_input_time = time.time()
            elif key.char == "z":
                # Explicit open
                self.gripper_cmd = -1.0
                print("Gripper command: Open")
                self._last_input_time = time.time()
            elif key.char == "x":
                # Explicit close
                self.gripper_cmd = 1.0
                print("Gripper command: Close")
                self._last_input_time = time.time()

        except AttributeError:
            if key == keyboard.Key.esc:
                return False
            # modifier release/press handled in on_release

    def _on_release(self, key):
        try:
            ch = key.char
            if ch in self._pressed:
                try:
                    self._pressed.remove(ch)
                except KeyError:
                    pass
        except AttributeError:
            # Track modifier release
            if key in self._modifiers:
                try:
                    self._modifiers.remove(key)
                except KeyError:
                    pass

    def get_command(self):
        # Compute velocity from currently pressed keys so multiple keys combine.
        v = np.zeros(6)
        p = self._pressed

        # translation
        if "w" in p:
            v[0] += self.translation_speed
        if "s" in p:
            v[0] -= self.translation_speed
        if "a" in p:
            v[1] += self.translation_speed
        if "d" in p:
            v[1] -= self.translation_speed
        if "q" in p:
            v[2] += self.translation_speed
        if "e" in p:
            v[2] -= self.translation_speed

        # rotation
        if "u" in p:
            v[3] += self.rotation_speed
        if "j" in p:
            v[3] -= self.rotation_speed
        if "i" in p:
            v[4] += self.rotation_speed
        if "k" in p:
            v[4] -= self.rotation_speed
        if "o" in p:
            v[5] += self.rotation_speed
        if "l" in p:
            v[5] -= self.rotation_speed

        # If no keys pressed recently, zero velocities to avoid stuck keys.
        if len(p) == 0 and (time.time() - self._last_input_time > 0.25):
            return np.zeros_like(v), self.gripper_cmd
        return v, self.gripper_cmd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, nargs="?", default=None, help="Task name")
    parser.add_argument("--viewer", type=str, default="default", help="Viewer type (e.g. viser)")
    parser.add_argument("--agent", type=str, choices=["zero", "random", "trained", "teleop"], default="teleop", help="Agent type")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Load environment config with play=True for teleop settings
    env_cfg = panda_lift_cube_cartesian_env_cfg(play=True)
    # Enable 6D control (position + rotation) for teleop
    env_cfg.actions["cartesian_delta"].position_only = False

    # Create environment
    print("Creating environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    obs, _ = env.reset()
    print("Environment created.")

    # Initialize controller
    controller = Se3KeyboardController()

    try:
        while True:
            twist, gripper_cmd = controller.get_command()
            
            # Construct action: 6D twist + 1D gripper
            # Scale twist for smoother control. Action is [dx, dy, dz, ax, ay, az, gripper]
            action = np.concatenate([twist * 0.1, [gripper_cmd]])
            action_tensor = torch.from_numpy(action).float().to(env.device).unsqueeze(0)
            env.step(action_tensor)
            env.render()

    except KeyboardInterrupt:
        print("\nExiting...")
        env.close()


if __name__ == "__main__":
    main()