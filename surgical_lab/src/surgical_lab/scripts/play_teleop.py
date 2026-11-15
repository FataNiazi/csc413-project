#!/usr/bin/env python3
"""Lightweight teleop runner that doesn't modify mjlab's play CLI.

Run with:
  /path/to/.venv/bin/python -m surgical_lab.scripts.play_teleop \
    Surgicallab-Lift-Cube-Panda --viewer viser --device cuda

This script loads the task's play env config (with play=True), prefers a
cartesian play env when available, and runs the `Se3KeyboardController`
from `surgical_lab.tasks.lift.config.panda.teleop_se3`.
"""

import argparse
import numpy as np
import torch

from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv

from surgical_lab.tasks.lift.config.panda.teleop_se3 import Se3KeyboardController
from mjlab.viewer import ViserPlayViewer
from mjlab.rl import RslRlVecEnvWrapper


class PolicyTeleop:
    def __init__(self, controller: Se3KeyboardController, env):
        self.controller = controller
        self.env = env

    def __call__(self, obs):
        # obs is ignored; return current teleop action for each env
        twist, gripper_cmd = self.controller.get_command()
        # apply global action_scale if present on controller/env; default is 1.0
        action_scale = getattr(self, "action_scale", None)
        if action_scale is None:
            # try to read from env.cfg if present (not guaranteed)
            action_scale = 1.0
        action = np.concatenate([twist * action_scale, [gripper_cmd]])
        a = torch.from_numpy(action).float().to(self.env.device)
        # expand to batch dimension (num_envs, action_dim)
        # unsqueeze to (1, action_dim) then repeat to (num_envs, action_dim)
        a = a.unsqueeze(0).repeat(self.env.num_envs, 1)
        # no debug prints
        return a


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("task", type=str, nargs="?", default="Surgicallab-Lift-Cube-Panda")
    parser.add_argument("--viewer", type=str, default="viser", help="Viewer (native|viser)")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda:0)")
    parser.add_argument("--trans-speed", type=float, default=0.5, help="Translation speed multiplier for teleop (default 0.5)")
    parser.add_argument("--rot-speed", type=float, default=1.0, help="Rotation speed multiplier for teleop (default 1.0)")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Additional global scale applied to teleop actions before sending to env (default 1.0)")
    args = parser.parse_args()

    # Load play environment config for the chosen task
    env_cfg = load_env_cfg(args.task, play=True)

    # If the task provides a cartesian play env descriptor, prefer enabling 6D
    try:
        # Many tasks expose a cartesian env cfg factory under task package; try common name
        from surgical_lab.tasks.lift.config.panda import panda_lift_cube_cartesian_env_cfg

        if args.task.startswith("Surgicallab-Lift-Cube-Panda"):
            env_cfg = panda_lift_cube_cartesian_env_cfg(play=True)
            if "cartesian_delta" in env_cfg.actions:
                env_cfg.actions["cartesian_delta"].position_only = False
    except Exception:
        # Fall back to the task's own play env config when cartesian helper isn't available
        pass

    # Do not overwrite env_cfg.viewer object — viewer selection is handled
    # by choosing which viewer to launch below (native vs viser).

    device = args.device if args.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Creating environment...")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    # Wrap env to provide `get_observations()` expected by BaseViewer
    env = RslRlVecEnvWrapper(env, clip_actions=None)
    obs, _ = env.reset()
    print("Environment created.")

    # pass through modifier/debug options via env var or defaults if needed
    # Default: require Ctrl modifier; can be overridden with env vars for quick testing
    modifier = "none"
    debug = False
    # allow env override for quick testing
    import os
    mod_env = os.environ.get("SURGICAL_LAB_TELEOP_MODIFIER")
    dbg_env = os.environ.get("SURGICAL_LAB_TELEOP_DEBUG")
    if mod_env:
        modifier = mod_env
    if dbg_env in ("1", "true", "True"):
        debug = True

    controller = Se3KeyboardController(translation_speed=args.trans_speed, rotation_speed=args.rot_speed, modifier=modifier, debug=debug)
    # attach action_scale so PolicyTeleop can access it
    action_scale = args.action_scale if hasattr(args, "action_scale") else 1.0

    # If using the viser viewer, use ViserPlayViewer so the server is started.
    if args.viewer == "viser":
        policy = PolicyTeleop(controller, env)
        # provide action_scale to policy
        policy.action_scale = action_scale
        viewer = ViserPlayViewer(env, policy)
        try:
            viewer.run()
        except KeyboardInterrupt:
            print("\nExiting teleop...")
        finally:
            viewer.close()
    else:
        try:
            while True:
                twist, gripper_cmd = controller.get_command()
                # apply the CLI action-scale and a small smoothing multiplier
                action = np.concatenate([twist * args.action_scale, [gripper_cmd]])
                action_tensor = torch.from_numpy(action).float().to(env.device).unsqueeze(0)
                # ensure batch dim matches env.num_envs
                if hasattr(env, "num_envs") and env.num_envs is not None and env.num_envs > 1:
                    action_tensor = action_tensor.expand(env.num_envs, -1).contiguous()
                env.step(action_tensor)
                try:
                    env.render()
                except Exception:
                    pass
        except KeyboardInterrupt:
            print("\nExiting teleop...")
        finally:
            env.close()


if __name__ == "__main__":
    main()
