from typing import Any, Dict, Union

import torch

from mani_skill.agents.robots import XArm7Ability
from mani_skill.envs.tasks.tabletop.stack_cube import StackCubeEnv
from mani_skill.utils.registration import register_env


@register_env("StackCubeXArm7Ability-v1", max_episode_steps=50)
class StackCubeXArm7AbilityEnv(StackCubeEnv):
    """
    StackCube task using the XArm7 with the Ability 5-finger dexterous hand
    (``xarm7_ability``).

    This is identical to :class:`StackCubeEnv` (stack the red cube on the green
    cube and let go) but is set up for the dexterous hand instead of a parallel
    gripper:

    - the default (and only supported) robot is ``xarm7_ability``
    - the ungrasp reward is computed from how open the 10 hand joints are rather
      than from the hard-coded 2-finger Panda gripper width

    Grasp detection is provided by :meth:`XArm7Ability.is_grasping` (thumb plus
    at least one other finger in contact with the cube).
    """

    SUPPORTED_ROBOTS = ["xarm7_ability"]
    agent: Union[XArm7Ability]

    def __init__(self, *args, robot_uids="xarm7_ability", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        # reaching reward
        tcp_pose = self.agent.tcp.pose.p
        cubeA_pos = self.cubeA.pose.p
        cubeA_to_tcp_dist = torch.linalg.norm(tcp_pose - cubeA_pos, axis=1)
        reward = 2 * (1 - torch.tanh(5 * cubeA_to_tcp_dist))

        # grasp and place reward
        cubeA_pos = self.cubeA.pose.p
        cubeB_pos = self.cubeB.pose.p
        goal_xyz = torch.hstack(
            [cubeB_pos[:, 0:2], (cubeB_pos[:, 2] + self.cube_half_size[2] * 2)[:, None]]
        )
        cubeA_to_goal_dist = torch.linalg.norm(goal_xyz - cubeA_pos, axis=1)
        place_reward = 1 - torch.tanh(5.0 * cubeA_to_goal_dist)

        reward[info["is_cubeA_grasped"]] = (4 + place_reward)[info["is_cubeA_grasped"]]

        # ungrasp and static reward
        # XArm7Ability has a 10-DOF dexterous hand (the last 10 qpos entries).
        # Reward for opening the hand once the cube is placed, normalized by the
        # hand joints' range so it is independent of joint sign conventions.
        is_cubeA_grasped = info["is_cubeA_grasped"]
        hand_qpos = self.agent.robot.get_qpos()[:, -10:]
        hand_qlimits = self.agent.robot.get_qlimits()[0, -10:].to(self.device)
        hand_range = (hand_qlimits[:, 1] - hand_qlimits[:, 0]).clamp(min=1e-6)
        # closure: 0 when fully open, 1 when fully closed
        closure = ((hand_qpos - hand_qlimits[:, 0]) / hand_range).mean(dim=1)
        ungrasp_reward = torch.clamp(1 - closure, min=0.0, max=1.0)
        ungrasp_reward[~is_cubeA_grasped] = 1.0

        v = torch.linalg.norm(self.cubeA.linear_velocity, axis=1)
        av = torch.linalg.norm(self.cubeA.angular_velocity, axis=1)
        static_reward = 1 - torch.tanh(v * 10 + av)
        reward[info["is_cubeA_on_cubeB"]] = (
            6 + (ungrasp_reward + static_reward) / 2.0
        )[info["is_cubeA_on_cubeB"]]

        reward[info["success"]] = 8

        # the 10-DOF hand with high stiffness can momentarily destabilize the
        # sim under large actions; guard the reward so NaNs don't corrupt training
        return torch.nan_to_num(reward)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 8
