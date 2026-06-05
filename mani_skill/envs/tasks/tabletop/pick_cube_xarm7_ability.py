from typing import Any, Dict

import torch

from mani_skill.agents.robots import XArm7Ability
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose


@register_env("PickCubeXArm7Ability-v1", max_episode_steps=50)
class PickCubeXArm7AbilityEnv(PickCubeEnv):
    """
    PickCube task using the XArm7 with the Ability 5-finger dexterous hand
    (``xarm7_ability``).

    This is identical to :class:`PickCubeEnv` (grasp the red cube and move it to
    the goal sphere) but is set up for the dexterous hand instead of a parallel
    gripper:

    - the default (and only supported) robot is ``xarm7_ability``
    - the static reward considers only the 7 arm joints, ignoring the 10 hand
      joints (the same convention :meth:`XArm7Ability.is_static` uses)

    Grasp detection is provided by :meth:`XArm7Ability.is_grasping` (thumb plus
    at least one other finger in contact with the cube). The numerical stability
    of the hand (armature) and its self-collision handling are configured on the
    :class:`XArm7Ability` agent itself, so this env needs no special sim config
    and trains with the standard PPO settings (e.g. ``--num_envs 512``).
    """

    SUPPORTED_ROBOTS = ["xarm7_ability"]
    agent: XArm7Ability

    # number of dexterous-hand joints at the tail of qpos/qvel; the static reward
    # excludes these so only the arm joints need to settle. Subclasses with a
    # different hand override this.
    num_hand_joints = 10

    # Goal height range, expressed as height of the goal sphere above the cube's
    # resting top face. The lower bound must exceed ``goal_thresh`` so the cube on
    # the table can never already be "placed" at the goal (otherwise ~1% of
    # episodes register success with zero manipulation and contaminate the
    # metric). The upper bound sets task difficulty: the full PickCube goal can be
    # ~0.3m up, but lifting a cube that high to a floating target and holding it
    # static is extremely hard to discover for a dexterous hand from scratch, so
    # we start with a short, learnable lift (curriculum). Raise ``goal_max_height``
    # once the policy reliably succeeds at the easier range.
    goal_min_height = 0.05
    goal_max_height = 0.10

    def __init__(self, *args, robot_uids="xarm7_ability", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        # Override the goal so it is always elevated well above the cube's start
        # height; the base PickCube samples goal z in [cube_z, cube_z + height],
        # which can place the goal directly on the cube (a free, unlearned win).
        with torch.device(self.device):
            b = len(env_idx)
            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = (
                torch.rand((b, 2)) * self.cube_spawn_half_size * 2
                - self.cube_spawn_half_size
            )
            goal_xyz[:, 0] += self.cube_spawn_center[0]
            goal_xyz[:, 1] += self.cube_spawn_center[1]
            goal_xyz[:, 2] = self.cube_half_size + self.goal_min_height + torch.rand(
                (b,)
            ) * (self.goal_max_height - self.goal_min_height)
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    # max dense reward, used to normalize. Stages: reach (1) + closure (1)
    # + contact (1) + grasp (1.5) + lift (1.5) + place (1) + static (1).
    _max_reward = 8.0

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):
        cube_pos = self.cube.pose.p

        # fingertip positions of the 5-finger hand: (b, 5, 3)
        tips = torch.stack(
            [link.pose.p for link in self.agent.finger_tip_links], dim=1
        )
        grasp_center = tips.mean(dim=1)

        # 1. reaching: drive the GRASP CENTER (fingertip centroid) to the cube,
        # not the wrist `base` link (which sits ~15cm from the fingers)
        reach_dist = torch.linalg.norm(grasp_center - cube_pos, axis=1)
        reaching_reward = 1 - torch.tanh(5 * reach_dist)
        reward = reaching_reward

        # 2. finger proximity: per-fingertip dense shaping that pulls the hand
        # down onto the cube. This is what actually teaches reaching; without it
        # the centroid-reach term alone is too weak and the hand stalls in the
        # air. On its own it plateaus at "hover near the cube" (farmable without
        # grasping), which is why stage 3 (contact) is layered on top of it.
        tip_dists = torch.linalg.norm(tips - cube_pos[:, None, :], axis=2)  # (b, 5)
        closure_reward = (1 - torch.tanh(5 * tip_dists)).mean(dim=1)
        reward += closure_reward

        # 3. contact shaping: reward actually PRESSING the fingertips onto the
        # cube (sum of fingertip contact forces), gated on the hand being close.
        # Proximity alone is farmable by hovering near the cube without grasping;
        # contact force can only be earned by touching it, giving a gradient out
        # of the hover plateau and through the reach->grasp gap.
        contact_force = torch.stack(
            [
                torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(link, self.cube), axis=1
                )
                for link in self.agent.hand_front_links
            ],
            dim=1,
        ).sum(dim=1)  # (b,)
        contact_reward = torch.tanh(contact_force / 5.0)
        reward += contact_reward * (reach_dist < 0.06)

        # 4. grasp bonus (thumb + >=1 finger in contact) -- weighted above the
        # reach/closure/contact plateau so grasping is clearly worth more than
        # hovering
        is_grasped = info["is_grasped"]
        reward += 1.5 * is_grasped

        # 5. lift: once grasped, reward raising the cube off the table. Bridges
        # grasp->place before the goal sphere is within reach.
        lifted = torch.clamp(cube_pos[:, 2] - self.cube_half_size, min=0.0)
        lift_reward = torch.clamp(5 * lifted, max=1.0)
        reward += 1.5 * lift_reward * is_grasped

        # 6. place: move the cube to the goal, gated on a grasp
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - cube_pos, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped

        # 7. static reward over the arm joints only (exclude the hand joints)
        qvel = self.agent.robot.get_qvel()[..., : -self.num_hand_joints]
        static_reward = 1 - torch.tanh(5 * torch.linalg.norm(qvel, axis=1))
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] = self._max_reward
        # guard against transient NaNs from the high-stiffness hand under large
        # actions so they don't corrupt training
        return torch.nan_to_num(reward)

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return (
            self.compute_dense_reward(obs=obs, action=action, info=info)
            / self._max_reward
        )
