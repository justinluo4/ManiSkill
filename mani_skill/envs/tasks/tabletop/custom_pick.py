from typing import Any, Dict, Union

import numpy as np
import sapien
import torch
import random

import mani_skill.envs.utils.randomization as randomization
from mani_skill.agents.robots import Fetch, Panda, XArm6Robotiq
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.geometry import rotation_conversions
from mani_skill.examples.motionplanning.panda.motionplanner import build_panda_gripper_grasp_pose_visual
from mani_skill.utils.grasping import orient_then_grasp, grasp_diff, grasp_reward
from scipy.spatial.transform import Rotation as R

@register_env("CustomPick-v1", max_episode_steps=100)
class CustomPickEnv(BaseEnv):
    """
    **Task Description:**
    A simple task where the objective is to grasp a red cube and move it to a target goal position.

    **Randomizations:**
    - the cube's xy position is randomized on top of a table in the region [0.1, 0.1] x [-0.1, -0.1]. It is placed flat on the table
    - the cube's z-axis rotation is randomized to a random angle
    - the target goal position (marked by a green sphere) of the cube has its xy position randomized in the region [0.1, 0.1] x [-0.1, -0.1] and z randomized in [0, 0.3]

    **Success Conditions:**
    - the cube position is within `goal_thresh` (default 0.025m) euclidean distance of the goal position
    - the robot is static (q velocity < 0.2)
    """
    object_name = "072-a_toy_airplane"
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    cube_half_size = 0.1
    bar_thickness = 0.007
    bar_length = 0.05
    goal_thresh = 0.025
    # Define target grasp (sample multiple if there are degrees of freedom)


    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.target_grasp = None
        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0, 0]))


    def _load_scene(self, options: dict):
        self.table_scene = TableSceneBuilder(
            self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()
        # self.cube = actors.build_cube(
        #     self.scene,
        #     half_size=self.cube_half_size,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )
        # cube_builder = self.scene.create_actor_builder()
        #
        # cube_builder.add_box_collision(pose=sapien.Pose([0.0, 0.0, 0.0]),
        #                                half_size=[self.bar_length, self.bar_thickness, self.bar_thickness])
        # cube_builder.add_box_visual(
        #     pose=sapien.Pose([0.0, 0.0, 0.0]), half_size=[self.bar_length, self.bar_thickness, self.bar_thickness],
        #     material=sapien.render.RenderMaterial(
        #         base_color=[1, 0, 0, 1],
        #     ),
        # )
        # cube_builder.add_sphere_collision(
        #     radius=self.cube_half_size, pose=sapien.Pose([self.bar_length, 0, 0]),
        # )
        # cube_builder.add_sphere_visual(
        #     radius=self.cube_half_size,
        #     pose=sapien.Pose([self.bar_length, 0, 0]),
        #     material=sapien.render.RenderMaterial(
        #         base_color=[1, 0, 0, 1],
        #     ),
        # )
        # cube_builder.add_sphere_collision(
        #     radius=self.cube_half_size, pose=sapien.Pose([-self.bar_length, 0, 0]),
        # )
        # cube_builder.add_sphere_visual(
        #     radius=self.cube_half_size,
        #     pose=sapien.Pose([-self.bar_length, 0, 0]),
        #     material=sapien.render.RenderMaterial(
        #         base_color=[1, 0, 0, 1],
        #     ),
        # )
        # cube_builder.set_initial_pose(sapien.Pose(p=[0, 0, self.cube_half_size]))
        # self.cube = cube_builder.build(name="cube")
        # self.cube = actors.build_tree(
        #     self.scene,
        #     half_thickness=self.bar_thickness,
        #     radius=self.cube_half_size,
        #     color=[1, 0, 0, 1],
        #     name="cube",
        #     initial_pose=sapien.Pose(p=[0, 0, self.cube_half_size]),
        # )
        builder = self.scene.create_actor_builder()
        builder.add_multiple_convex_collisions_from_file(
            filename="/home/justin/PycharmProjects/ManiSkill/mani_skill2_ycb/models/" + self.object_name + "/collision.ply",
            scale=[1] * 3,
            material=None,
            density=1000,
        )

        builder.add_visual_from_file(filename="/home/justin/PycharmProjects/ManiSkill/mani_skill2_ycb/models/"  + self.object_name + "/textured.obj", scale=[1] * 3)
        builder.set_initial_pose(sapien.Pose())
        self.cube = builder.build(name="cube")

        self.grasp_vis = build_panda_gripper_grasp_pose_visual(self.scene)
        self.grasp_vis.initial_pose = sapien.Pose()
        self.goal_site = actors.build_sphere(
            self.scene,
            radius=self.goal_thresh,
            color=[0, 1, 0, 1],
            name="goal_site",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):



        with torch.device(self.device):
            b = len(env_idx)

            all_grasps = np.load(
                "/home/justin/PycharmProjects/ManiSkill/mani_skill2_ycb/models/grasp_dataset_top_down.npy",
                allow_pickle=True)
            grasps = all_grasps.item().get(self.object_name)
            grasp_pos = torch.tensor([g["pos"] for g in grasps])
            grasp_quats = torch.tensor([(R.from_euler("Y", 90, degrees=True) * R.from_quat(g["quat"])).as_quat() for g in grasps])
            scores = np.array([g["score"] for g in grasps])
            scores /= scores.sum()
            selected_grasps = np.random.choice(len(grasp_pos), b, p=scores)
            grasp_pos = grasp_pos[selected_grasps]
            grasp_quats = grasp_quats[selected_grasps]

            self.table_scene.initialize(env_idx)
            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 2] = self.cube_half_size
            qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            # qs[:, [0, 2, 1, 3]] = qs[:, [2, 0, 3, 1]]
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            self.local_grasp = Pose.create_from_pq(grasp_pos,  grasp_quats)
            # self.local_grasp.p[:, 2] -= 0.06
            ax = torch.zeros((b, 3))
            ax[:, 1] += 1
            q_noise = rotation_conversions.axis_angle_to_quaternion((ax.T * (torch.rand(b)* 0.4 - 0.2)).T)
            # self.local_grasp = self.local_grasp * Pose.create_from_pq(q=q_noise)
            self.target_grasp = self.cube.pose * self.local_grasp
            # for g in grasps:
            #     grasp_vis = build_panda_gripper_grasp_pose_visual(self.scene)
            #     print(g["quat"].as_quat())
            #     grasp_vis.set_pose(self.cube.pose * Pose.create_from_pq(g["pos"], (R.from_euler("Y", 90, degrees=True) * g["quat"]).as_quat()))

            goal_xyz = torch.zeros((b, 3))
            goal_xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            goal_xyz[:, 2] = torch.rand((b)) * 0.3 + xyz[:, 2]
            self.goal_site.set_pose(Pose.create_from_pq(goal_xyz))

    def _get_obs_extra(self, info: Dict):
        # in reality some people hack is_grasped into observations by checking if the gripper can close fully or not
        obs = dict(
            is_grasped=info["is_grasped"],
            tcp_pose=self.agent.tcp.pose.raw_pose,
            goal_pos=self.goal_site.pose.p,
            orient=self.cube.pose.p
        )
        if "state" in self.obs_mode:
            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                # tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
                tcp_to_target_pos=self.target_grasp.p - self.agent.tcp.pose.p,
                target_pose=self.target_grasp.raw_pose,
            )
        return obs

    def update_grasp(self):
        # grasp_dist = grasp_diff(self.agent.tcp.pose, self.target_grasp)
        # self.local_grasp.p[grasp_dist < 0.1, 2] += 0.02
        # self.local_grasp.p[grasp_dist > 0.5, 2] -= 0.02
        # self.local_grasp.p[:, 2] = torch.clamp(self.local_grasp.p[:, 2], min = -0.08, max=0)
        self.target_grasp = self.cube.pose * self.local_grasp
        self.grasp_vis.set_pose(self.target_grasp)

    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        self.update_grasp()

        return super().step(action)

    def evaluate(self):
        is_obj_placed = (
                torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
                <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube, max_angle=20)
        is_robot_static = self.agent.is_static(0.2)
        self.grasp_vis.set_pose(self.target_grasp)
        rot_diff = torch.acos(torch.sum(self.agent.tcp.pose.q * self.target_grasp.q, dim=1)**2 * 2 - 1)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "rot_diff": rot_diff,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        reward = grasp_reward(self.agent.tcp.pose, self.target_grasp)


        # reward += (torch.tanh(diff1).clamp(min = 0) + torch.tanh(diff2).clamp(min = 0) ) * 0.5

        # reward = (reaching_reward*0.5 + diff1 + (diff1.clamp(min = 0) * reaching_reward.clamp(min = 0))**2) * 0.5
        # Add grasp pose reward

        is_grasped = info["is_grasped"]
        reward += is_grasped
        obj_to_goal_dist = torch.linalg.norm(
            self.goal_site.pose.p - self.cube.pose.p, axis=1
        )
        place_reward = 1 - torch.tanh(5 * obj_to_goal_dist)
        reward += place_reward * is_grasped * 3

        qvel_without_gripper = self.agent.robot.get_qvel()
        if self.robot_uids == "xarm6_robotiq":
            qvel_without_gripper = qvel_without_gripper[..., :-6]
        elif self.robot_uids == "panda":
            qvel_without_gripper = qvel_without_gripper[..., :-2]
        static_reward = 1 - torch.tanh(
            5 * torch.linalg.norm(qvel_without_gripper, axis=1)
        )
        reward += static_reward * info["is_obj_placed"]

        reward[info["success"]] += 5
        return reward

    def compute_normalized_dense_reward(
            self, obs: Any, action: torch.Tensor, info: Dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 5
