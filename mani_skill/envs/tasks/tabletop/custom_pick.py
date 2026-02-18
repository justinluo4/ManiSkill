from typing import Any, Dict, Union
from pathlib import Path
import numpy as np
import sapien
import torch
import random
import quaternion
from mani_skill import ASSET_DIR
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
from mani_skill.utils.grasping import orient_then_grasp, grasp_diff, grasp_reward, rotation_difference
from scipy.spatial.transform import Rotation as R
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig
import yaml
import os
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
    _sample_video_link = "https://github.com/haosulab/ManiSkill/raw/main/figures/environment_demos/PickCube-v1_rt.mp4"
    SUPPORTED_ROBOTS = [
        "panda",
        "panda_no_collision",
        "fetch",
        "xarm6_robotiq",
    ]
    agent: Union[Panda, Fetch, XArm6Robotiq]
    cube_half_size = 0.1
    bar_thickness = 0.007
    bar_length = 0.05
    goal_thresh = 0.025
    starting_offset = 0.04
    # Define target grasp (sample multiple if there are degrees of freedom)


    def __init__(self, *args, robot_uids="panda", robot_init_qpos_noise=0.02, **kwargs):
        self.asset_root = Path(f"{ASSET_DIR}/tasks/grasping/")
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.target_grasp = None
        self.use_decomp = kwargs.pop("use_decomp", True)
        self.object_name = kwargs.pop("object_name", None)
        self.guide_traj = False

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                max_rigid_contact_count=2**22,
                max_rigid_patch_count=2**20,
            )
        )

    @property
    def _default_sensor_configs(self):
        pose = sapien_utils.look_at(eye=[0.3, 0, 0.6], target=[-0.1, 0, 0.1])
        return [CameraConfig("base_camera", pose, 128, 128, np.pi / 2, 0.01, 100)]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.6, 0.7, 0.6], [0.0, 0.0, 0.35])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    def _load_agent(self, options: dict):
        super()._load_agent(options, sapien.Pose(p=[-0.615, 0.2, 0]))

    # def reset(self, seed: Union[None, int, list[int]] = None, options: Union[None, dict] = None):
    #     return super().reset(options = {"reconfigure": True})

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
        b = self.num_envs

        self.grasp_quats = []
        self.grasp_pos = []
        if self.object_name is None:

            objects = os.listdir(self.asset_root / "mani_skill2_ycb" / "models")
            objects.remove("072-d_toy_airplane")
            objs = [objects[i] for i in np.random.choice(len(objects), b)]
            collision_files = []
            original_files = []
            c = 0
            for obj in objs:
                with open(str(self.asset_root / "mani_skill2_ycb" / "models" / obj / "grasps.yaml"), 'r') as stream:

                    grasps = list(yaml.safe_load(stream)["grasps"].values())

                    g = random.choice(grasps)
                    rot = R.from_euler("X", 90, degrees=True)* R.from_quat([g["orientation"]["w"]] + g["orientation"]["xyz"])
                    # while rot.apply([0, 0, 1])[2] < 0:
                    #     g = random.choice(grasps)
                    #     rot = R.from_euler("X", 90, degrees=True) * R.from_quat(
                    #         [g["orientation"]["w"]] + g["orientation"]["xyz"])
                    self.grasp_quats.append(rot.as_quat() )
                    self.grasp_pos.append(g["position"])
                    if self.use_decomp:

                        collision_files.append(str(self.asset_root / "mani_skill2_ycb" / "models" / obj / "grasp_decomp" / f"decomp_{g["mesh_id"]}.ply"))
                    else:
                        collision_files.append(str(self.asset_root / "mani_skill2_ycb" / "models" / obj / "collision_mesh_t=0.04.ply"))
                    original_files.append(str(self.asset_root / "mani_skill2_ycb" / "models" / obj / "textured.obj"))
                c += 1

            builder = self.scene.create_decomposition_builder()
            builder.add_multiple_convex_collisions_from_multiple_files(
                files=collision_files,
                scale=[1] * 3,
                material=None,
                density=1000,
            )
            self.local_grasp = Pose.create_from_pq(torch.tensor(np.array(self.grasp_pos)),  torch.tensor(np.array(self.grasp_quats)))
            builder.add_visuals_from_files(
                files = original_files,
                scale=[1] * 3)
        else:

            with open(str(self.asset_root / "mani_skill2_ycb" / "models" / self.object_name / "grasps.yaml"), 'r') as stream:
                grasps = yaml.safe_load(stream)["grasps"].values()
            grasp_ids = torch.tensor([g["mesh_id"] for g in grasps])
            self.grasp_pos = torch.tensor(np.array([g["position"] for g in grasps]))
            self.grasp_quats = torch.tensor(np.array([(R.from_euler("X", 90, degrees=True) * R.from_quat(
                [g["orientation"]["w"]] + g["orientation"]["xyz"])).as_quat() for g in grasps]))
            scores = np.array([g["confidence"] for g in grasps])
            scores /= scores.sum()
            self.selected_grasps = np.random.choice(len(grasp_ids), b, p=scores)
            self.grasp_pos = self.grasp_pos[self.selected_grasps]
            self.grasp_quats = self.grasp_quats[self.selected_grasps]
            self.local_grasp = Pose.create_from_pq(self.grasp_pos, self.grasp_quats)
            if self.use_decomp:
                builder = self.scene.create_decomposition_builder()
                builder.auto_inertial = False
                collision_files = [str(self.asset_root / "mani_skill2_ycb" / "models" / self.object_name / "grasp_decomp" / f"decomp_{gid}.ply") for gid in grasp_ids[self.selected_grasps]]
                builder.add_multiple_convex_collisions_from_multiple_files(
                    files=collision_files,
                    scale=[1] * 3,
                    material=None,
                    density=1000,
                )
                builder.match_mass_and_inertia(str(self.asset_root / "mani_skill2_ycb" / "models" / self.object_name / "textured.obj"))


                builder.add_visuals_from_files(
                    files=[str(self.asset_root / "mani_skill2_ycb" / "models" / self.object_name / "textured.obj")] * b,
                    scale=[1] * 3) 
            else:
                builder = self.scene.create_actor_builder()
                builder.add_multiple_convex_collisions_from_file(
                    filename=str(self.asset_root / "mani_skill2_ycb" / "models" / self.object_name / "collision_mesh_t=0.04.ply"),
                    scale=[1] * 3,
                    material=None,
                )
                builder.add_visual_from_file(filename = str(self.asset_root / "mani_skill2_ycb" / "models" / self.object_name / "textured.obj"))

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
        self.cmass_marker = actors.build_sphere(
            self.scene,
            radius=0.01,
            color=[1, 0, 0, 1],
            name="cmass",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.goal_site)


    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):



        with torch.device(self.device):
            b = len(env_idx)
            self.stage = torch.zeros(b)
            # if self.object_name is not None:
            #     with open(str(self.asset_root / "mani_skill2_ycb" / "models" / self.object_name / "grasps.yaml"),
            #               'r') as stream:
            #         grasps = yaml.safe_load(stream)["grasps"].values()
            #     grasp_pos = torch.tensor([g["position"] for g in grasps])
            #     grasp_quats = torch.tensor([(R.from_euler("X", 90, degrees=True)* R.from_quat([g["orientation"]["w"]] + g["orientation"]["xyz"])).as_quat() for g in grasps])
            #     scores = np.array([g["confidence"] for g in grasps])
            #     scores /= scores.sum()
            #     grasp_pos = grasp_pos[self.selected_grasps]
            #     grasp_quats = grasp_quats[self.selected_grasps]
            #     self.local_grasp = Pose.create_from_pq(grasp_pos, grasp_quats)
            self.table_scene.initialize(env_idx)


            xyz = torch.zeros((b, 3))
            xyz[:, :2] = torch.rand((b, 2)) * 0.2 - 0.1
            xyz[:, 0] += 0.12
            reach_angle = torch.arctan2(xyz[:, 1], xyz[:, 0] + 0.5)

            xyz[:, 2] = 0.1
            # qs = randomization.random_quaternions(b, lock_x=True, lock_y=True)
            rot = quaternion.as_quat_array(self.local_grasp.q)
            v = np.quaternion(0, 0, 0, 1)
            v = rot * v * rot.conjugate()
            v = quaternion.as_float_array(v)
            q_angle = np.arctan2(v[:, 2], v[:, 1])
            q_height = -v[:, 3]
            turn_angle = -(q_angle - np.array(reach_angle.cpu()))
            qs = np.zeros((b, 4))
            qs[:, 0] = np.cos(turn_angle/2)
            qs[:, 3] = np.sin(turn_angle/2)
            qs = torch.from_numpy(qs)
            self.cube.set_pose(Pose.create_from_pq(xyz, qs))
            offsets = torch.zeros((b, 3))
            if self.guide_traj:
                offsets[:, 2] = self.starting_offset
            else:
                offsets[:, 2] = 0.1
            self.offset = Pose.create_from_pq(offsets)
            # self.local_grasp = Pose.create_from_pq(grasp_pos,  grasp_quats)
            ax = torch.zeros((b, 3))
            ax[:, 1] += 1
            q_noise = rotation_conversions.axis_angle_to_quaternion((ax.T * (torch.rand(b)* 0.4 - 0.2)).T)
            # self.local_grasp = self.local_grasp * Pose.create_from_pq(q=q_noise)

            self.target_grasp = self.cube.pose * (self.local_grasp * self.offset)

            # for g in grasps:
            #     grasp_vis = build_panda_gripper_grasp_pose_visual(self.scene)
            #     grasp_vis.set_pose(self.cube.pose * Pose.create_from_pq(g["position"], (R.from_euler("X", 90, degrees=True)* R.from_quat([g["orientation"]["w"]] + g["orientation"]["xyz"])).as_quat()))

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
            rot_diff = rotation_difference(self.agent.tcp.pose.q, self.target_grasp.q, symmetric=True)
            tcp_euler = rotation_conversions.matrix_to_euler_angles(rotation_conversions.quaternion_to_matrix(self.agent.tcp.pose.q), "XYZ")
            target_euler = rotation_conversions.matrix_to_euler_angles(rotation_conversions.quaternion_to_matrix(self.target_grasp.q), "XYZ")

            obs.update(
                obj_pose=self.cube.pose.raw_pose,
                # tcp_to_obj_pos=self.cube.pose.p - self.agent.tcp.pose.p,
                obj_to_goal_pos=self.goal_site.pose.p - self.cube.pose.p,
                tcp_to_target_pos=self.target_grasp.p - self.agent.tcp.pose.p,
                target_pose=self.target_grasp.raw_pose,
                target_euler=target_euler,
                tcp_euler=tcp_euler,
                rotation_diff=rot_diff,
            )
        return obs

    def update_grasp(self):
        if self.guide_traj:
            grasp_dist = grasp_diff(self.agent.tcp.pose, self.target_grasp)
            self.stage += 1 * (grasp_dist < 0.1)
            step = 0.01
            self.offset.p[grasp_dist < 0.1, 2] += step
            self.offset.p[grasp_dist > 0.5, 2] -= step

            self.offset.p[:, 2] = torch.clamp(self.offset.p[:, 2], min = self.starting_offset, max=0.11)
        self.target_grasp = self.cube.pose *  (self.local_grasp * self.offset)
        self.grasp_vis.set_pose(self.target_grasp)



    def step(self, action: Union[None, np.ndarray, torch.Tensor, Dict]):
        self.update_grasp()
        self.cmass_marker.set_pose(self.cube.pose * self.cube.cmass_local_pose)

        return super().step(action)

    def evaluate(self):
        is_obj_placed = (
                torch.linalg.norm(self.goal_site.pose.p - self.cube.pose.p, axis=1)
                <= self.goal_thresh
        )
        is_grasped = self.agent.is_grasping(self.cube, max_angle=20)
        is_robot_static = self.agent.is_static(0.2)
        self.grasp_vis.set_pose(self.target_grasp)
        rot_diff = rotation_difference(self.agent.tcp.pose.q, self.target_grasp.q, symmetric=True)
        return {
            "success": is_obj_placed & is_robot_static,
            "is_obj_placed": is_obj_placed,
            "is_robot_static": is_robot_static,
            "is_grasped": is_grasped,
            "rot_diff": rot_diff,
        }

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: Dict):

        reward = grasp_reward(self.agent.tcp.pose, self.target_grasp, reach_weight=1, orient_weight = 1)



        # reward += (torch.tanh(diff1).clamp(min = 0) + torch.tanh(diff2).clamp(min = 0) ) * 0.5

        # reward = (reaching_reward*0.5 + diff1 + (diff1.clamp(min = 0) * reaching_reward.clamp(min = 0))**2) * 0.5
        # Add grasp pose reward
        #reward += torch.clamp(self.stage * 0.2, min = 0, max = 1)
        is_grasped = info["is_grasped"]*2
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
