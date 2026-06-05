from copy import deepcopy
from typing import Dict, List, Tuple, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent(asset_download_ids=["xarm6"])
class XArm6Allegro(BaseAgent):
    """XArm6 arm with the Allegro 16-DOF four-finger dexterous hand mounted on
    the flange. Uses the pre-merged ``xarm6_allegro_right.urdf`` shipped with the
    ``xarm6`` asset pack, so no URDF surgery is needed.

    Follows the same conventions as :class:`XArm7Ability`: separate arm/hand
    controller groups, a contact-based dexterous ``is_grasping`` (thumb + at
    least one other fingertip), an ``is_static`` that ignores the hand joints,
    and ``tcp``/``tcp_pose`` at the palm.
    """

    uid = "xarm6_allegro_right"
    urdf_path = f"{ASSET_DIR}/robots/xarm6/xarm6_allegro_right.urdf"
    urdf_config = dict(
        _materials=dict(
            tip=dict(static_friction=2.0, dynamic_friction=1.0, restitution=0.0)
        ),
        link={
            name: dict(material="tip", patch_radius=0.1, min_patch_radius=0.1)
            for name in [
                "link_3.0_tip",
                "link_7.0_tip",
                "link_11.0_tip",
                "link_15.0_tip",
            ]
        },
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [0, 0.22, -1.23, 0, 1.01, 0] + [0.0] * 16,
            ),
            pose=sapien.Pose(p=[0, 0, 0]),
        )
    )

    def __init__(self, *args, **kwargs):
        self.arm_joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 100

        # Allegro 16-DOF hand joints
        self.hand_joint_names = [f"joint_{i}.0" for i in range(16)]
        self.hand_stiffness = 4e2
        self.hand_damping = 1e1
        self.hand_force_limit = 5e1

        self.ee_link_name = "palm"
        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # ---------------------------------- Arm ---------------------------------- #
        arm_pd_joint_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            None,
            None,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            normalize_action=False,
        )
        arm_pd_joint_delta_pos = PDJointPosControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            use_delta=True,
        )
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            self.arm_joint_names,
            -0.1,
            0.1,
            0.1,
            self.arm_stiffness,
            self.arm_damping,
            self.arm_force_limit,
            ee_link=self.ee_link_name,
            urdf_path=self.urdf_path,
        )
        arm_pd_ee_target_delta_pose = deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        # ---------------------------------- Hand --------------------------------- #
        hand_target_delta_pos = PDJointPosControllerConfig(
            self.hand_joint_names,
            -0.1,
            0.1,
            self.hand_stiffness,
            self.hand_damping,
            self.hand_force_limit,
            use_delta=True,
        )
        hand_target_delta_pos.use_target = True

        controller_configs = dict(
            pd_joint_delta_pos=dict(
                arm=arm_pd_joint_delta_pos, gripper=hand_target_delta_pos
            ),
            pd_joint_pos=dict(arm=arm_pd_joint_pos, gripper=hand_target_delta_pos),
            pd_ee_delta_pose=dict(
                arm=arm_pd_ee_delta_pose, gripper=hand_target_delta_pos
            ),
            pd_ee_target_delta_pose=dict(
                arm=arm_pd_ee_target_delta_pose, gripper=hand_target_delta_pos
            ),
        )
        return deepcopy_dict(controller_configs)

    def _after_init(self):
        # fingertip links, thumb first so is_grasping can require the thumb
        # (Allegro order: thumb=link_15, index=link_3, middle=link_7, ring=link_11)
        finger_tip_link_names = [
            "link_15.0_tip",
            "link_3.0_tip",
            "link_7.0_tip",
            "link_11.0_tip",
        ]
        self.finger_tip_links: List[sapien.Entity] = sapien_utils.get_objs_by_names(
            self.robot.get_links(), finger_tip_link_names
        )
        # contact links used for grasp detection / contact shaping == the tips
        self.hand_front_links = self.finger_tip_links

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()

    def is_grasping(self, object: Union[Actor, None] = None, min_force: float = 0.2):
        """Grasp = thumb plus at least one other fingertip in contact with the
        object above ``min_force`` Newtons."""
        forces = torch.stack(
            [
                torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(link, object), axis=1
                )
                for link in self.finger_tip_links
            ],
            dim=0,
        )  # (4, b); index 0 == thumb
        in_contact = forces >= min_force
        thumb_contact = in_contact[0]
        num_in_contact = in_contact.sum(dim=0)
        return torch.logical_and(thumb_contact, num_in_contact >= 2)

    def is_static(self, threshold: float = 0.2):
        # exclude the 16 hand joints, only check the 6 arm joints
        qvel = self.robot.get_qvel()[..., :-16]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
