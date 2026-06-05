from copy import deepcopy
from typing import Dict, Tuple, Union

import numpy as np
import sapien
import sapien.physx as physx
import torch

from mani_skill import PACKAGE_ASSET_DIR
from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.utils import sapien_utils
from mani_skill.utils.structs.actor import Actor


@register_agent()
class XArm7Ability(BaseAgent):
    uid = "xarm7_ability"
    urdf_path = f"{PACKAGE_ASSET_DIR}/robots/xarm7/xarm7_ability_right_hand.urdf"
    # The 5-finger Ability hand's collision meshes overlap at rest; without this
    # the fingers self-penetrate and PhysX generates explosive contact forces
    # (qpos/qvel blow up to NaN within a few steps) plus a huge number of contact
    # pairs that overflows GPU memory. All other dexterous hands in ManiSkill
    # (inspire_hand, h1_dextrous_hand) disable self-collisions for this reason.
    disable_self_collisions = True
    urdf_config = dict(
        _materials=dict(
            front_finger=dict(
                static_friction=2.0, dynamic_friction=1.5, restitution=0.0
            )
        ),
        link=dict(
            thumnb_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            index_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            middle_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            ring_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
            pinky_L2=dict(
                material="front_finger", patch_radius=0.05, min_patch_radius=0.04
            ),
        ),
    )

    keyframes = dict(
        rest=Keyframe(
            qpos=np.array(
                [
                    0.0,
                    -0.4,
                    0.0,
                    0.5,
                    0.0,
                    0.9,
                    -3.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                ]
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
            "joint7",
        ]
        self.arm_stiffness = 1e3
        self.arm_damping = 1e2
        self.arm_force_limit = 500

        self.hand_joint_names = [
            "thumb_q1",
            "index_q1",
            "middle_q1",
            "ring_q1",
            "pinky_q1",
            "thumb_q2",
            "index_q2",
            "middle_q2",
            "ring_q2",
            "pinky_q2",
        ]
        self.hand_stiffness = 1e3
        self.hand_damping = 1e2
        self.hand_force_limit = 50

        self.ee_link_name = "base"

        super().__init__(*args, **kwargs)

    @property
    def _controller_configs(self):
        # -------------------------------------------------------------------------- #
        # Arm
        # -------------------------------------------------------------------------- #
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
        arm_pd_joint_target_delta_pos = deepcopy(arm_pd_joint_delta_pos)
        arm_pd_joint_target_delta_pos.use_target = True

        # PD ee position
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

        # -------------------------------------------------------------------------- #
        # Hand
        # -------------------------------------------------------------------------- #
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

        # Make a deepcopy in case users modify any config
        return deepcopy_dict(controller_configs)

    # Joint armature for the 10 hand joints. The Ability hand's finger links have
    # very small (and partly malformed, e.g. thumb_L2 with ixx==ixy) inertia
    # tensors, which makes the joint-space mass matrix nearly singular: with the
    # high-stiffness PD drive the finger qpos/qvel diverge to NaN within a couple
    # of steps even when holding the rest pose. Adding armature regularizes the
    # mass matrix and stabilizes the sim. Increasing sim_freq does not help, which
    # confirms the problem is the mass matrix rather than the integration step.
    hand_armature = 0.1

    def _after_init(self):
        # regularize the unstable, low-inertia hand joints (see hand_armature)
        hand_joints = sapien_utils.get_objs_by_names(
            self.robot.get_active_joints(), self.hand_joint_names
        )
        armature = np.array([self.hand_armature], dtype=np.float32)
        for joint in hand_joints:
            for obj in joint._objs:
                obj.set_armature(armature)

        hand_front_link_names = [
            "thumb_L2",
            "index_L2",
            "middle_L2",
            "ring_L2",
            "pinky_L2",
        ]
        self.hand_front_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), hand_front_link_names
        )

        finger_tip_link_names = [
            "thumb_tip",
            "index_tip",
            "middle_tip",
            "ring_tip",
            "pinky_tip",
        ]
        self.finger_tip_links = sapien_utils.get_objs_by_names(
            self.robot.get_links(), finger_tip_link_names
        )

        self.tcp = sapien_utils.get_obj_by_name(
            self.robot.get_links(), self.ee_link_name
        )

        self.queries: Dict[str, Tuple[physx.PhysxGpuContactQuery, Tuple[int]]] = dict()

    def is_grasping(self, object: Union[Actor, None] = None, min_force: float = 0.2):
        """Check if the dexterous hand is grasping an object.

        Considers the hand to be grasping when the thumb and at least one other
        finger are each in contact with the object above ``min_force`` Newtons.

        Args:
            object (Actor | None): the object to check against. If None, checks
                whether any object exerts contact force on the fingers.
            min_force (float): minimum contact force (N) for a finger to count.
        """
        # thumb is the first entry in hand_front_link_names, ordered
        # [thumb, index, middle, ring, pinky]
        forces = torch.stack(
            [
                torch.linalg.norm(
                    self.scene.get_pairwise_contact_forces(link, object), axis=1
                )
                for link in self.hand_front_links
            ],
            dim=0,
        )  # (5, b)
        in_contact = forces >= min_force
        thumb_contact = in_contact[0]
        num_in_contact = in_contact.sum(dim=0)
        return torch.logical_and(thumb_contact, num_in_contact >= 2)

    def is_static(self, threshold: float = 0.2):
        # exclude the 10 hand joints, only check the 7 arm joints
        qvel = self.robot.get_qvel()[..., :-10]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold

    @property
    def tcp_pos(self):
        return self.tcp.pose.p

    @property
    def tcp_pose(self):
        return self.tcp.pose
