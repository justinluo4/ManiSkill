from mani_skill.agents.robots import XArm6Allegro
from mani_skill.envs.tasks.tabletop.pick_cube_xarm7_ability import (
    PickCubeXArm7AbilityEnv,
)
from mani_skill.utils.registration import register_env


@register_env("PickCubeXArm6Allegro-v1", max_episode_steps=50)
class PickCubeXArm6AllegroEnv(PickCubeXArm7AbilityEnv):
    """
    PickCube with the XArm6 + Allegro 16-DOF four-finger dexterous hand
    (``xarm6_allegro_right``).

    Reuses the staged dense reward and elevated-goal initialization from
    :class:`PickCubeXArm7AbilityEnv` (reach to the fingertip centroid, finger
    proximity, contact-force shaping, grasp, lift, place, static). Only the robot
    and the hand-joint count differ; the reward reads the fingertip/contact links
    generically off the agent, so it works unchanged for the 4-finger Allegro.
    """

    SUPPORTED_ROBOTS = ["xarm6_allegro_right"]
    agent: XArm6Allegro

    # Allegro has 16 hand joints (vs the Ability hand's 10)
    num_hand_joints = 16

    def __init__(self, *args, robot_uids="xarm6_allegro_right", **kwargs):
        super().__init__(*args, robot_uids=robot_uids, **kwargs)
