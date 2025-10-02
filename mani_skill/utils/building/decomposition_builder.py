from __future__ import annotations

import typing
from typing import TYPE_CHECKING, List, Union, Dict, Any, Tuple, Literal, Sequence
import numpy as np
import sapien
import sapien.physx as physx
import torch
from mani_skill.utils.building import ActorBuilder
from sapien.wrapper.actor_builder import preprocess_mesh_file
from mani_skill import logger
from mani_skill.utils import common
from mani_skill.utils.structs.actor import Actor
from mani_skill.utils.structs.pose import Pose, to_sapien_pose

if TYPE_CHECKING:
    from mani_skill.envs.scene import ManiSkillScene



Vec3 = Tuple
class DecompositionBuilder(ActorBuilder):
    def __init__(self):
        super().__init__()
        self.files = None
    def add_multiple_convex_collisions_from_multiple_files(
        self,
        files: Sequence[str],
        pose: sapien.Pose = sapien.Pose(),
        scale: Vec3 = (1, 1, 1),
        material: Union[sapien.physx.PhysxMaterial, None] = None,
        density: float = 1000,
        patch_radius: float = 0,
        min_patch_radius: float = 0,
        is_trigger: bool = False,
        decomposition: typing.Literal["none", "coacd"] = "none",
        decomposition_params=dict(),
        variety = False,
    ):
        self.files = files

        for file in files:
            self.add_multiple_convex_collisions_from_file(file, pose, scale, material, density, patch_radius, min_patch_radius, is_trigger, decomposition, decomposition_params)

    def add_visuals_from_files(
        self,
        files: Sequence[str],
        pose: sapien.Pose = sapien.Pose(),
        scale: Vec3 = (1, 1, 1),
        material: Union[sapien.render.RenderMaterial, None, Vec3] = None,
        name: str = "",
    ):
        if material is not None and not isinstance(
            material, sapien.render.RenderMaterial
        ):
            material = sapien.render.RenderMaterial(base_color=(*material[:3], 1))
        for file in files:
            self.add_visual_from_file(file, pose, scale, material, name)

        return self
    def match_mass_and_inertia(self, file):
        self.add_multiple_convex_collisions_from_file(file, sapien.Pose(), (1, 1, 1), None, 1000, 0,
                                                      0, False, "none", dict())
        self.base_cr = self.collision_records.pop(-1)
        component = physx.PhysxRigidDynamicComponent()
        r = self.base_cr
        shapes = physx.PhysxCollisionShapeConvexMesh.load_multiple(
            filename=r.filename,
            scale=r.scale,
            material=r.material,
        )

        for shape in shapes:
            shape.local_pose = r.pose
            shape.set_collision_groups(self.collision_groups)
            shape.set_density(r.density)
            shape.set_patch_radius(r.patch_radius)
            shape.set_min_patch_radius(r.min_patch_radius)
            component.attach(shape)
        self.set_mass_and_inertia(component.mass, component.cmass_local_pose, component.inertia)




    def build(self, name):
        """
        Build the actor with the given name.

        Different to the original SAPIEN API, a unique name is required here.
        """
        self.set_name(name)

        assert (
            self.name is not None
            and self.name != ""
            and self.name not in self.scene.actors
        ), "built actors in ManiSkill must have unique names and cannot be None or empty strings"

        if self.scene_idxs is not None:
            self.scene_idxs = common.to_tensor(
                self.scene_idxs, device=self.scene.device
            ).to(torch.int)
        else:
            self.scene_idxs = torch.arange((self.scene.num_envs), dtype=int)
        num_actors = len(self.scene_idxs)

        if self.initial_pose is None:
            logger.warn(
                f"No initial pose set for actor builder of {self.name}, setting to default pose q=[1,0,0,0], p=[0,0,0]. Not setting reasonable initial poses may slow down simulation, see https://github.com/haosulab/ManiSkill/issues/421."
            )
            self.initial_pose = Pose.create(sapien.Pose())
        else:
            self.initial_pose = Pose.create(self.initial_pose, device=self.scene.device)

        initial_pose_b = self.initial_pose.raw_pose.shape[0]
        assert initial_pose_b == 1 or initial_pose_b == num_actors
        initial_pose_np = common.to_numpy(self.initial_pose.raw_pose)
        if initial_pose_b == 1:
            initial_pose_np = initial_pose_np.repeat(num_actors, axis=0)
        if self.scene.parallel_in_single_scene:
            initial_pose_np[:, :3] += self.scene.scene_offsets_np[
                common.to_numpy(self.scene_idxs)
            ]
        entities = []
        for i, scene_idx in enumerate(self.scene_idxs):
            if self.scene.parallel_in_single_scene:
                sub_scene = self.scene.sub_scenes[0]
            else:
                sub_scene = self.scene.sub_scenes[scene_idx]
            entity = sapien.Entity()
            if self.scene.can_render():
                if self.visual_records or len(self._procedural_shapes) > 0:
                    render_component = sapien.render.RenderBodyComponent()
                    r = self.visual_records[i]

                    shape = sapien.render.RenderShapeTriangleMesh(
                        preprocess_mesh_file(r.filename), r.scale, r.material
                    )
                    if r.scale[0] * r.scale[1] * r.scale[2] < 0:
                        shape.set_front_face("clockwise")

                    shape.local_pose = r.pose
                    shape.name = r.name
                    render_component.attach(shape)

                    render_component.name = self.name
                    entity.add_component(render_component)
            if self.physx_body_type == "dynamic":
                component = physx.PhysxRigidDynamicComponent()
            elif self.physx_body_type == "kinematic":
                component = physx.PhysxRigidDynamicComponent()
                component.kinematic = True
            elif self.physx_body_type == "static":
                component = physx.PhysxRigidStaticComponent()
            elif self.physx_body_type == "link":
                component = physx.PhysxArticulationLinkComponent(None)
            else:
                raise Exception(f"invalid physx body type [{self.physx_body_type}]")
            r = self.collision_records[i]
            shapes = physx.PhysxCollisionShapeConvexMesh.load_multiple(
                filename=r.filename,
                scale=r.scale,
                material=r.material,
            )
            for shape in shapes:
                shape.local_pose = r.pose
                shape.set_collision_groups(self.collision_groups)
                shape.set_density(r.density)
                shape.set_patch_radius(r.patch_radius)
                shape.set_min_patch_radius(r.min_patch_radius)
                component.attach(shape)
            if self.visual_records:
                self.add_nonconvex_collision_from_file(self.visual_records[i].filename)
                self.base_cr = self.collision_records.pop(-1)
                mass_component = physx.PhysxRigidDynamicComponent()
                mr = self.base_cr
                mass_shape = physx.PhysxCollisionShapeTriangleMesh(
                    filename=mr.filename,
                    scale=mr.scale,
                    material=mr.material,
                )

                mass_shape.local_pose = mr.pose
                mass_shape.set_density(mr.density)
                mass_shape.set_patch_radius(mr.patch_radius)
                mass_shape.set_min_patch_radius(mr.min_patch_radius)
                mass_component.attach(mass_shape)

                component.mass = mass_component.mass
                component.cmass_local_pose = mass_component.cmass_local_pose
                component.inertia = mass_component.inertia

            # else:
            #     if hasattr(self, "_auto_inertial"):
            #         if not self._auto_inertial and self.physx_body_type != "kinematic":
            #             component.mass = self._mass
            #             component.cmass_local_pose = self._cmass_local_pose
            #             component.inertia = self._inertia
            entity.add_component(component)
            entity.name = self.name
            # prepend scene idx to entity name to indicate which sub-scene it is in
            entity.name = f"scene-{scene_idx}_{self.name}"
            # set pose before adding to scene
            entity.pose = to_sapien_pose(initial_pose_np[i])
            sub_scene.add_entity(entity)
            entities.append(entity)
        actor = Actor.create_from_entities(entities, self.scene, self.scene_idxs)

        # if it is a static body type and this is a GPU sim but we are given a single initial pose, we repeat it for the purposes of observations
        if (
            self.physx_body_type == "static"
            and initial_pose_b == 1
            and self.scene.gpu_sim_enabled
        ):
            actor.initial_pose = Pose.create(
                self.initial_pose.raw_pose.repeat(num_actors, 1)
            )
        else:
            actor.initial_pose = self.initial_pose
        self.scene.actors[self.name] = actor
        self.scene.add_to_state_dict_registry(actor)
        return actor

