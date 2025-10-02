"""A simple example for contact."""
import os
from mani_skill import ASSET_DIR
import sapien
from pathlib import Path
import numpy as np
import trimesh
from trimesh.collision import CollisionManager
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider
import shutil
import yaml
import io
def main():


    asset_root = Path(f"{ASSET_DIR}/tasks/grasping/")

    scene = sapien.Scene()

    dt = 1 / 100.0

    scene.set_timestep(dt)


    actor_builder = scene.create_actor_builder()

    actor_builder.add_box_collision(half_size=[2, 2, 0.5])

    actor_builder.add_box_visual(half_size=[2, 2, 0.5], material=sapien.render.RenderMaterial(
            base_color=[1, 0, 0, 1],
        ))

    box1 = actor_builder.build_kinematic(name="floor")

    box1.set_pose(sapien.Pose(p=[0, 0, -0.5]))

    actor_builder = scene.create_actor_builder()

    collision_filename = "collision.ply"
    object_name = "072-a_toy_airplane"

    actor_builder.add_multiple_convex_collisions_from_file(
        filename=str(asset_root / "mani_skill2_ycb" / "models" / object_name / collision_filename),
        scale=[1] * 3,
        material=None,
        density=1000,
    )

    actor_builder.add_visual_from_file(
        filename=str(asset_root / "mani_skill2_ycb" / "models" / object_name / "textured.obj"), scale=[1] * 3)
    actor_builder.set_initial_pose(sapien.Pose())
    obj = actor_builder.build(name="object")

    obj.set_pose(sapien.Pose(p=[0, 0, 0.2]))

    # ---------------------------------------------------------------------------- #

    # Check contacts

    # ---------------------------------------------------------------------------- #

    for _ in range(50):

        scene.step()

    debug = False
    if debug:
        # Add some lights so that you can observe the scene

        scene.set_ambient_light([0.5, 0.5, 0.5])

        scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])
        viewer = scene.create_viewer()  # Create a viewer (window)

        # The coordinate frame in Sapien is: x(forward), y(left), z(upward)
        # The principle axis of the camera is the x-axis
        viewer.set_camera_xyz(x=-4, y=0, z=2)
        # The rotation of the free camera is represented as [roll(x), pitch(-y), yaw(-z)]
        # The camera now looks at the origin
        viewer.set_camera_rpy(r=0, p=-np.arctan2(2, 4), y=0)
        viewer.window.set_camera_parameters(near=0.05, far=100, fovy=1)
        while not viewer.closed:  # Press key q to quit

            scene.step()  # Simulate the world

            contacts = scene.get_contacts()

            support_force = 0

            for contact in contacts:
                #
                # print(contact)

                for point in contact.points:

                    # print("Impulse (F * dt) on the first actor:", point.impulse)
                    #
                    # print("Normal (same direction as impulse):", point.normal)
                    #
                    print("Contact position (in the world frame):", point.position, contact.bodies[0].entity.name, contact.bodies[1].entity.name)
                    #
                    # print("Minimum distance between two shapes:", point.separation)

                    if contact.bodies[0].entity.name == "object":

                        support_force += point.impulse[2] / dt

                    elif contact.bodies[0].entity.name == "box1":

                        support_force -= point.impulse[2] / dt

                    else:

                        raise RuntimeError("Impossible case in this example.")

            print(support_force)

            scene.update_render()  # Update the world to the renderer

            viewer.render()

    contacts = scene.get_contacts()

    local_positions = []
    for contact in contacts:

        for point in contact.points:

            # print("Impulse (F * dt) on the first actor:", point.impulse)
            #
            # print("Normal (same direction as impulse):", point.normal)
            #
            print("Contact position (in the world frame):", point.position, contact.bodies[0].entity.name,
                  contact.bodies[1].entity.name)
            #
            # print("Minimum distance between two shapes:", point.separation)
            p = sapien.Pose(point.position)
            print(obj.pose)
            local_positions.append((p * obj.pose.inv() ).get_p())
    local_positions = np.array(local_positions)
    print(local_positions)

    mesh = trimesh.load_mesh(
        str(asset_root / "mani_skill2_ycb" / "models" / object_name / collision_filename))

    cc = mesh.split(only_watertight=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax2 = fig.add_axes([0.1, 0.85, 0.8, 0.1])
    # for obj in cc:
    #     ax.plot_trisurf(
    #         obj.vertices[:, 0],
    #         obj.vertices[:, 1],
    #         obj.vertices[:, 2],
    #         triangles=obj.faces,
    #     )
    # ax.set_xlim([-0.2, 0.2])
    # ax.set_ylim([-0.2, 0.2])
    # ax.set_zlim([-0.2, 0.2])
    #
    # plt.show()

    manager = CollisionManager()
    for n, obj in enumerate(cc):
        manager.add_object(str(n), obj)
    adj = []
    for n in range(len(cc)):
        _, obj_adj = manager.in_collision_single(cc[n], return_names=True)
        adj.append([int(x) for x in obj_adj if int(x) != n])

    id = 8
    print(adj)
    dist_array = np.zeros((len(cc), len(local_positions)))
    for n, m in enumerate(cc):
        proximity_query = trimesh.proximity.ProximityQuery(m)
        closest_points, distances, triangle_ids = proximity_query.on_surface(local_positions)
        dist_array[n,:] = distances

    closest_meshes = dist_array.argmin(axis=0)



    ax.set_xlim([-0.2, 0.2])
    ax.set_ylim([-0.2, 0.2])
    ax.set_zlim([-0.2, 0.2])


    # for obj in range(len(local_positions)):
    #     ax.plot_trisurf(
    #         cc[closest_meshes[obj]].vertices[:, 0],
    #         cc[closest_meshes[obj]].vertices[:, 1],
    #         cc[closest_meshes[obj]].vertices[:, 2],
    #         triangles=cc[closest_meshes[obj]].faces,
    #     )
    #     ax.scatter(local_positions[obj, 0], local_positions[obj, 1], local_positions[obj, 2])
    #     plt.show()


    amp_slider = Slider(
        ax=ax2,
        label='depth',
        valmin=1,
        valmax=20,
        valinit=1,
        valstep=1,
    )


    def get_connected(depth):
        objs = []
        q = list(np.unique(closest_meshes))
        if id not in q:
            q.append(id)
        seen = [_ in q for _ in range(len(cc))]
        for i in range(depth):
            qnext = []
            for o in q:
                objs.append(o)
                for a in adj[o]:
                    if not seen[a]:
                        qnext.append(a)
                        seen[a] = True
            q = qnext
        print(objs)
        return objs
    # Define the update function (callback)
    def update(val):
        ax.clear()
        # val = int(val)
        # ax.plot_trisurf(
        #     cc[closest_meshes[val]].vertices[:, 0],
        #     cc[closest_meshes[val]].vertices[:, 1],
        #     cc[closest_meshes[val]].vertices[:, 2],
        #     triangles=cc[closest_meshes[val]].faces,
        # )
        # ax.scatter(local_positions[val, 0], local_positions[val, 1], local_positions[val, 2])
        depth = int(val)
        objs = get_connected(depth)

        for obj in objs:
            ax.plot_trisurf(
                cc[obj].vertices[:, 0],
                cc[obj].vertices[:, 1],
                cc[obj].vertices[:, 2],
                triangles=cc[obj].faces,
            )
        ax.scatter(local_positions[:, 0], local_positions[:, 1], local_positions[:, 2])
        ax.set_xlim([-0.2, 0.2])
        ax.set_ylim([-0.2, 0.2])
        ax.set_zlim([-0.2, 0.2])

    # Register the update function as the callback
    amp_slider.on_changed(update)

    used_meshes = get_connected(2)

    final = trimesh.util.concatenate([cc[i] for i in used_meshes])
    result = trimesh.exchange.ply.export_ply(final)
    output_file = open("export.ply", "wb+")
    output_file.write(result)
    output_file.close()

    plt.show()

def calc_contact_points(mesh_file):
    scene = sapien.Scene()

    dt = 1 / 100.0

    scene.set_timestep(dt)

    actor_builder = scene.create_actor_builder()

    actor_builder.add_box_collision(half_size=[2, 2, 0.5])

    box1 = actor_builder.build_kinematic(name="floor")

    box1.set_pose(sapien.Pose(p=[0, 0, -0.5]))

    actor_builder = scene.create_actor_builder()


    actor_builder.add_multiple_convex_collisions_from_file(
        filename=mesh_file,
        scale=[1] * 3,
        material=None,
        density=1000,
    )

    actor_builder.set_initial_pose(sapien.Pose(p=[0, 0, 0.2]))
    obj = actor_builder.build(name="object")


    # ---------------------------------------------------------------------------- #

    # Check contacts

    # ---------------------------------------------------------------------------- #

    for _ in range(50):
        scene.step()


    contacts = scene.get_contacts()

    local_positions = []
    for contact in contacts:

        for point in contact.points:

            p = sapien.Pose(point.position)
            local_positions.append((p * obj.pose.inv()).get_p())


    local_positions = np.array(local_positions)
    return local_positions


def get_contact_mesh_ids(mesh_file):
    points = calc_contact_points(mesh_file)
    print(len(points))
    mesh = trimesh.load_mesh(mesh_file)

    cc = mesh.split(only_watertight=True)

    dist_array = np.zeros((len(cc), len(points)))
    for n, m in enumerate(cc):
        proximity_query = trimesh.proximity.ProximityQuery(m)
        closest_points, distances, triangle_ids = proximity_query.on_surface(points)
        dist_array[n,:] = distances

    closest_meshes = dist_array.argmin(axis=0)
    return closest_meshes


def generate_meshes(mesh_file, depth = 1):
    print(mesh_file)
    grasp_file = os.path.join(os.path.dirname(mesh_file), "grasps.yaml")
    with open(grasp_file, 'r') as stream:
        grasps = yaml.safe_load(stream)
    save_dir = os.path.join(os.path.dirname(mesh_file), "grasp_decomp")
    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)
    os.makedirs(save_dir)

    mesh = trimesh.load_mesh(mesh_file)
    cc = mesh.split(only_watertight=True)

    manager = CollisionManager()
    for n, obj in enumerate(cc):
        manager.add_object(str(n), obj)
    adj = []
    for n in range(len(cc)):
        m = cc[n]
        centroid = m.centroid
        m.apply_translation(-centroid)
        m.apply_scale(1.05)
        m.apply_translation(centroid)
        _, obj_adj = manager.in_collision_single(m, return_names=True)
        m.apply_translation(-centroid)
        m.apply_scale(1/1.05)
        m.apply_translation(centroid)
        adj.append([int(x) for x in obj_adj if int(x) != n])
    closest_meshes = get_contact_mesh_ids(mesh_file)

    positions = np.array([grasps["grasps"][grasp]["position"] for grasp in grasps["grasps"]])
    dist_array = np.zeros((len(cc), len(grasps["grasps"])))
    for n, m in enumerate(cc):
        proximity_query = trimesh.proximity.ProximityQuery(m)
        closest_points, distances, triangle_ids = proximity_query.on_surface(positions)
        dist_array[n,:] = distances

    grasp_meshes = dist_array.argmin(axis=0)
    for n, g in enumerate(grasps["grasps"]):
        grasps["grasps"][g]["mesh_id"] = int(grasp_meshes[n])
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for obj in cc:
    #     ax.plot_trisurf(
    #         obj.vertices[:, 0],
    #         obj.vertices[:, 1],
    #         obj.vertices[:, 2],
    #         triangles=obj.faces,
    #     )
    #
    # ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2])
    # plt.show()


    for g_id in range(len(cc)):
        objs = []
        q = [g_id]
        seen = [_ in q for _ in range(len(cc))]
        for i in range(depth):
            qnext = []
            for o in q:
                objs.append(o)
                for a in adj[o]:
                    if not seen[a]:
                        qnext.append(a)
                        seen[a] = True
            q = qnext
        objs = set(objs) | set(closest_meshes)
        final = trimesh.util.concatenate([cc[i] for i in objs])
        result = trimesh.exchange.ply.export_ply(final)
        output_file = open(os.path.join(save_dir, f"decomp_{g_id}.ply"), "wb+")
        output_file.write(result)
        output_file.close()

    with open(grasp_file, 'w', encoding='utf8') as outfile:
        yaml.dump(grasps, outfile, default_flow_style=False, allow_unicode=True)






if __name__ == "__main__":

    root = "/home/jluo/.maniskill/data/tasks/grasping/mani_skill2_ycb/models"
    dirs = sorted([os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])

    for name in dirs:
        #os.chdir(os.path.join(root, name))
        #os.system("pwd")
        #os.system("coacd -i textured.obj -t 0.04 -o collision_mesh_t=0.04.ply")
        generate_meshes(f"{name}/collision_mesh_t=0.04.ply", depth = 3)
