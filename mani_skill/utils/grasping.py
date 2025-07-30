from mani_skill.utils.geometry import geometry
import torch
from mani_skill.utils.structs.pose import Pose
import trimesh
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
def grasp_diff(cur_pose, target_pose, reach_weight = 1, orient_weight = 1, symmetric = True):

    rot_diff = torch.acos(torch.sum(cur_pose.q * target_pose.q, axis=1)**2 * 2 - 1)/torch.pi
    if symmetric:
        rot_diff = rot_diff.minimum(1 - rot_diff)
    pos_diff = torch.linalg.norm(cur_pose.p - target_pose.p, axis=1)
    return pos_diff * reach_weight + rot_diff * orient_weight

def grasp_reward(cur_pose, target_pose, reach_weight = 1, orient_weight = 1, symmetric = True):
    rot_diff = torch.acos(torch.sum(cur_pose.q * target_pose.q, axis=1)**2 * 2 - 1)/torch.pi
    if symmetric:
        rot_diff = rot_diff.minimum(1 - rot_diff)
    pos_diff = torch.linalg.norm(cur_pose.p - target_pose.p, axis=1)
    reaching_reward = 1 - torch.tanh(5 * pos_diff)
    orient_reward = 1 - rot_diff
    return reaching_reward * reach_weight + orient_reward * orient_weight

def orient_then_grasp(cur_pose, target_pose, symmetric = True):
    rot_diff = torch.acos(torch.sum(cur_pose.q * target_pose.q, axis=1)**2 * 2 - 1)/torch.pi
    if symmetric:
        rot_diff = rot_diff.minimum(1 - rot_diff)
    pos_diff = torch.linalg.norm(cur_pose.p - target_pose.p, axis=1)
    reaching_reward = 1 - torch.tanh(5 * pos_diff)
    reward = torch.minimum(1 - rot_diff, torch.tensor(0.95))
    reward += reaching_reward * (reward >= 0.95)
    return reward

def sample_points_on_mesh(mesh):
    samples, face_indices = trimesh.sample.sample_surface_even(mesh, count=1000)
    return samples


if __name__ == "__main__":
    mesh = trimesh.load_mesh("/home/justin/PycharmProjects/ManiSkill/collision_box_t=0.05.ply")
    samples = sample_points_on_mesh(mesh)
    print(samples)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(samples[:, 0], samples[:, 1], samples[:, 2])
    plt.show()