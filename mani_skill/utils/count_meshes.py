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
import sys


file = sys.argv[1]
mesh = trimesh.load_mesh(file)

cc = mesh.split(only_watertight=True)
print(len(cc))