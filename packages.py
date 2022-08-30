from operator import imod

# Basic Numerical operations
import numpy as np
from random import uniform, randint

# Neural Network
import torch
from torch import random
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
import torchvision.models as models
from torch.functional import Tensor

# Drawing Plots 
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from mpl_toolkits.mplot3d import Axes3D

# Import Adam Optimizer
import torch.optim as optim

# Report progress
import sys

# Film the learning progress
from PIL import Image

# Render in Jupyter Notebook
import k3d

# Fourier Features
import rff

# To Paraview
from evtk.hl import structuredToVTK

# Chamfer Distance
from typing import Union

import torch
import torch.nn.functional as F
from pytorch3d.ops.knn import knn_gather, knn_points
from pytorch3d.structures.pointclouds import Pointclouds
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from typing import Union