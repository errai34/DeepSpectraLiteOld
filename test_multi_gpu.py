
import itertools
import numpy as np
import gc
import matplotlib.pyplot as plt
from time import time
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch import distributions
from torch.distributions import (
    Normal,
    MultivariateNormal,
    Uniform,
    TransformedDistribution,
    SigmoidTransform,
)
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer, required

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    device_ids = list(range(torch.cuda.device_count()))
    gpus = len(device_ids)
    print('GPU detected')
else:
    DEVICE = torch.device("cpu")
    print('No GPU. switching to CPU')