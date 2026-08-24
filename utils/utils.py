import os
import sys
import errno
import numpy as np
import numpy.random as random
import torch
from torch import distributed as dist
from tqdm import tqdm
import yaml
from easydict import EasyDict as edict
import pprint
import datetime
import dateutil.tz
from PIL import Image

import importlib
from torchvision.transforms import InterpolationMode
import torch.nn.functional as F
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC



def get_time_stamp():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')  
    return timestamp