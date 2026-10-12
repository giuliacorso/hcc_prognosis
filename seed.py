import numpy as np
import random
import torch


def init_seed(args):
    # Set the seed for Torch CUDA operations
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # Set the seed for Torch CPU operations
    torch.manual_seed(args.seed)
    # Set the seed for NumPy operations
    np.random.seed(args.seed)
    # Set the seed for random number generation
    random.seed(args.seed)
