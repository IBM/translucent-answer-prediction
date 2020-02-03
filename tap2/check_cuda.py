import torch
import sys

if torch.torch.cuda.is_available():
    sys.exit(0)
else:
    sys.exit(1)
