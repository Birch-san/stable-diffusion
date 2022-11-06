import torch
from typing import Literal, TypeAlias

DeviceType: TypeAlias = Literal['cuda', 'mps', 'cpu']

def get_device_type() -> DeviceType:
    if(torch.cuda.is_available()):
        return 'cuda'
    elif(torch.backends.mps.is_available()):
        return 'mps'
    else:
        return 'cpu'
