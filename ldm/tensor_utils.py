from torch import Tensor

# from lucidrains' imagen-pytorch, MIT-licensed:
# https://github.com/lucidrains/imagen-pytorch/blob/a5d69b9c076b2fdbe99f88ce183dd31f5a956da4/imagen_pytorch/imagen_pytorch.py#L128-L132
def right_pad_dims_to(x: Tensor, t: Tensor) -> Tensor:
    padding_dims: int = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))