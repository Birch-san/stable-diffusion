from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class ToMeInfo:
  size: Optional[int]
  source: Optional[Tensor]
  trace_source: bool
  prop_attn: bool
  class_token: bool
  distill_token: bool