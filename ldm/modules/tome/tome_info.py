from dataclasses import dataclass
from typing import List, Optional

from torch import Tensor


@dataclass
class ToMeInfo:
  # how many layers may participate in ToMe. when the layer is constructed, it should increment this counter.
  candidates: int
  r: List[int]
  size: Optional[int]
  source: Optional[Tensor]
  trace_source: bool
  prop_attn: bool
  class_token: bool
  distill_token: bool