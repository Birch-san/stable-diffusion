from dataclasses import dataclass
from typing import Optional

from torch import Tensor


@dataclass
class ToMeInfo:
  source: Optional[Tensor]
  trace_source: bool