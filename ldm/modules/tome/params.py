from abc import ABC
from dataclasses import dataclass
from typing import Optional, Protocol

from ldm.modules.tome.layer import ToMeLayer


class MergeParams(ABC): pass

@dataclass
class BipartiteMixin():
  r: int

class BipartiteParams(BipartiteMixin): pass
class RandomBipartiteParams(BipartiteMixin): pass

@dataclass
class KthBipartiteParams():
  k: int

MergeParams.register(BipartiteParams)
MergeParams.register(RandomBipartiteParams)
MergeParams.register(KthBipartiteParams)

class GetMergeParams(Protocol):
  @staticmethod
  def __call__(
    token_count: int,
    layer: ToMeLayer
  ) -> Optional[MergeParams]: ...