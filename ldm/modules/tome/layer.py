from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto


class ToMeLayer(ABC): pass

class UNetSection(Enum):
  Input = auto()
  Middle = auto()
  Output = auto()

@dataclass
class UNetLayerLocation:
  section: UNetSection
  level: int
  res_block_ix: int

@dataclass
class SpatialTransformerLayerLocation:
  spatial_transformer_block_depth: int

@dataclass
class SpatialTransformerSelfAttnLocation():
  unet_location: UNetLayerLocation
  spatial_transformer_location: SpatialTransformerLayerLocation

ToMeLayer.register(SpatialTransformerSelfAttnLocation)

@dataclass
class LayerDescription:
  layer: ToMeLayer