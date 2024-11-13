from .pointnet2_backbone import PointNet2Backbone, PointNet2MSG, PointNet2MSG_fsa, PointNet2MSG_dsa
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_backbone_2d import PillarBackBone8x, PillarRes18BackBone8x, PillarRes18v2
from .spconv_backbone_focal import VoxelBackBone8xFocal
from .spconv_backbone_voxelnext import VoxelResBackBone8xVoxelNeXt
from .spconv_backbone_voxelnext2d import VoxelResBackBone8xVoxelNeXt2D
from .spconv_unet import UNetV2
from .dsvt import DSVT
from .spconv_backbone_largekernel import PillarResBackBone8xLargeKernel2D
from .Inceptionnext import PillarInceptionNextBackbone
from .starnet import StarNet

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2Backbone': PointNet2Backbone,
    'PointNet2MSG': PointNet2MSG,
    'PointNet2MSG_fsa': PointNet2MSG_fsa,
    'PointNet2MSG_dsa': PointNet2MSG_dsa,
    'VoxelResBackBone8x': VoxelResBackBone8x,
    'VoxelBackBone8xFocal': VoxelBackBone8xFocal,
    'VoxelResBackBone8xVoxelNeXt': VoxelResBackBone8xVoxelNeXt,
    'VoxelResBackBone8xVoxelNeXt2D': VoxelResBackBone8xVoxelNeXt2D,
    'PillarBackBone8x': PillarBackBone8x,
    'PillarRes18BackBone8x': PillarRes18BackBone8x,
    'PillarRes18v2': PillarRes18v2,
    'DSVT': DSVT,
    'PillarResBackBone8xLargeKernel2D': PillarResBackBone8xLargeKernel2D,
    'PillarInceptionNextBackbone': PillarInceptionNextBackbone,
    'StarNet': StarNet

}
