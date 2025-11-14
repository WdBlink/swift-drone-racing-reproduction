"""感知模块包入口。

文件说明:
    提供视觉/惯性里程计（VIO）与闸门检测的仿真存根，
    以及融合生成低维观测的工具。

作者:
    wdblink
"""

from .vio_stub import VIOStub, VIONoiseConfig
from .gate_detector_stub import GateDetectorStub, DetectorNoiseConfig
from .obs_fusion import PerceptionFusion, FusionConfig
