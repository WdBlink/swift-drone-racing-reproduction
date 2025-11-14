"""感知融合生成低维观测。

文件说明:
    将 VIO 与闸门检测结果融合，输出策略使用的低维状态表示，
    包含到当前/下一门的相对向量、速度与不确定度信息。

作者:
    wdblink
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from .vio_stub import VIOStub, VIONoiseConfig
from .gate_detector_stub import GateDetectorStub, DetectorNoiseConfig


@dataclass
class FusionConfig:
    """融合配置。"""
    use_conf_weight: bool = True


class PerceptionFusion:
    """感知融合器。"""

    def __init__(self, fusion_cfg: Optional[FusionConfig] = None):
        self.vio = VIOStub(VIONoiseConfig())
        self.detector = GateDetectorStub(DetectorNoiseConfig())
        self.cfg = fusion_cfg or FusionConfig()

    def fuse(self, truth_p: np.ndarray, truth_v: np.ndarray, yaw_rad: float,
             to_gate_truth: np.ndarray, to_next_truth: np.ndarray) -> Dict[str, Any]:
        """融合生成观测字典。

        Args:
            truth_p: 真实位置。
            truth_v: 真实速度。
            yaw_rad: 真实yaw。
            to_gate_truth: 到当前门的真实相对向量。
            to_next_truth: 到下一门的真实相对向量。

        Returns:
            低维观测字典，包含带噪估计与置信度。
        """
        self.vio.push_truth(truth_p, truth_v, yaw_rad, 0.0)
        p_est, v_est, yaw_est = self.vio.estimate()
        to_gate_est, conf_gate = self.detector.detect(to_gate_truth)
        to_next_est, conf_next = self.detector.detect(to_next_truth)

        if self.cfg.use_conf_weight:
            to_gate_out = conf_gate * to_gate_est + (1.0 - conf_gate) * to_gate_truth.astype(np.float32)
            to_next_out = conf_next * to_next_est + (1.0 - conf_next) * to_next_truth.astype(np.float32)
        else:
            to_gate_out = to_gate_est
            to_next_out = to_next_est

        return {
            "pos": p_est,
            "vel": v_est,
            "yaw": float(yaw_est),
            "to_gate": to_gate_out.astype(np.float32),
            "to_next": to_next_out.astype(np.float32),
            "conf_gate": float(conf_gate),
            "conf_next": float(conf_next),
        }
