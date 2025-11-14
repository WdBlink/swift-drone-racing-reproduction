"""sim 包初始化模块。

文件说明:
    提供竞速赛道与仿真环境的包入口，便于外部导入与使用。

作者:
    wdblink
"""

from .track import Gate, RaceTrack
from .env import DroneRaceEnv, EnvConfig
