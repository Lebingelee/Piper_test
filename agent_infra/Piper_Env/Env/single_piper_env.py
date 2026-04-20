from typing import List, Optional

from agent_infra.Piper_Env.Env.utils.piper_base_env import PiperEnv
from agent_infra.Piper_Env.Env.piper_camera_wrapper import PiperCameraWrapper


class SinglePiperEnv(PiperCameraWrapper):
    """
    单臂 Piper 最终用户入口。
    负责装配单臂 PiperEnv 与相机 wrapper。
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        control_mode: Optional[str] = None,
        hz: Optional[int] = None,
        camera_sns: Optional[List[str]] = None,
        crop_size: Optional[tuple] = None,
        exposure: Optional[List[int]] = None,
        source_resolution: Optional[tuple] = None,
        **kwargs,
    ):
        core_env = PiperEnv(
            config_path=config_path,
            control_mode=control_mode,
            hz=hz,
            **kwargs,
        )

        if len(core_env.arm_names) != 1:
            core_env.close()
            raise ValueError(
                "SinglePiperEnv requires exactly one robot arm in config."
            )

        super().__init__(
            core_env,
            camera_sns=camera_sns,
            crop_size=crop_size,
            exposure=exposure,
            source_resolution=source_resolution,
        )
