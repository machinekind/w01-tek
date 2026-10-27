"""The four SCAN pieces a sim needs, assembled: map, sensor, planner, executor.

Both simulators use the identical stack -- the Tier-A kinematic sim and the
physics room demo differ only in how the resulting velocity command is
consumed (integrated directly vs handed to the locomotion policy). Keeping
the assembly in one place is what makes a Tier-A result mean something about
the Tier-B run.
"""

from __future__ import annotations

from wojtek_rl.navigation import NavConfig
from wojtek_rl.scan.executor import ScanExecutor
from wojtek_rl.scan.footprint import Footprint
from wojtek_rl.scan.localmap import MapConfig, SlidingOccupancyMap
from wojtek_rl.scan.planner import PlannerConfig
from wojtek_rl.scan.sense import EgoDepthSensor

# Same velocity envelope the room demo drives the policy with, so a Tier-A
# trajectory is one the physics tier can actually execute.
NAV_PROFILE = NavConfig(vx_max=0.4, vy_max=0.25, yaw_max=0.7, stop_radius=0.12)


class ScanStack:
    """Depth sensor + sliding map + planner-backed mid-level executor."""

    def __init__(self, model, data, dt: float, cfg: NavConfig | None = None,
                 camera: str = "ego", map_cfg: MapConfig | None = None):
        self.model = model
        self.data = data
        self.dt = dt
        self.cfg = cfg or NAV_PROFILE
        self.map_cfg = map_cfg or MapConfig()
        self.footprint = Footprint(radius=self.map_cfg.inflate_r)
        self.sensor = EgoDepthSensor(model, data, camera=camera)
        self.map = SlidingOccupancyMap(self.map_cfg)
        self.executor = self._new_executor()

    def _new_executor(self) -> ScanExecutor:
        return ScanExecutor(
            self.cfg, self.map, dt=self.dt,
            planner=_planner(self.map, self.cfg, self.footprint),
        )

    @property
    def planner(self):
        return self.executor.planner

    def reset(self) -> None:
        """Forget the map: a new episode has not seen anything yet."""
        self.map = SlidingOccupancyMap(self.map_cfg)
        self.executor = self._new_executor()

    def sense(self, x: float, y: float) -> None:
        self.sensor.update(self.map, (x, y))

    def close(self) -> None:
        self.sensor.close()


def _planner(omap, cfg: NavConfig, footprint: Footprint):
    from wojtek_rl.scan.planner import ScanPlanner

    return ScanPlanner(
        omap,
        PlannerConfig(v_max=cfg.vx_max, vy_max=cfg.vy_max, yaw_max=cfg.yaw_max),
        footprint=footprint,
    )
