"""SCAN-Planner: spatial collision-aware local planning for Wojtek.

Port of "SCAN-Planner: Spatial Collision-Aware Local Planning for
Route-Guided Long-Range Quadruped Navigation" (Zheng et al., arXiv 2606.19555)
onto the existing VLM navigation stack, sim-only and ROS-free.

The VLM keeps deciding *where* to go -- it emits the same mid-level commands
as before -- and this layer decides how to get there without hitting
anything, using only the simulated RealSense depth channel.

    from wojtek_rl.scan import ScanExecutor, ScanPlanner, SlidingOccupancyMap

Modules: localmap (sliding log-odds map), sense (self-masked ego depth),
footprint (yaw-aware twin cylinders), guide (projected A*), traj (B-spline +
rebound optimisation), planner (replan/track), executor (mid-level drop-in),
viz (map overlay). See training/docs/scan-planner.md.
"""

from wojtek_rl.scan.executor import ScanExecutor
from wojtek_rl.scan.footprint import Footprint
from wojtek_rl.scan.guide import GuideSearch
from wojtek_rl.scan.localmap import MapConfig, SlidingOccupancyMap
from wojtek_rl.scan.planner import PlannerConfig, ScanPlanner
from wojtek_rl.scan.sense import EgoDepthSensor

__all__ = [
    "EgoDepthSensor",
    "Footprint",
    "GuideSearch",
    "MapConfig",
    "PlannerConfig",
    "ScanExecutor",
    "ScanPlanner",
    "SlidingOccupancyMap",
]
