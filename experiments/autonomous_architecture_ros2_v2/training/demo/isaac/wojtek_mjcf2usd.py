"""Convert Wojtek MJCF -> USD on a GPU box via Isaac Sim MJCF importer (headless)."""
import os, sys
from isaacsim import SimulationApp
app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.mjcf")
app.update()

import omni.kit.commands
import omni.usd

XML = os.path.expanduser("~/wojtek_asset/mujoco/wojtek.xml")
DEST = os.path.expanduser("~/wojtek_asset/wojtek.usd")
assert os.path.isfile(XML), XML

# import config
status, cfg = omni.kit.commands.execute("MJCFCreateImportConfig")
cfg.set_fix_base(False)              # freejoint root -> floating base
cfg.set_import_inertia_tensor(True)
cfg.set_make_default_prim(True)
cfg.set_distance_scale(1.0)          # MuJoCo meters
cfg.set_self_collision(False)
print("CONFIG_OK", flush=True)

ok = omni.kit.commands.execute(
    "MJCFCreateAsset",
    mjcf_path=XML,
    import_config=cfg,
    prim_path="/wojtek",
    dest_path=DEST,
)
print("IMPORT_CMD_RESULT", ok, flush=True)

# sanity: open the produced USD and list articulation joints
app.update()
ctx = omni.usd.get_context()
if ctx.open_stage(DEST):
    stage = ctx.get_stage()
    prims = [p.GetPath().pathString for p in stage.Traverse()]
    joints = [p for p in prims if "joint" in p.lower()]
    print("USD_PRIM_COUNT", len(prims), flush=True)
    print("JOINT_PRIMS", len(joints), flush=True)
    for j in joints[:20]:
        print("  J", j, flush=True)
    print("USD_WRITTEN", DEST, os.path.getsize(DEST), "bytes", flush=True)
else:
    print("FATAL: could not open produced USD", flush=True)

print("CLEAN_EXIT", flush=True)
app.close()
