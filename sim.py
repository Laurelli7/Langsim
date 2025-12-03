# random_cylinders_sdg.py
# Static baked scenes with cylinders already in place (no SDG when opening USD)

from isaacsim import SimulationApp
import os
import json  # <--- NEW

CONFIG = {
    "launch_config": {"renderer": "RayTracedLighting", "headless": False},
    "scene_path": "/home/dotin13/isaac-proj/tb4.usd",

    "resolution": [640, 480],
    "num_frames": 50,
    "rt_subframes": 8,

    "spawn_area": {
        "xmin": -2.0, "xmax": 2.0,
        "ymin": -2.0, "ymax": 2.0,
        "z": 0.00,
    },

    "objects_per_frame": 3,

    "writer_config": {
        "output_dir": "_out_random_cylinders",
        "rgb": True,
        "semantic_segmentation": True,
        "bounding_box_3d": True,
        "bounding_box_2d_tight": True,
    },

    "scene_snapshot_dir": "/home/dotin13/isaac-proj/scene_snapshots",
}

simulation_app = SimulationApp(CONFIG["launch_config"])

import omni.usd
import omni.replicator.core as rep
from pxr import Gf, Usd, UsdGeom, UsdShade
import carb

# ----------------------------------------
# Load scene
# ----------------------------------------
def load_scene(path):
    omni.usd.get_context().open_stage(path)
    stage = omni.usd.get_context().get_stage()
    if not stage:
        carb.log_error(f"Failed to load scene: {path}")
        simulation_app.close()
        raise SystemExit
    print(f"[Scene] Loaded: {path}")
    return stage


stage = load_scene(CONFIG["scene_path"])


# ----------------------------------------
# Camera & writer
# ----------------------------------------
def setup_camera_and_writer(config):
    cam = rep.create.camera(
        position=(0.0, -3.5, 1.2),
        rotation=(0, 0, 0),
        focal_length=24,
        name="MainCam",
    )

    rp = rep.create.render_product(cam, config["resolution"])

    outdir = config["writer_config"]["output_dir"]
    if not os.path.isabs(outdir):
        config["writer_config"]["output_dir"] = os.path.join(os.getcwd(), outdir)

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(**config["writer_config"])
    writer.attach([rp])

    return cam, rp


camera, render_product = setup_camera_and_writer(CONFIG)


# ----------------------------------------
# Helper: strip Replicator/SDG graph from a USD file
# (so snapshots are static when opened)
# ----------------------------------------
def strip_replicator_graph_from_file(usd_path):
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        carb.log_warn(f"[Replicator] Could not open {usd_path} to strip graph")
        return

    to_remove = []

    # Collect prims that look like Replicator/SDG/OmniGraph nodes
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        type_name = prim.GetTypeName().lower()
        if (
            "sdg" in name
        ):
            to_remove.append(prim.GetPath())

    # Remove from deepest to highest so children go first
    to_remove = sorted(set(to_remove), key=lambda p: len(str(p)), reverse=True)

    for path in to_remove:
        stage.RemovePrim(path)

    stage.GetRootLayer().Save()
    print(f"[Replicator] Stripped {len(to_remove)} prim(s) from {usd_path}")


# ----------------------------------------
# Helper: collect cylinder ground truth from current stage
# ----------------------------------------
def collect_cylinder_ground_truth(stage):
    """
    Returns a list of dicts like:
    {
      "path": "/World/training_cylinder_01",
      "position": [x, y, z],
      "color": [r, g, b] or null if not found
    }
    """
    gt = []
    time = Usd.TimeCode.Default()

    for prim in stage.Traverse():
        name = prim.GetName()
        if name == "training_cylinder":
            # World transform
            try:
                world_mat = omni.usd.get_world_transform_matrix(prim)
                pos = world_mat.ExtractTranslation()
                position = [float(pos[0]), float(pos[1]), float(pos[2])]
            except Exception as e:
                carb.log_warn(f"[GT] Failed to get world transform for {prim.GetPath()}: {e}")
                position = [0.0, 0.0, 0.0]

            # Try to get bound material and its OmniPBR base color
            color = None
            try:
                binding_api = UsdShade.MaterialBindingAPI(prim)
                mat, _ = binding_api.ComputeBoundMaterial()
                if mat:
                    material_prim = mat.GetPrim()
                    shader = None
                    # Look for a Shader child (common pattern for OmniPBR)
                    for child in material_prim.GetChildren():
                        if child.GetTypeName() == "Shader":
                            shader = UsdShade.Shader(child)
                            break

                    if shader:
                        # OmniPBR base color input name is "diffuse_color_constant"
                        # (linear RGB in USD space)
                        inp = shader.GetInput("diffuse_color_constant")
                        if inp:
                            val = inp.Get(time)
                            if val is not None:
                                # val is a Gf.Vec3f
                                color = [float(val[0]), float(val[1]), float(val[2])]
            except Exception as e:
                # Don't kill the run if GT color fails; just leave color=None
                carb.log_warn(f"[GT] Failed to read material color for {prim.GetPath()}: {e}")

            gt.append(
                {
                    "path": str(prim.GetPath()),
                    "position": position,
                    "color": color,
                }
            )
    return gt


def save_ground_truth_for_snapshot(stage, frame_index, snapshot_filename, out_dir):
    """
    Save a JSON file with ground truth data for cylinders in this frame.

    File name: scene_XXXX_gt.json (aligned with USD snapshot).
    """
    cylinders = collect_cylinder_ground_truth(stage)
    data = {
        "frame_index": int(frame_index),
        "scene_usd": snapshot_filename,
        "cylinders": cylinders,
    }

    gt_filename = f"scene_{frame_index:04d}_gt.json"
    gt_path = os.path.join(out_dir, gt_filename)

    with open(gt_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"[GT] Saved ground truth: {gt_path}")


# ----------------------------------------
# Random cylinder generator (correct API)
# ----------------------------------------
def register_cylinder_randomizer(config):

    area = config["spawn_area"]
    count = config["objects_per_frame"]

    def spawn_cylinders():

        cylinders = rep.create.cylinder(
            count=count,
            position=rep.distribution.uniform(
                (area["xmin"], area["ymin"], area["z"]),
                (area["xmax"], area["ymax"], area["z"]),
            ),
            scale=rep.distribution.uniform(
                (0.05, 0.05, 1.0),
                (0.15, 0.15, 1.0),
            ),
            rotation=rep.distribution.uniform((0.0, 0.0, 0.0), (0.0, 0.0, 360.0)),
            semantics={"class": "Cylinder"},
            name="training_cylinder",
            as_mesh=True,
            visible=True,
        )

        # Random colors + physics
        with cylinders:
            rep.randomizer.materials(
                rep.create.material_omnipbr(
                    diffuse=rep.distribution.uniform(
                        (0.0, 0.0, 0.0),
                        (1.0, 1.0, 1.0),
                    )
                )
            )
            rep.physics.collider(approximation_shape="convexHull")
            rep.physics.rigid_body()

        return cylinders.node

    rep.randomizer.register(spawn_cylinders)


# ----------------------------------------
# Create a base scene in snapshot dir (fixes TB4 relative paths)
# ----------------------------------------
os.makedirs(CONFIG["scene_snapshot_dir"], exist_ok=True)

new_scene_path = os.path.join(CONFIG["scene_snapshot_dir"], "base_scene.usd")
omni.usd.get_context().save_as_stage(new_scene_path)
print(f"[Scene] Saved base scene: {new_scene_path}")

# After save_as_stage, the context now points to base_scene.usd
stage = omni.usd.get_context().get_stage()

# ----------------------------------------
# Register SDG and run per-frame
# ----------------------------------------
register_cylinder_randomizer(CONFIG)

with rep.trigger.on_frame():
    rep.randomizer.spawn_cylinders()

print(f"[SDG] Generating {CONFIG['num_frames']} frames")

for i in range(CONFIG["num_frames"]):
    # Run randomizers + writers for this frame
    rep.orchestrator.step(rt_subframes=CONFIG["rt_subframes"], delta_time=0.0)

    # Use a consistent name for snapshot
    snapshot_filename = f"scene_{i:04d}.usd"
    snapshot_path = os.path.join(
        CONFIG["scene_snapshot_dir"],
        snapshot_filename,
    )

    # Current stage reflects this frame – export it
    stage = omni.usd.get_context().get_stage()

    # --- Save ground truth BEFORE stripping Replicator graph ---
    save_ground_truth_for_snapshot(
        stage,
        frame_index=i,
        snapshot_filename=snapshot_filename,
        out_dir=CONFIG["scene_snapshot_dir"],
    )

    # Export the current composed stage (with SDG graph still present)
    stage.GetRootLayer().Export(snapshot_path)

    # Now strip the Replicator/SDG graph from the *file* so it's static
    strip_replicator_graph_from_file(snapshot_path)

    print(f"[Scene] Saved baked snapshot: {snapshot_path}")

rep.orchestrator.wait_until_complete()
print(f"[DONE] Output saved to {CONFIG['writer_config']['output_dir']}")
simulation_app.close()
