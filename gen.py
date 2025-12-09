# random_cylinders_sdg_with_constraints.py
# Static baked scenes with cylinders and a randomized TurtleBot4 pose
# Includes constraints: No objects in exclusion zone, no uniform colors.

from isaacsim import SimulationApp
import os
import json
import webcolors
import math

CONFIG = {
    "launch_config": {"renderer": "RayTracedLighting", "headless": False},
    "scene_path": "/home/dotin13/isaac-proj/tb4.usd",
    "resolution": [640, 480],
    "num_frames": 50,
    "rt_subframes": 8,
    # Area for randomizing cylinders (and robot XY)
    "spawn_area": {
        "xmin": -3,
        "xmax": 3,
        "ymin": -3,
        "ymax": 3,
        "z": -0.25,
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
    "world_cam_prim": "/World/Camera",
    "tb_cam_prim": "/World/turtlebot4/oakd_link/Camera",
    "robot_prim": "/World/turtlebot4",
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


# ----------------------------------------
# Camera & writer
# ----------------------------------------
def setup_cameras_and_writer(config):
    render_products = []
    world_cam_path = config.get("world_cam_prim")
    tb_cam_path = config.get("tb_cam_prim")

    if world_cam_path:
        rp_world = rep.create.render_product(world_cam_path, config["resolution"])
        render_products.append(rp_world)

    if tb_cam_path:
        rp_tb = rep.create.render_product(tb_cam_path, config["resolution"])
        render_products.append(rp_tb)

    if not render_products:
        carb.log_error("[Camera] No render products created! Check camera prim paths.")
        simulation_app.close()
        raise SystemExit

    outdir = config["writer_config"]["output_dir"]
    if not os.path.isabs(outdir):
        config["writer_config"]["output_dir"] = os.path.join(os.getcwd(), outdir)

    print(f"[Writer] Output dir: {config['writer_config']['output_dir']}")

    writer = rep.WriterRegistry.get("BasicWriter")
    writer.initialize(**config["writer_config"])
    writer.attach(render_products)

    return render_products


# ----------------------------------------
# Strip Replicator graph
# ----------------------------------------
def strip_replicator_graph_from_file(usd_path):
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        carb.log_warn(f"[Replicator] Could not open {usd_path} to strip graph")
        return

    to_remove = []
    for prim in stage.Traverse():
        name = prim.GetName().lower()
        if "sdg" in name:
            to_remove.append(prim.GetPath())

    to_remove = sorted(set(to_remove), key=lambda p: len(str(p)), reverse=True)
    for path in to_remove:
        stage.RemovePrim(path)

    stage.GetRootLayer().Save()


# ----------------------------------------
# Extract Data: Cylinders & Robot
# ----------------------------------------
def get_prim_translation(prim):
    """Safely extract world translation from a prim."""
    try:
        world_mat = omni.usd.get_world_transform_matrix(prim)
        pos = world_mat.ExtractTranslation()
        return [float(pos[0]), float(pos[1]), float(pos[2])]
    except Exception:
        return None


def collect_cylinder_ground_truth(stage):
    gt = []
    time = Usd.TimeCode.Default()

    for prim in stage.Traverse():
        name = prim.GetName()
        if name == "training_cylinder":
            position = get_prim_translation(prim) or [0.0, 0.0, 0.0]

            color = None
            try:
                binding_api = UsdShade.MaterialBindingAPI(prim)
                mat, _ = binding_api.ComputeBoundMaterial()
                if mat:
                    material_prim = mat.GetPrim()
                    shader = None
                    for child in material_prim.GetChildren():
                        if child.GetTypeName() == "Shader":
                            shader = UsdShade.Shader(child)
                            break
                    if shader:
                        inp = shader.GetInput("diffuse_color_constant")
                        if inp:
                            val = inp.Get(time)
                            if val is not None:
                                color = [float(val[0]), float(val[1]), float(val[2])]
            except Exception:
                pass

            gt.append(
                {
                    "path": str(prim.GetPath()),
                    "position": position,
                    "color": color,
                }
            )
    return gt


def get_robot_position(stage, robot_path):
    prim = stage.GetPrimAtPath(robot_path)
    if not prim.IsValid():
        return None
    return get_prim_translation(prim)


# ----------------------------------------
# NEW: Validation Logic
# ----------------------------------------
def validate_scene(stage, config):
    """
    Returns (is_valid, cylinder_data)
    Constraints:
    1. No object (robot or cylinder) within x=[-1, 1] AND y=[-1, 1].
    2. Not all cylinders have the same color.
    """

    # --- Check Robot Position ---
    robot_pos = get_robot_position(stage, config["robot_prim"])
    if robot_pos:
        rx, ry = robot_pos[0], robot_pos[1]
        # Check exclusion zone: if inside x +/- 1.0 AND y +/- 1.0
        if -1.0 < rx < 1.0 and -1.0 < ry < 1.0:
            return False, "Robot inside exclusion zone"

    # --- Check Cylinders ---
    cylinders = collect_cylinder_ground_truth(stage)
    colors_seen = set()

    for cyl in cylinders:
        cx, cy = cyl["position"][0], cyl["position"][1]

        # 1. Exclusion Zone Check
        if -1.0 < cx < 1.0 and -1.0 < cy < 1.0:
            return False, f"Cylinder at ({cx:.2f}, {cy:.2f}) inside exclusion zone"

        # Collect color for next check (convert list to tuple for set hashing)
        if cyl["color"]:
            # Rounding to 3 decimal places to avoid float precision mismatches
            c_tuple = tuple(round(c, 3) for c in cyl["color"])
            colors_seen.add(c_tuple)

    # 2. Color Uniformity Check
    # If we found colors, but the set length is only 1, they are all the same
    if len(cylinders) > 0 and len(colors_seen) <= 1:
        return False, "All cylinders have the same color"

    return True, cylinders


def save_ground_truth_for_snapshot(
    cylinder_data, frame_index, snapshot_filename, out_dir
):
    data = {
        "frame_index": int(frame_index),
        "scene_usd": snapshot_filename,
        "cylinders": cylinder_data,
    }
    gt_filename = f"scene_{frame_index:04d}_gt.json"
    gt_path = os.path.join(out_dir, gt_filename)
    with open(gt_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"[GT] Saved ground truth: {gt_path}")


# ----------------------------------------
# Scene Setup
# ----------------------------------------
def create_cylinders(config):
    area = config["spawn_area"]
    count = config["objects_per_frame"]
    with rep.create.cylinder(
        count=count,
        position=rep.distribution.uniform(
            (area["xmin"], area["ymin"], area["z"]),
            (area["xmax"], area["ymax"], area["z"]),
        ),
        scale=rep.distribution.uniform((0.05, 0.05, 1.0), (0.15, 0.15, 1.0)),
        rotation=rep.distribution.uniform((0.0, 0.0, 0.0), (0.0, 0.0, 360.0)),
        semantics={"class": "Cylinder"},
        name="training_cylinder",
        as_mesh=True,
        visible=True,
    ) as cylinders:
        rep.physics.collider(approximation_shape="convexHull")
        rep.physics.rigid_body()
    return cylinders


def register_randomizers(config, cylinders, stage):
    area = config["spawn_area"]
    count = config["objects_per_frame"]
    robot_path = config.get("robot_prim", "/World/turtlebot4")

    # Colors
    colors = [webcolors.name_to_rgb(name) for name in webcolors.names("html4")]
    colors = [(c.red / 255.0, c.green / 255.0, c.blue / 255.0) for c in colors]

    def randomize_existing_cylinders():
        with cylinders:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (area["xmin"], area["ymin"], area["z"]),
                    (area["xmax"], area["ymax"], area["z"]),
                ),
                rotation=rep.distribution.uniform((0.0, 0.0, 0.0), (0.0, 0.0, 360.0)),
                scale=rep.distribution.uniform((0.05, 0.05, 1.0), (0.15, 0.15, 1.0)),
            )
            mats = rep.create.material_omnipbr(
                diffuse=rep.distribution.choice(colors), count=count
            )
            rep.randomizer.materials(mats)
        return cylinders.node

    def randomize_robot_pose():
        robot_prims = rep.get.prim_at_path(robot_path)
        with robot_prims:
            rep.modify.pose(
                position=rep.distribution.uniform(
                    (area["xmin"], area["ymin"], -0.75),
                    (area["xmax"], area["ymax"], -0.7),
                ),
                rotation=rep.distribution.uniform((0.0, 0.0, 0.0), (0.0, 0.0, 360.0)),
            )
        return robot_prims.node

    rep.randomizer.register(randomize_existing_cylinders)
    rep.randomizer.register(randomize_robot_pose)


# ----------------------------------------
# Initialization
# ----------------------------------------
stage = load_scene(CONFIG["scene_path"])
os.makedirs(CONFIG["scene_snapshot_dir"], exist_ok=True)

new_scene_path = os.path.join(CONFIG["scene_snapshot_dir"], "base_scene.usd")
omni.usd.get_context().save_as_stage(new_scene_path)
stage = omni.usd.get_context().get_stage()

render_products = setup_cameras_and_writer(CONFIG)
cylinders = create_cylinders(CONFIG)
register_randomizers(CONFIG, cylinders, stage)

with rep.trigger.on_frame():
    rep.randomizer.randomize_existing_cylinders()
    rep.randomizer.randomize_robot_pose()


# ----------------------------------------
# Main Loop (Filtered)
# ----------------------------------------
valid_frames_generated = 0
target_frames = CONFIG["num_frames"]

print(f"[SDG] Starting generation. Target: {target_frames} valid frames.")

while valid_frames_generated < target_frames:
    # 1. Step the physics/randomizer
    # Note: This technically triggers the Writer to write images for *this* step.
    # If the frame is invalid, we simply won't save the matching USD/JSON.
    rep.orchestrator.step(rt_subframes=CONFIG["rt_subframes"], delta_time=1.0 / 60.0)

    # 2. Update stage reference and validate
    stage = omni.usd.get_context().get_stage()
    is_valid, validation_result = validate_scene(stage, CONFIG)

    if not is_valid:
        print(f"[Skip] Invalid frame: {validation_result}")
        continue  # Skip saving, do not increment counter

    # 3. Save Valid Data
    snapshot_filename = f"scene_{valid_frames_generated:04d}.usd"
    snapshot_path = os.path.join(CONFIG["scene_snapshot_dir"], snapshot_filename)

    # Save GT (passing the data we already collected during validation)
    # validation_result contains the cylinder list if is_valid is True
    save_ground_truth_for_snapshot(
        validation_result,
        frame_index=valid_frames_generated,
        snapshot_filename=snapshot_filename,
        out_dir=CONFIG["scene_snapshot_dir"],
    )

    # Export USD
    stage.GetRootLayer().Export(snapshot_path)
    strip_replicator_graph_from_file(snapshot_path)

    print(f"[Scene] Saved frame {valid_frames_generated}: {snapshot_path}")
    valid_frames_generated += 1

rep.orchestrator.wait_until_complete()
print(f"[DONE] Output saved to {CONFIG['writer_config']['output_dir']}")
simulation_app.close()
