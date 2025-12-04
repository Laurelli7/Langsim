import omni
import omni.graph.core as og

from isaacsim.core.utils import extensions
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.sensors.camera import Camera
from isaacsim.core.nodes.scripts.utils import set_target_prims

# ----------------------------------------------------------------------
# CONFIG – set this to your existing camera prim
# ----------------------------------------------------------------------
CAMERA_PRIM_PATH = "/World/turtlebot4/oakd_link/Camera"  # <-- EDIT THIS

# Graph paths
ROS_CAMERA_GRAPH_PATH = "/World/ROS2CameraGraph"
ROS_TF_GRAPH_PATH = "/World/CameraTFActionGraph"

# ROS topics / frame
RGB_TOPIC = "/camera_1/rgb"
DEPTH_TOPIC = "/camera_1/depth"
PC_TOPIC = "/camera_1/pointcloud"
INFO_TOPIC = "/camera_1/camera_info"
FRAME_ID = "turtle"  # or camera prim name, etc.

FREQ = 30.0  # Hz, used if you later add rate limiting


# ----------------------------------------------------------------------
# 1) Enable ROS 2 bridge and validate camera prim
# ----------------------------------------------------------------------
extensions.enable_extension("isaacsim.ros2.bridge")

if not is_prim_path_valid(CAMERA_PRIM_PATH):
    raise RuntimeError(f"Camera prim path '{CAMERA_PRIM_PATH}' is not valid in the stage")

camera = Camera(prim_path=CAMERA_PRIM_PATH)
camera.initialize()

print(f"[INFO] Camera prim: {CAMERA_PRIM_PATH}")
print(f"[INFO] Camera name: {camera.name}")


# ----------------------------------------------------------------------
# 2) Build graph that:
#    - creates a render product from CAMERA_PRIM_PATH
#    - feeds Camera Helper / Camera Info Helper nodes
# ----------------------------------------------------------------------
def build_ros_camera_graph():
    print(f"[INFO] Creating ROS camera graph at '{ROS_CAMERA_GRAPH_PATH}'")

    (graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": ROS_CAMERA_GRAPH_PATH,
            "evaluator_name": "execution",
            "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
        },
        {
            og.Controller.Keys.CREATE_NODES: [
                # Ticking & time / ROS context
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("RosContext", "isaacsim.ros2.bridge.ROS2Context"),
                ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),

                # Render product creation node
                # UI name: "Isaac Create Render Product"
                ("CreateRenderProduct", "isaacsim.core.nodes.IsaacCreateRenderProduct"),

                # UI name: "ROS 2 Camera Helper"
                ("RgbHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("DepthHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("PcHelper", "isaacsim.ros2.bridge.ROS2CameraHelper"),

                # UI name: "ROS 2 Camera Info Helper"
                ("InfoHelper", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],

            og.Controller.Keys.SET_VALUES: [
                # ROS context: use env ROS_DOMAIN_ID if set
                ("RosContext.inputs:useDomainIDEnvVar", True),
                ("RosContext.inputs:domainID", 0),

                # Simulation time config
                ("SimTime.inputs:useWallClock", False),

                # Create render product from our camera
                ("CreateRenderProduct.inputs:cameraPrim", CAMERA_PRIM_PATH),
                ("CreateRenderProduct.inputs:enabled", True),

                # RGB helper
                ("RgbHelper.inputs:type", "rgb"),  # dropdown in UI
                ("RgbHelper.inputs:topicName", RGB_TOPIC),
                ("RgbHelper.inputs:frameId", FRAME_ID),

                # Depth helper
                ("DepthHelper.inputs:type", "depth"),
                ("DepthHelper.inputs:topicName", DEPTH_TOPIC),
                ("DepthHelper.inputs:frameId", FRAME_ID),

                # Pointcloud helper
                ("PcHelper.inputs:type", "pointcloud"),
                ("PcHelper.inputs:topicName", PC_TOPIC),
                ("PcHelper.inputs:frameId", FRAME_ID),

                # Camera info helper
                ("InfoHelper.inputs:topicName", INFO_TOPIC),
                ("InfoHelper.inputs:frameId", FRAME_ID),
            ],

            og.Controller.Keys.CONNECT: [
                # Tick drives sim time, render product, and helpers
                ("OnTick.outputs:tick", "CreateRenderProduct.inputs:execIn"),
                ("OnTick.outputs:tick", "RgbHelper.inputs:execIn"),
                ("OnTick.outputs:tick", "DepthHelper.inputs:execIn"),
                ("OnTick.outputs:tick", "PcHelper.inputs:execIn"),
                ("OnTick.outputs:tick", "InfoHelper.inputs:execIn"),

                # Time to helpers (timestamps)
                ("SimTime.outputs:simulationTime", "RgbHelper.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "DepthHelper.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "PcHelper.inputs:timeStamp"),
                ("SimTime.outputs:simulationTime", "InfoHelper.inputs:timeStamp"),

                # Render product feeds all helpers
                # NOTE: the output name is often "outputs:renderProduct" or
                # "outputs:renderProductPath". Check in the node UI if needed.
                ("CreateRenderProduct.outputs:renderProductPath", "RgbHelper.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "DepthHelper.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "PcHelper.inputs:renderProductPath"),
                ("CreateRenderProduct.outputs:renderProductPath", "InfoHelper.inputs:renderProductPath"),
            ],
        },
    )

    print("[INFO] ROS camera graph built with render product + helpers.")
    print(f"       RGB topic:       {RGB_TOPIC}")
    print(f"       Depth topic:     {DEPTH_TOPIC}")
    print(f"       PointCloud:      {PC_TOPIC}")
    print(f"       CameraInfo:      {INFO_TOPIC}")


# ----------------------------------------------------------------------
# 3) Build TF graph for the camera (TransformTree + Clock)
# ----------------------------------------------------------------------
def build_camera_tf_graph():
    camera_frame_id = CAMERA_PRIM_PATH.split("/")[-1]

    if not is_prim_path_valid(CAMERA_PRIM_PATH):
        raise ValueError(f"Camera path '{CAMERA_PRIM_PATH}' is invalid.")

    if not is_prim_path_valid(ROS_TF_GRAPH_PATH):
        (tf_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": ROS_TF_GRAPH_PATH,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnPlaybackTick"),
                    ("SimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("RosClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "SimTime.inputs:execIn"),
                    ("OnTick.outputs:tick", "RosClock.inputs:execIn"),
                    ("SimTime.outputs:simulationTime", "RosClock.inputs:timeStamp"),
                ],
            },
        )
        print(f"[INFO] Base TF graph created at '{ROS_TF_GRAPH_PATH}'")

    og.Controller.edit(
        ROS_TF_GRAPH_PATH,
        {
            og.Controller.Keys.CREATE_NODES: [
                ("PublishTF_" + camera_frame_id,
                 "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishTF_" + camera_frame_id + ".inputs:topicName", "/tf"),
            ],
            og.Controller.Keys.CONNECT: [
                (ROS_TF_GRAPH_PATH + "/OnTick.outputs:tick",
                 "PublishTF_" + camera_frame_id + ".inputs:execIn"),
                (ROS_TF_GRAPH_PATH + "/SimTime.outputs:simulationTime",
                 "PublishTF_" + camera_frame_id + ".inputs:timeStamp"),
            ],
        },
    )

    # Attach the camera prim so its pose is published
    set_target_prims(
        primPath=ROS_TF_GRAPH_PATH + "/PublishTF_" + camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[CAMERA_PRIM_PATH],
    )

    print(f"[INFO] TF publishing set up for frame '{camera_frame_id}' in '{ROS_TF_GRAPH_PATH}'")


# ----------------------------------------------------------------------
# 4) Build graphs ONCE, then save the stage
# ----------------------------------------------------------------------
# build_ros_camera_graph()
build_camera_tf_graph()

print("\n[INFO] Graphs created:")
print(f"       {ROS_CAMERA_GRAPH_PATH}")
print(f"       {ROS_TF_GRAPH_PATH}")
print("[INFO] Now SAVE your stage (.usd).")
print("[INFO] After that, simply open the scene and press PLAY;")
print("       Isaac Sim will use the created render product and helper nodes")
print("       to publish RGB, depth, pointcloud, camera info, and TF automatically.")
