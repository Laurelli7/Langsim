import omni
import omni.graph.core as og
import omni.replicator.core as rep
import omni.syntheticdata._syntheticdata as sd

from isaacsim.core.utils import extensions
from isaacsim.sensors.camera import Camera
from isaacsim.core.utils.prims import is_prim_path_valid
from isaacsim.core.nodes.scripts.utils import set_target_prims

# ----------------------------------------------------------------------
# 0) CONFIG – set this to your existing camera prim
# ----------------------------------------------------------------------
CAMERA_PRIM_PATH = "/World/turtlebot4/oakd_link/Camera"  # <-- EDIT THIS

# ----------------------------------------------------------------------
# 1) Enable ROS 2 bridge (if not already enabled via UI)
# ----------------------------------------------------------------------
extensions.enable_extension("isaacsim.ros2.bridge")

# ----------------------------------------------------------------------
# 2) Wrap existing camera prim with isaacsim.sensors.camera.Camera
# ----------------------------------------------------------------------
camera = Camera(prim_path=CAMERA_PRIM_PATH)
camera.initialize()

print(f"[INFO] Wrapped existing camera at: {CAMERA_PRIM_PATH}")
print(f"[INFO] Camera name: {camera.name}")
print(f"[INFO] Render product: {camera._render_product_path}")

# ----------------------------------------------------------------------
# 3) Helper: publish CameraInfo
# ----------------------------------------------------------------------
def publish_camera_info(camera: Camera, freq: float):
    from isaacsim.ros2.bridge import read_camera_info

    render_product = camera._render_product_path
    step_size = int(60 / freq)  # sim ~60 Hz

    topic_name = camera.name + "_info"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    camera_info, _ = read_camera_info(render_product_path=render_product)

    writer = rep.writers.get("ROS2PublishCameraInfo")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
        width=camera_info.width,
        height=camera_info.height,
        projectionType=camera_info.distortion_model,
        k=camera_info.k.reshape([1, 9]),
        r=camera_info.r.reshape([1, 9]),
        p=camera_info.p.reshape([1, 12]),
        physicalDistortionModel=camera_info.distortion_model,
        physicalDistortionCoefficients=camera_info.d,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        "PostProcessDispatch" + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    print(f"[INFO] CameraInfo -> '{topic_name}', frame_id='{frame_id}', ~{freq} Hz")


# ----------------------------------------------------------------------
# 4) Helper: publish RGB images
# ----------------------------------------------------------------------
def publish_rgb(camera: Camera, freq: float):
    render_product = camera._render_product_path
    step_size = int(60 / freq)

    topic_name = camera.name + "_rgb"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.Rgb.name
    )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    print(f"[INFO] RGB Image -> '{topic_name}', frame_id='{frame_id}', ~{freq} Hz")


# ----------------------------------------------------------------------
# 5) Helper: publish Depth images
# ----------------------------------------------------------------------
def publish_depth(camera: Camera, freq: float):
    render_product = camera._render_product_path
    step_size = int(60 / freq)

    topic_name = camera.name + "_depth"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )
    writer = rep.writers.get(rv + "ROS2PublishImage")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    print(f"[INFO] Depth Image -> '{topic_name}', frame_id='{frame_id}', ~{freq} Hz")


# ----------------------------------------------------------------------
# 6) Helper: publish PointCloud from depth
# ----------------------------------------------------------------------
def publish_pointcloud_from_depth(camera: Camera, freq: float):
    render_product = camera._render_product_path
    step_size = int(60 / freq)

    topic_name = camera.name + "_pointcloud"
    queue_size = 1
    node_namespace = ""
    frame_id = camera.prim_path.split("/")[-1]

    # Depth -> pointcloud using camera intrinsics
    rv = omni.syntheticdata.SyntheticData.convert_sensor_type_to_rendervar(
        sd.SensorType.DistanceToImagePlane.name
    )

    writer = rep.writers.get(rv + "ROS2PublishPointCloud")
    writer.initialize(
        frameId=frame_id,
        nodeNamespace=node_namespace,
        queueSize=queue_size,
        topicName=topic_name,
    )
    writer.attach([render_product])

    gate_path = omni.syntheticdata.SyntheticData._get_node_path(
        rv + "IsaacSimulationGate", render_product
    )
    og.Controller.attribute(gate_path + ".inputs:step").set(step_size)

    print(f"[INFO] PointCloud -> '{topic_name}', frame_id='{frame_id}', ~{freq} Hz")


def publish_camera_tf(camera: Camera):
    camera_prim = camera.prim_path

    if not is_prim_path_valid(camera_prim):
        raise ValueError(f"Camera path '{camera_prim}' is invalid.")

    camera_frame_id = camera_prim.split("/")[-1]
    ros_camera_graph_path = "/World/CameraTFActionGraph"

    if not is_prim_path_valid(ros_camera_graph_path):
        (ros_camera_graph, _, _, _) = og.Controller.edit(
            {
                "graph_path": ros_camera_graph_path,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.
                    GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    ("OnTick", "omni.graph.action.OnTick"),
                    ("IsaacClock", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                    ("RosPublisher", "isaacsim.ros2.bridge.ROS2PublishClock"),
                ],
                og.Controller.Keys.CONNECT: [
                    ("OnTick.outputs:tick", "RosPublisher.inputs:execIn"),
                    ("IsaacClock.outputs:simulationTime",
                     "RosPublisher.inputs:timeStamp"),
                ],
            },
        )

    og.Controller.edit(
        ros_camera_graph_path,
        {
            og.Controller.Keys.CREATE_NODES: [
                ("PublishTF_" + camera_frame_id,
                 "isaacsim.ros2.bridge.ROS2PublishTransformTree"),
                ("PublishRawTF_" + camera_frame_id + "_world",
                 "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"),
            ],
            og.Controller.Keys.SET_VALUES: [
                ("PublishTF_" + camera_frame_id + ".inputs:topicName", "/tf"),
                ("PublishRawTF_" + camera_frame_id + "_world.inputs:topicName",
                 "/tf"),
                ("PublishRawTF_" + camera_frame_id + "_world.inputs:parentFrameId",
                 camera_frame_id),
                ("PublishRawTF_" + camera_frame_id + "_world.inputs:childFrameId",
                 camera_frame_id + "_world"),
                # Static rotation from ROS camera frame to world frame:
                ("PublishRawTF_" + camera_frame_id + "_world.inputs:rotation",
                 [0.5, -0.5, 0.5, 0.5]),
            ],
            og.Controller.Keys.CONNECT: [
                (ros_camera_graph_path + "/OnTick.outputs:tick",
                 "PublishTF_" + camera_frame_id + ".inputs:execIn"),
                (ros_camera_graph_path + "/OnTick.outputs:tick",
                 "PublishRawTF_" + camera_frame_id + "_world.inputs:execIn"),
                (ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                 "PublishTF_" + camera_frame_id + ".inputs:timeStamp"),
                (ros_camera_graph_path + "/IsaacClock.outputs:simulationTime",
                 "PublishRawTF_" + camera_frame_id + "_world.inputs:timeStamp"),
            ],
        },
    )

    # Attach the camera prim so its pose is published
    set_target_prims(
        primPath=ros_camera_graph_path + "/PublishTF_" + camera_frame_id,
        inputName="inputs:targetPrims",
        targetPrimPaths=[camera_prim],
    )

    print(f"[INFO] TF publishing set up for frame '{camera_frame_id}'")



# ----------------------------------------------------------------------
# 7) Set up all publishers for this camera
# ----------------------------------------------------------------------
FREQ = 30.0  # Hz (approximate ROS publish rate)

publish_camera_info(camera, FREQ)
publish_rgb(camera, FREQ)
publish_depth(camera, FREQ)
publish_pointcloud_from_depth(camera, FREQ)
# publish_camera_tf(camera)

print("\n[INFO] Publishers set up for camera prim:", CAMERA_PRIM_PATH)
print("[INFO] Now press PLAY in the main Isaac Sim toolbar.")
print("[INFO] In a ROS 2 terminal (with environment sourced), you should see topics like:")
print(f"       /{camera.name}_camera_info")
print(f"       /{camera.name}_rgb")
print(f"       /{camera.name}_depth")
print(f"       /{camera.name}_pointcloud")
print("       (use `ros2 topic echo` on each to verify)")

