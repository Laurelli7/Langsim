from isaacsim.core.utils import extensions
import omni.graph.core as og
from isaacsim.core.utils.prims import is_prim_path_valid

# Make sure these are enabled somewhere up top (ros2 already is in your script)
extensions.enable_extension("isaacsim.sensors.physx")
extensions.enable_extension("isaacsim.ros2.bridge")


LIDAR_PRIM_PATH = "/World/turtlebot4/rplidar_link/Lidar"  # your lidar
LIDAR_GRAPH_PATH = "/LidarROS2Graph"


def publish_physx_lidar_scan(
    lidar_prim_path: str,
    topic_name: str = "/scan",
    frame_id: str = "base_scan",
    queue_size: int = 10,
):
    if not is_prim_path_valid(lidar_prim_path):
        raise ValueError(f"Lidar path '{lidar_prim_path}' is invalid.")

    # Create the graph the first time if it does not exist
    if not is_prim_path_valid(LIDAR_GRAPH_PATH):
        og.Controller.edit(
            {
                "graph_path": LIDAR_GRAPH_PATH,
                "evaluator_name": "execution",
                "pipeline_stage": og.GraphPipelineStage.GRAPH_PIPELINE_STAGE_SIMULATION,
            },
            {
                og.Controller.Keys.CREATE_NODES: [
                    # Tick each sim frame
                    ("OnTick", "omni.graph.action.OnTick"),

                    # Simulation time → LaserScan timestamp
                    ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),

                    # Read PhysX lidar beams
                    ("ReadLidar", "isaacsim.sensors.physx.IsaacReadLidarBeams"),

                    # Publish LaserScan (ROS 2)
                    ("PublishScan", "isaacsim.ros2.bridge.ROS2PublishLaserScan"),
                ],

                og.Controller.Keys.CONNECT: [
                    # Drive everything from the tick
                    ("OnTick.outputs:tick", "ReadLidar.inputs:execIn"),
                    ("OnTick.outputs:tick", "PublishScan.inputs:execIn"),

                    # Time for ROS messages
                    ("ReadSimTime.outputs:simulationTime",
                     "PublishScan.inputs:timeStamp"),

                    # Wire lidar outputs → LaserScan inputs
                    ("ReadLidar.outputs:azimuthRange",
                     "PublishScan.inputs:azimuthRange"),
                    ("ReadLidar.outputs:depthRange",
                     "PublishScan.inputs:depthRange"),
                    ("ReadLidar.outputs:horizontalFov",
                     "PublishScan.inputs:horizontalFov"),
                    ("ReadLidar.outputs:horizontalResolution",
                     "PublishScan.inputs:horizontalResolution"),
                    ("ReadLidar.outputs:intensitiesData",
                     "PublishScan.inputs:intensitiesData"),
                    ("ReadLidar.outputs:linearDepthData",
                     "PublishScan.inputs:linearDepthData"),
                    ("ReadLidar.outputs:numCols",
                     "PublishScan.inputs:numCols"),
                    ("ReadLidar.outputs:numRows",
                     "PublishScan.inputs:numRows"),
                    ("ReadLidar.outputs:rotationRate",
                     "PublishScan.inputs:rotationRate"),
                ],
            },
        )

    # Configure lidar prim + ROS topic / frame
    og.Controller.edit(
        LIDAR_GRAPH_PATH,
        {
            og.Controller.Keys.SET_VALUES: [
                # Which lidar to read from
                ("ReadLidar.inputs:lidarPrim", lidar_prim_path),

                # ROS2PublishLaserScan inputs
                ("PublishScan.inputs:frameId", frame_id),
                ("PublishScan.inputs:topicName", topic_name),
                ("PublishScan.inputs:queueSize", queue_size),
                # optional nodeNamespace / qosProfile if you want
            ],
        },
    )

    print(f"[INFO] PhysX lidar scan publishing set up:")
    print(f"       lidar prim  : {lidar_prim_path}")
    print(f"       topic       : {topic_name}")
    print(f"       frame_id    : {frame_id}")
    print(f"       graph       : {LIDAR_GRAPH_PATH}")


# ----------------------------------------------------------------------
# Call this once after your camera setup
# ----------------------------------------------------------------------
publish_physx_lidar_scan(
    lidar_prim_path=LIDAR_PRIM_PATH,
    topic_name="/scan",       # classic ROS laser topic
    frame_id="base_scan",     # match your TF tree
)
