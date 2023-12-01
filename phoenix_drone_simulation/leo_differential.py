import os
import time
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client

def get_assets_path() -> str:
    r""" Returns the path to the files located in envs/data."""
    file = "/home/sesem/WorldWideWeb/Lando/phoenix_drone_simulation/envs/"
    data_path = os.path.join(os.path.dirname(file), 'assets')
    return data_path

bc = bullet_client.BulletClient(connection_mode=p.GUI)

# disable GUI debug visuals
bc.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)

bc.resetSimulation()
bc.setPhysicsEngineParameter(fixedTimeStep=1.0/240.0, numSolverIterations=5,deterministicOverlappingPairs=1)
bc.setGravity(0, 0, -9.81)

bc.setAdditionalSearchPath(pybullet_data.getDataPath())
planeId = bc.loadURDF("plane.urdf")

# Load LeoRover
startPos = (0,0,0.2)
startRPY = (0,0,0)
leo_path = "leo_description/urdf/leo.urdf"
ledId = bc.loadURDF(os.path.join(get_assets_path(), leo_path),
    startPos, 
    p.getQuaternionFromEuler(startRPY), 
    flags=p.URDF_USE_INERTIA_FROM_FILE)

# === Set camera position
bc.resetDebugVisualizerCamera(
    cameraTargetPosition=(0.0, 0.0, 0.0),
    cameraDistance=1.5,
    cameraYaw=90,
    cameraPitch=-70
)

for i in range (10000):
    bc.stepSimulation()
    time.sleep(1./240.)
