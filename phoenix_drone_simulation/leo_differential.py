import os
import time
import numpy as np
import pybullet as p
import pybullet_data
from pybullet_utils import bullet_client
from phoenix_drone_simulation.envs.utils import rad2deg, differential_drive, generate_path

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

# === Set camera position
bc.resetDebugVisualizerCamera(
    cameraTargetPosition=(0.0, 0.0, 0.0),
    cameraDistance=1.5,
    cameraYaw=90,
    cameraPitch=-70
)

# Load LeoRover
startPos = (0,0,0.1)
startRPY = (0,0,0)
leo_path = "leo_description/urdf/leo.urdf"
leo = bc.loadURDF(os.path.join(get_assets_path(), leo_path),
    startPos, 
    p.getQuaternionFromEuler(startRPY), 
    flags=p.URDF_USE_INERTIA_FROM_FILE)

wheels = [2,3,5,6]
wheelVelocities = [0, 0, 0, 0]

path = generate_path(10)
waypoints = path.circle()
target_index = 0
leo_target = waypoints[target_index]

target_visual = bc.createVisualShape(
            bc.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[1, 0.1, 0.05, 0.99],
        )
target_body_id = bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=leo_target)

# === Draw reference circle
for k in range(path.num_points):
    bc.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=bc.createVisualShape(
            bc.GEOM_SPHERE,
            radius=0.05,
            rgbaColor=[0.0, 0.0, 1.0, 1],
        ),
        basePosition=waypoints[k]
    )


for i in range (100000):
    bc.stepSimulation()
     
    bc.resetBasePositionAndOrientation(
            target_body_id,
            posObj=leo_target,
            ornObj=(0, 0, 0, 1)
        )
    
    # Get the current position and orientation
    leo_xyz, leo_w = bc.getBasePositionAndOrientation(leo)   

    # Calculate Distance
    dist = np.linalg.norm(np.array(leo_target - leo_xyz))
    yaw = bc.getEulerFromQuaternion(leo_w)[2]

    if dist < 0.15:
        target_index += 1
        leo_target = waypoints[target_index]
    

    
    # Two DOF PID Controller
    wheelVelocities = differential_drive(leo_xyz, leo_target, yaw, (0.2, 1.0))
    wheelVelocities = wheelVelocities.tolist()

    for i in range(len(wheels)):
        bc.setJointMotorControl2(leo,
                            wheels[i],
                            bc.VELOCITY_CONTROL,
                            targetVelocity=wheelVelocities[i],
                            force=100)
    # bc.setJointMotorControlArray(leo, wheels, bc.VELOCITY_CONTROL, wheelVelocities, forces=[1000.0, 1000.0, 1000.0, 1000.0])

    time.sleep(1./240.)
