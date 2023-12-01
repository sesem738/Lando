import pybullet as p
import pybullet_data
import math
import time

# Function to calculate Lemniscate path coordinates
def lemniscate_path(t):
    a = 5.0  # Adjust the values of 'a' and 'b' for the desired Lemniscate shape
    b = 2.0
    x = a * math.cos(t) / (1 + math.sin(t) ** 2)
    y = b * math.sin(t) * math.cos(t) / (1 + math.sin(t) ** 2)
    return x, y

# Initialize PyBullet
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

planeId = p.loadURDF("plane.urdf")

# Load Husky URDF file
husky_start_pos = [0, 0, 0.1]
husky_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
husky = p.loadURDF("husky/husky.urdf", husky_start_pos, husky_start_orientation)

# Set up simulation parameters
p.setGravity(0, 0, -9.81)
p.setTimeStep(1.0 / 240.0)
p.setRealTimeSimulation(0)

# Simulation loop
t = 0
while True:
    # Calculate Lemniscate path coordinates
    x, y = lemniscate_path(t)

    # Get current Husky position and orientation
    husky_pos, husky_orn = p.getBasePositionAndOrientation(husky)

    # Calculate desired Husky orientation towards the Lemniscate path
    target_angle = math.atan2(y - husky_pos[1], x - husky_pos[0])
    target_orn = p.getQuaternionFromEuler([0, 0, target_angle])

    # Apply linear and angular velocities to follow the Lemniscate path
    p.setJointMotorControl2(husky, 0, p.VELOCITY_CONTROL, targetVelocity=1.0, force=5.0)
    p.setJointMotorControl2(husky, 1, p.VELOCITY_CONTROL, targetVelocity=1.0, force=5.0)
    p.setJointMotorControl2(husky, 2, p.POSITION_CONTROL, targetPosition=0, force=5.0)
    p.setJointMotorControl2(husky, 3, p.POSITION_CONTROL, targetPosition=0, force=5.0)
    p.setJointMotorControl2(husky, 4, p.POSITION_CONTROL, targetPosition=0, force=5.0)
    p.setJointMotorControl2(husky, 5, p.POSITION_CONTROL, targetPosition=0, force=5.0)

    # Update simulation time
    t += 0.01
    time.sleep(0.01)
    p.stepSimulation()

# Clean up
p.disconnect()
