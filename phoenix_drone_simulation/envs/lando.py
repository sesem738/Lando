import numpy as np
from phoenix_drone_simulation.envs.base_lando import DroneLandoBaseEnv
from phoenix_drone_simulation.envs.utils import deg2rad
from phoenix_drone_simulation.envs.utils import differential_drive, generate_path


class LandoBaseEnv(DroneLandoBaseEnv):
    def __init__(
            self,
            physics,
            control_mode: str,
            drone_model: str,
            observation_noise=1,  # must be positive in order to add noise
            domain_randomization: float = 0.10,  # use 10% DR as default
            target_pos: np.ndarray = np.array([0, 0, 1.0], dtype=np.float32),
            sim_freq=200,  # in Hz
            aggregate_phy_steps=2,  # sub-steps used to calculate motor dynamics
            observation_frequency=100,  # in Hz
            penalty_action: float = 1e-4,
            penalty_angle: float = 0,
            penalty_spin: float = 1e-3,  # Note: 1e-4 resulted in fast yaw spin
            penalty_terminal: float = 100,
            penalty_velocity: float = 1e-4,
            **kwargs
    ):
        # === Circle task specific attributes
        # must be defined before calling super class constructor:
        self.target_pos = target_pos  # used in _computePotential()
        self.ARP = 1e-3  # action rate penalty
        self.penalty_action = penalty_action
        self.penalty_angle = penalty_angle
        self.penalty_spin = penalty_spin
        self.penalty_terminal = penalty_terminal
        self.penalty_velocity = penalty_velocity

        self.done_dist_threshold = 100

        # === Costs: The following constants are used for cost calculation:
        self.vel_limit = 0.25  # [m/s]
        self.roll_pitch_limit = deg2rad(10)  # [rad]
        self.rpy_dot_limit = deg2rad(200)  # [rad/s]
        self.x_lim = 0.10
        self.y_lim = 0.10
        self.z_lim = 1.20

        # task specific parameters - init drone state
        init_xyz = np.array([0, 0, 1], dtype=np.float32)
        init_rpy = np.zeros(3)
        init_xyz_dot = np.zeros(3)
        init_rpy_dot = np.zeros(3)

        # Leo Trajectory
        self.path = generate_path(10)
        self.waypoints = self.path.circle()
        self.target_index = 0
        self.leo_target = self.waypoints[self.target_index]

        super(LandoBaseEnv, self).__init__(
            control_mode=control_mode,
            drone_model=drone_model,
            init_xyz=init_xyz,
            init_rpy=init_rpy,
            init_xyz_dot=init_xyz_dot,
            init_rpy_dot=init_rpy_dot,
            physics=physics,
            observation_noise=observation_noise,
            domain_randomization=domain_randomization,
            sim_freq=sim_freq,
            aggregate_phy_steps=aggregate_phy_steps,
            observation_frequency=observation_frequency,
            **kwargs
        )


    def _setup_task_specifics(self):
        """Initialize task specifics. Called by _setup_simulation()."""
        target_visual = self.bc.createVisualShape(
            self.bc.GEOM_SPHERE,
            radius=0.1,
            rgbaColor=[1, 0.1, 0.05, 0.99],
        )
        target_body_id = self.bc.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=target_visual,
            basePosition=self.leo_target)

        # === Draw reference circle
        for k in range(self.path.num_points):
            self.bc.createMultiBody(
                baseMass=0,
                baseCollisionShapeIndex=-1,
                baseVisualShapeIndex=self.bc.createVisualShape(
                    self.bc.GEOM_SPHERE,
                    radius=0.05,
                    rgbaColor=[0.0, 0.0, 1.0, 1],
                ),
                basePosition=self.waypoints[k]
            )
        
        # === Set camera position
        self.bc.resetDebugVisualizerCamera(
            cameraTargetPosition=(0.0, 0.0, 0.0),
            cameraDistance=1.5,
            cameraYaw=90,
            cameraPitch=-70
        )

    def compute_observation(self) -> np.ndarray:
        self.set_leo_actions()

        if self.observation_noise > 0:  # add noise only for positive values

            if self.iteration % self.obs_rate == 0:
                # === 100 Hz Part ===
                # update state information with 100 Hz (except for rpy_dot)
                # apply noise to perfect simulation state:
                xyz, vel, rpy, omega, acc = self.sensor_noise.add_noise(
                    pos=self.drone.xyz,
                    vel=self.drone.xyz_dot,
                    rot=self.drone.rpy,
                    omega=self.drone.rpy_dot,
                    acc=np.zeros(3),  # irrelevant
                    dt=1 / self.SIM_FREQ
                )
                quat = np.asarray(self.bc.getQuaternionFromEuler(rpy))
                error_to_ref = self.leo_xyz - xyz
                self.state = np.concatenate(
                    [xyz, quat, vel, omega, self.drone.last_action])
            else:
                # === 200 Hz Part ===
                # This part runs with >100Hz, re-use Kalman Filter values:
                xyz, quat, vel = self.state[0:3], self.state[3:7], self.state[
                                                                   7:10]
                error_to_ref = self.leo_xyz - xyz
                # read Gyro data with >100 Hz and add noise:
                omega = self.sensor_noise.add_noise_to_omega(
                    omega=self.drone.rpy_dot, dt=1 / self.SIM_FREQ)

            # apply low-pass filtering to gyro
            omega = self.gyro_lpf.apply(omega)
            obs = np.concatenate(
                [xyz, quat, vel, omega, error_to_ref])
        else:
            # no observation noise is applied
            error_to_ref = self.target_pos - self.drone.xyz
            obs = np.concatenate([self.drone.xyz, self.drone.quaternion,
                                  self.drone.xyz_dot, self.drone.rpy_dot,
                                  error_to_ref])
        return obs


    def compute_reward(self, action):
        # Determine penalties
        # spin_penalty = 1e-4 * np.linalg.norm(self.drone.rpy_dot)**2
        act_diff = action - self.last_action

        # todo sven: change reward function such that it is better comparable between PWM, Attitude Rate and ATTITUDE
        normed_clipped_a = 0.5 * (np.clip(action, -1, 1) + 1)

        penalty_action = self.penalty_action * np.linalg.norm(normed_clipped_a)
        penalty_action_rate = self.ARP * np.linalg.norm(act_diff)
        penalty_rpy = self.penalty_angle * np.linalg.norm(self.drone.rpy)
        penalty_spin = self.penalty_spin * np.linalg.norm(self.drone.rpy_dot)
        penalty_terminal = self.penalty_terminal if self.compute_done() else 0.
        penalty_velocity = self.penalty_velocity * np.linalg.norm(
            self.drone.xyz_dot)

        penalties = np.sum([penalty_rpy, penalty_action_rate, penalty_spin,
                            penalty_velocity, penalty_action, penalty_terminal])
        # L2 norm
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)
        reward = -dist - penalties
        return reward
    
    def set_leo_actions(self):
        # Get the current position and orientation
        self.leo_xyz, leo_w = self.bc.getBasePositionAndOrientation(self.leo)
        # Calculate Distance
        dist = np.linalg.norm(np.array(self.leo_target - self.leo_xyz))
        yaw = self.bc.getEulerFromQuaternion(leo_w)[2]

        if dist < 0.15:
            self.target_index += 1
            self.leo_target = self.waypoints[self.target_index]

        # Two DOF PID Controller
        wheelVelocities = differential_drive(self.leo_xyz, self.leo_target, yaw, (0.2, 1.0))
        wheelVelocities = wheelVelocities.tolist()

        for i in range(len(self.wheels)):
            self.bc.setJointMotorControl2(self.leo,
                                self.wheels[i],
                                self.bc.VELOCITY_CONTROL,
                                targetVelocity=wheelVelocities[i],
                                force=100)

    def task_specific_reset(self):
        # Note:
        #  - use copy since += operations are call by reference
        #  - init_xyz, init_xyz_dot, ... are set the SimOpt classes
        pos = self.init_xyz.copy()
        xyz_dot = self.init_xyz_dot.copy()
        rpy_dot = self.init_rpy_dot.copy()
        quat = self.init_quaternion.copy()

        self.bc.resetBasePositionAndOrientation(
            self.drone.body_unique_id,
            posObj=pos,
            ornObj=quat
        )

        R = np.array(self.bc.getMatrixFromQuaternion(quat)).reshape(3, 3)
        self.bc.resetBaseVelocity(
            self.drone.body_unique_id,
            linearVelocity=xyz_dot,
            # PyBullet assumes world frame, so local frame -> world frame
            angularVelocity=R.T @ rpy_dot
        )

        self.bc.resetBasePositionAndOrientation(
            self.leo,
            posObj=self.startPos,
            ornObj=self.bc.getQuaternionFromEuler(self.startRPY)
        )

    def compute_done(self) -> bool:
        """Compute end of episode if dist(drone - ref) > d."""
        dist = np.linalg.norm(self.drone.xyz - self.target_pos)
        done = True if dist > self.done_dist_threshold else False
        return done

    def compute_info(self) -> dict:
        # state = self.drone.get_state()
        c = 0.
        info = {'cost': c}
        return info

    def compute_potential(self) -> float:
        """Euclidean distance from current ron position to target position."""
        return np.linalg.norm(self.drone.xyz - self.target_pos)
    
    def get_reference_trajectory(self):
        raise NotImplementedError
    
    


""" ==================
        PWM control
    ==================
"""


class DroneCircleSimpleEnv(LandoBaseEnv):
    def __init__(self,
                 aggregate_phy_steps: int = 1,
                 control_mode='PWM',
                 **kwargs):
        super(DroneCircleSimpleEnv, self).__init__(
            control_mode=control_mode,
            drone_model='cf21x_sys_eq',
            physics='SimplePhysics',
            # use 100 Hz since no motor dynamics and PID is used
            sim_freq=100,
            aggregate_phy_steps=aggregate_phy_steps,
            **kwargs
        )


class LandoEnv(LandoBaseEnv):
    def __init__(self,
                 aggregate_phy_steps: int = 2,  # sub-steps used to calculate motor dynamics
                 control_mode: str = 'AttitudeRate',
                 **kwargs):
        super(LandoEnv, self).__init__(
            aggregate_phy_steps=aggregate_phy_steps,
            control_mode=control_mode,
            drone_model='cf21x_bullet',
            physics='PyBulletPhysics',
            observation_frequency=100,  # use 100Hz PWM control loop
            sim_freq=200,  # but step physics with 200Hz
            **kwargs
        )
