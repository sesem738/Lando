r""""Environment utilities for Project Phoenix.

Author:     Sven Gronauer
Created:    17.05.2021

Updates:
    14.07.2021: Added low-pass filter and sensor noise (according to USC)
    05.08.2021: Added transformation function for quaternions, rotation matrices
    14.04.2022: Add depreciation warnings for future clean-ups
"""
import os
import numpy as np
from math import exp


def rad2deg(x):
    r"""Converts radians to degrees."""
    return 180*x/np.pi


def deg2rad(x):
    r"""Converts degrees to radians."""
    return np.pi*x/180


def get_assets_path() -> str:
    r""" Returns the path to the files located in envs/data."""
    data_path = os.path.join(os.path.dirname(__file__), 'assets')
    return data_path


def get_quaternion_from_euler(rpy) -> np.ndarray:
    r"""Get quaternion [x, y, z, w] from Euler angles RPY [rad].

    See:
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles
    """
    assert rpy.shape == (3, )
    roll, pitch, yaw = rpy
    halfYaw = yaw * 0.5
    halfPitch = pitch * 0.5
    halfRoll = roll * 0.5
    cosYaw = np.cos(halfYaw)
    sinYaw = np.sin(halfYaw)
    cosPitch = np.cos(halfPitch)
    sinPitch = np.sin(halfPitch)
    cosRoll = np.cos(halfRoll)
    sinRoll = np.sin(halfRoll)

    x = sinRoll * cosPitch * cosYaw - cosRoll * sinPitch * sinYaw
    y = cosRoll * sinPitch * cosYaw + sinRoll * cosPitch * sinYaw
    z = cosRoll * cosPitch * sinYaw - sinRoll * sinPitch * cosYaw
    w = cosRoll * cosPitch * cosYaw + sinRoll * sinPitch * sinYaw

    # Note: PyBullet uses the order: [X,Y,Z,W]
    return np.array([x, y, z, w])


class LowPassFilter:
    def __init__(self, gain: float, time_constant: float, sample_time: float):
        """

        Parameters
        ----------
        gain: K
        time_constant: T
            The filter time constant. Remember: ~4T until step response
        sample_time: T_s
            Typically 1/sim_freq
        """
        self._x = 0
        self.K = gain
        self.T = time_constant
        self.T_s = sample_time

    def apply(self, u):
        ratio = self.T_s / self.T
        self._x = (1 - ratio) * self._x + self.K * ratio * u
        return self._x

    def set(self, x):
        self._x = x


class OUNoise:
    """Ornstein–Uhlenbeck process"""

    def __init__(self, action_dimension, mu=0, theta=0.15, sigma=0.3):
        """
        @param: mu: mean of noise
        @param: theta: stabilization coeff (i.e. noise return to mean)
        @param: sigma: noise scale coeff
        """
        self.action_dimension = action_dimension
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dimension) * self.mu
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dimension) * self.mu

    def noise(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

def wrap_to_pi(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def differential_drive(current_pos, target_pos, current_heading, p_gain=(0.5, 10),
                       ang_thresh=0.0174):
    # Robot parameters
    wheel_base = 0.447
    wheel_radius = 0.065

    # Position and heading difference
    dx = target_pos[0] - current_pos[0]
    dy = target_pos[1] - current_pos[1]
    dtheta = wrap_to_pi(np.arctan2(dy, dx) - (current_heading))
    # print(rad2deg(dtheta))
    if (dtheta < ang_thresh) & (dtheta > -ang_thresh):
        dtheta = 0.0

    # Calculate linear and angular velocities
    linear_velocities = np.sqrt(dx ** 2 + dy ** 2) * p_gain[0]
    angular_velocities = dtheta * p_gain[1]

    # Calculate individual wheel speeds for a differential drive robot
    left_speeds = (2 * linear_velocities - angular_velocities * wheel_base) / (2 * wheel_radius)
    right_speeds = (2 * linear_velocities + angular_velocities * wheel_base) / (2 * wheel_radius)

    # Scale speeds if necessary
    # max_wheel_speed = max(np.max(np.abs(left_speeds)), np.max(np.abs(right_speeds)))
    # if max_wheel_speed > max_speed:
    #     scaling_factor = max_speed / max_wheel_speed
    #     left_speeds *= scaling_factor
    #     right_speeds *= scaling_factor

    return np.stack((left_speeds, left_speeds, right_speeds, right_speeds), axis=-1)

class generate_path:
    def __init__(self, num_points: int):
        self.num_points = num_points

    def lemniscate(self):
        a=np.sqrt(2)
        num_points = 20

        # Generate theta values
        theta = np.linspace(-np.pi / 2, 3 * np.pi / 2, num_points)
        # Calculate x and y coordinates for the lemniscate curve
        x = a * np.cos(theta) / (np.sin(theta) ** 2 + 1)
        y = a * np.cos(theta) * np.sin(theta) / (np.sin(theta) ** 2 + 1)
        z = y*0

        return np.column_stack((x, y, z))
    
    def circle(self):
        radius = 2
        center = (0,0)
        theta = np.linspace(0, 2 * np.pi, self.num_points, endpoint=False)  # create angles for points on circle
        a = radius * np.cos(theta) + center[0]  # calculate x coordinates
        b = radius * np.sin(theta) + center[1]  # calculate y coordinates
        z = b*0
        return np.stack((a, b, z), axis=1)  # stack x and y into NumPy array

    
    def square(self):

        num_points = 20
        # Calculate side length based on number of points (assuming perfect square)
        side_length = int(np.sqrt(num_points))

        # Pre-allocate the array for the trajectory
        trajectory = np.zeros((num_points, 2))

        # Define starting and ending indices for each side
        start_idx = {
            "right": 0, "up": side_length, "left": 2 * side_length - 1, "down": 3 * side_length - 2
        }
        end_idx = {
            "right": side_length - 1, "up": 2 * side_length - 2, "left": 3 * side_length - 3, "down": num_points - 1
        }

        # Loop through each side and fill the trajectory array
        for side, (start, end) in start_idx.items():
            # Determine direction (increasing/decreasing)
            direction = 1 if side in ("right", "up") else -1

            # Create NumPy arrays for x and y coordinates for the current side
            x = np.arange(start, end + direction, direction)
            y = np.full_like(x, start if side in ("right", "left") else end)

            # Stack x and y into the trajectory array for the current side
            trajectory[start:end + 1] = np.stack((x, y), axis=1)

        return trajectory
