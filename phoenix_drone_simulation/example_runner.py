import numpy as np
# Plot the trajectory (requires matplotlib)
import matplotlib.pyplot as plt


def circle_trajectory_numpy(num_points, center=(0, 0), radius=1):
  """
  Generates a NumPy array representing a circular trajectory with a given number of points.

  Args:
    num_points: The number of points in the trajectory.
    center: A 2-element list or tuple representing the center coordinates of the circle (default: (0, 0)).
    radius: The radius of the circle (default: 1).

  Returns:
    A NumPy array with shape (num_points, 2), where each row represents a point's x and y coordinates.
  """

  theta = np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # create angles for points on circle
  a = radius * np.cos(theta) + center[0]  # calculate x coordinates
  b = radius * np.sin(theta) + center[1]  # calculate y coordinates
  return np.stack((a, b), axis=1)  # stack x and y into NumPy array

# Example usage
trajectory = circle_trajectory_numpy(16)

# Print the first few points
print(trajectory[:5])


plt.plot(trajectory[:, 0], trajectory[:, 1])
plt.xlabel("X")
plt.ylabel("Y")
# plt.gca().set_aspect('equal')  # make the plot appear circular
plt.grid(True)
plt.show()
