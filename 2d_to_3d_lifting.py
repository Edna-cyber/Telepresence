import numpy as np

def lift_2d_to_3d(points_2d):
    # Define the transformation matrix to lift 2D points to 3D
    transformation_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])

    # Add a third coordinate with value 1 to each 2D point
    points_3d = np.hstack((points_2d, np.ones((points_2d.shape[0], 1))))

    # Apply the transformation matrix to lift the points to 3D
    points_lifted = np.dot(points_3d, transformation_matrix.T)

    return points_lifted

# Example usage
points_2d = np.array([[1, 2], [3, 4], [5, 6]])
lifted_points = lift_2d_to_3d(points_2d)
print(lifted_points)