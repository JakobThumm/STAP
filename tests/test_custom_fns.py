import numpy as np
import torch

# from stap.planners.custom_fns import axis_angle_to_matrix, matrix_to_axis_angle
from stap.utils.transformation_utils import axis_angle_to_matrix, matrix_to_axis_angle


def test_axis_angle_to_matrix():
    # Test known values
    axis_angle = torch.tensor([[np.pi, 0.0, 0.0]])  # 180 degrees around the x-axis
    expected_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    assert torch.allclose(axis_angle_to_matrix(axis_angle[:, :3]), expected_matrix, atol=1e-5)
    # Test zero angle
    axis_angle = torch.tensor([[0.0, 0.0, 0.0, 0.0]])
    expected_matrix = torch.eye(3)
    assert torch.allclose(axis_angle_to_matrix(axis_angle[:, :3]), expected_matrix, atol=1e-5)
    # Test random values and conversion back to axis-angle
    for _ in range(100):
        random_axis_angle = torch.rand(1, 3) * 2 - 1  # Random axis
        random_axis_angle = (
            random_axis_angle / torch.linalg.norm(random_axis_angle) * (torch.rand(1) * 2 * np.pi)
        )  # Random angle
        rotation_matrix = axis_angle_to_matrix(random_axis_angle)
        recovered_axis_angle = matrix_to_axis_angle(rotation_matrix)
        # Normalize original and recovered for a fair comparison
        random_axis_angle_normalized = random_axis_angle / torch.linalg.norm(random_axis_angle)
        recovered_normalized = recovered_axis_angle / torch.linalg.norm(recovered_axis_angle)
        assert torch.allclose(random_axis_angle_normalized, recovered_normalized, atol=1e-5) or torch.allclose(
            random_axis_angle_normalized, -recovered_normalized, atol=1e-5
        )


def test_axis_angle_to_matrix_multi_dim():
    # Test known values
    axis_angle = torch.tensor([[np.pi, 0.0, 0.0], [0.0, 0.0, 0.0]])
    matrices = axis_angle_to_matrix(axis_angle)
    expected_matrix_1 = torch.tensor([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    expected_matrix_2 = torch.eye(3)
    assert torch.allclose(matrices[0], expected_matrix_1, atol=1e-5)
    assert torch.allclose(matrices[1], expected_matrix_2, atol=1e-5)


def test_matrix_to_axis_angle_multi_dim():
    # Test known values
    matrix = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]], [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]]
    )
    axis_angles = matrix_to_axis_angle(matrix)
    expected_axis_angle_1 = torch.tensor([[np.pi, 0.0, 0.0]])
    expected_axis_angle_2 = torch.tensor([[0.0, 0.0, 0.0]])
    assert torch.allclose(axis_angles[0], expected_axis_angle_1, atol=1e-5)
    assert torch.allclose(axis_angles[1], expected_axis_angle_2, atol=1e-5)


if __name__ == "__main__":
    print("Running tests for custom_fns.py")
    test_matrix_to_axis_angle_multi_dim()
