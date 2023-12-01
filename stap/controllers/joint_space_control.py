from typing import Optional, Tuple

import numpy as np
import spatialdyn as dyn


def joint_space_control(
    ab: dyn.ArticulatedBody,
    ddq_desired: np.ndarray,
    gravity_comp: bool = True,
    integration_step: Optional[float] = None,
) -> Tuple[np.ndarray, bool]:
    """
    Computes command torques for controlling the robot to a given pose using
    joint space control.

    Args:
        ab: Articulated body with `ab.q` and `ab.dq` set to the current state.

        ddq_desired: Desired joint acceleration.

        gravity_comp: Compensate for gravity.

        integration_step: Optional integration time step. If set to a positive
            number, this function will update `ab` with the expected position
            and velocity of the robot after applying the returned command
            torques `tau` for the given timsetep. This is helpful if the robot
            only supports position/velocity control, not torque control.

    Returns:
        2-tuple (`tau`, `converged`), where `tau` is an [N] array of torques (N
        is the dof of the given articulated body), and `converged` is a boolean
        that indicates whether the position and orientation convergence criteria
        are satisfied, if given.
    """
    tau_cmd = dyn.inverse_dynamics(ab, ddq_desired, centrifugal_coriolis=False, gravity=gravity_comp)

    # Apply command torques to update ab.q, ab.dq.
    if integration_step is not None:
        dyn.integrate(ab, tau_cmd, integration_step)

    return tau_cmd
