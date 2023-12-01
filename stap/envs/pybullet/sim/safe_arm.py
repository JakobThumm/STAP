import copy
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pybullet as p
from ctrlutils import eigen

import stap.envs.pybullet.sim.body as body
import stap.envs.pybullet.sim.shapes as shapes
from stap.controllers.failsafe_controller import FailsafeController
from stap.controllers.joint_space_control import joint_space_control
from stap.envs.pybullet.sim import articulated_body
from stap.envs.pybullet.sim.arm import Arm
from stap.envs.pybullet.sim.math import Pose


class SafeArm(Arm):
    """Arm controlled with operational space control."""

    def __init__(
        self,
        base_pos: Union[List[float], np.ndarray] = np.zeros(3),
        base_orientation: Union[List[float], np.ndarray] = np.array([0, 0, 0, 1]),
        shield_type: str = "SSM",
        robot_name: str = "panda",
        damping_ratio: float = 1.0,
        end_effector_id: int = 6,
        **kwargs,
    ):
        """Constructs the arm from yaml config.

        Args:
            kwargs: Keyword arguments for `Arm`.
        """
        super().__init__(**kwargs)
        self._redisgl = None
        self.end_effector_id = end_effector_id
        self._base_pos = base_pos
        self._base_orientation = base_orientation
        self._shield_type = shield_type
        self._shield = FailsafeController(
            init_qpos=self.ab.q,
            base_pos=base_pos,
            base_orientation=base_orientation,
            shield_type=shield_type,
            robot_name=robot_name,
            kp=self.pos_gains[0],
            damping_ratio=damping_ratio,
        )
        self._visualized_before = False
        self._human_sphere_viz = []
        self._robot_sphere_viz = []

    def reset(self, time: float) -> bool:
        """Disables torque control and resets the arm to the home configuration (bypassing simulation)."""
        super().reset()
        q, _ = self.get_joint_state(self.torque_joints)
        self._shield.reset(
            init_qpos=q,
            time=time,
            base_pos=self._base_pos,
            base_orientation=self._base_orientation,
            shield_type=self._shield_type,
        )
        return True

    def set_pose_goal(
        self,
        pos: Optional[np.ndarray] = None,
        quat: Optional[Union[eigen.Quaterniond, np.ndarray]] = None,
        pos_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        ori_gains: Optional[Union[Tuple[float, float], np.ndarray]] = None,
        timeout: Optional[float] = None,
    ) -> None:
        """Sets the pose goal.

        To actually control the robot, call `Arm.update_torques()`.

        Args:
            pos: Optional position. Maintains current position if None.
            quat: Optional quaternion. Maintains current orientation if None.
            pos_gains: Not used here.
            ori_gains: Not used here.
            timeout: Not used here.
        """
        q, _ = self.get_joint_state(self.torque_joints)
        # Calculate the desired joint position from the desired pose.
        desired_q_pos = p.calculateInverseKinematics(
            self.body_id, self.end_effector_id, pos, quat, physics_id=self.physics_id
        )
        # Set the desired joint position as new goal for sara-shield
        self._shield.set_goal(desired_q_pos)

    def update_torques(self, time: Optional[float] = None) -> articulated_body.ControlStatus:
        """Computes and applies the torques to control the articulated body to the goal set with `Arm.set_pose_goal().

        Returns:
            Controller status.
        """
        assert time is not None
        if not self._arm_state.torque_control:
            return articulated_body.ControlStatus.UNINITIALIZED

        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)

        if (self._q_limits[0] >= self.ab.q).any() or (self.ab.q >= self._q_limits[1]).any():
            return articulated_body.ControlStatus.ABORTED

        # Step sara-shield to get desired pos, vel, acc
        ddq_desired = self._shield.step(self.ab.q, self.ab.dq, time)
        # Calculate torques
        torques, success = joint_space_control(ab=self.ab, ddq_desired=ddq_desired, gravity_comp=True)

        self.apply_torques(torques)
        # Return positioned converged.
        # if status == 15:
        #     return articulated_body.ControlStatus.POS_CONVERGED

        ## TODO: Check if goal joint position is reached

        # Return timeout.
        self._arm_state.iter_timeout -= 1
        if self._arm_state.iter_timeout <= 0:
            return articulated_body.ControlStatus.TIMEOUT

        return articulated_body.ControlStatus.IN_PROGRESS

    def get_state(self) -> Dict[str, Any]:
        state = {
            "articulated_body": super().get_state(),
            "arm": copy.deepcopy(self._arm_state),
        }
        return state

    def set_state(self, state: Dict[str, Any]) -> None:
        super().set_state(state["articulated_body"])
        self._arm_state = copy.deepcopy(state["arm"])
        self.ab.q, self.ab.dq = self.get_joint_state(self.torque_joints)

    def visualize(self) -> None:
        robot_spheres = self._shield.get_robot_spheres()
        human_spheres = self._shield.get_human_spheres()
        if not self._visualized_before:
            self.init_visualization(len(robot_spheres), len(human_spheres))
        for i in range(len(robot_spheres)):
            self._robot_sphere_viz[i].set_pose(Pose(np.array(robot_spheres[i]), np.array([0, 0, 0, 1])))
        for i in range(len(human_spheres)):
            self._human_sphere_viz[i].set_pose(Pose(np.array(human_spheres[i]), np.array([0, 0, 0, 1])))

    def init_visualization(self, n_robot_spheres: int, n_human_spheres: int) -> None:
        assert not self._visualized_before
        human_color = [1, 0, 0, 0.5]
        robot_color = [0, 1, 0, 0.5]
        robot_radius = 0.1
        human_radius = 0.1
        for _ in range(n_robot_spheres):
            shape = shapes.Visual(shape=shapes.Sphere(mass=0, color=np.array(robot_color), radius=robot_radius))
            body_id = shapes.create_body(shapes=shape, physics_id=self.physics_id)
            self._robot_sphere_viz.append(body.Body(physics_id=self.physics_id, body_id=body_id))
        for _ in range(n_human_spheres):
            shape = shapes.Visual(shape=shapes.Sphere(mass=0, color=np.array(human_color), radius=human_radius))
            body_id = shapes.create_body(shapes=shape, physics_id=self.physics_id)
            self._human_sphere_viz.append(body.Body(physics_id=self.physics_id, body_id=body_id))
        self._visualized_before = True
