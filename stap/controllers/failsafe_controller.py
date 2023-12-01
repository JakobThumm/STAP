"""This file describes the failsafe controller.

The failsafe controller allows safe operation of a robot in the vacinity of humans.

Owner:
    Jakob Thumm (JT)

Contributors:

Changelog:
    2.5.22 JT Formatted docstrings
"""

import os
from ast import List
from typing import List, Union

import numpy as np
from safety_shield_py import SafetyShield, ShieldType  # noqa: F401
from scipy.spatial.transform import Rotation

from stap.utils.macros import SIMULATION_TIME_STEP

# from matplotlib import pyplot as plt


class FailsafeController:
    """Controller for safely controlling robot arm in the vacinity of humans.

    Args:
        init_qpos (List[float]): Initial joint angles

        base_pos (List[float]): position of base [x, y, z]

        base_orientation (List[float]): orientation of base as quaternion [x, y, z, w]

        shield_type (str): Shield type to use. Valid options are: "OFF", "SSM", and "PFL"

        robot_name (str): Name of the robot to find config files.

        kp (float or Iterable of float): positional gain for determining desired torques based upon the joint pos error.
            Can be either be a scalar (same value for all action dims), or a list (specific values for each dim)

        damping_ratio (float or Iterable of float): used in conjunction with kp to determine the velocity gain for
            determining desired torques based upon the joint pos errors. Can be either be a scalar (same value for all
            action dims), or a list (specific values for each dim)

    Raises:
        AssertionError: [Invalid impedance mode]
    """

    def __init__(
        self,
        init_qpos: Union[List[float], np.ndarray],
        base_pos: Union[List[float], np.ndarray] = [0.0, 0.0, 0.0],
        base_orientation: Union[List[float], np.ndarray] = [0.0, 0.0, 0.0, 1.0],
        shield_type: str = "SSM",
        robot_name: str = "panda",
        kp: float = 50,
        damping_ratio: float = 1,
    ):
        self.kp = kp
        self.kd = 2 * np.sqrt(self.kp) * damping_ratio
        # Control dimension
        dir_path = os.path.dirname(os.path.realpath(__file__))
        rot = Rotation.from_quat(
            [
                base_orientation[0],
                base_orientation[1],
                base_orientation[2],
                base_orientation[3],
            ]
        )
        rpy = rot.as_euler("XYZ")

        # Unfortunately, all other native python enum functions seem to fail.
        self.shield_type = eval("ShieldType." + shield_type)

        self.safety_shield = SafetyShield(
            sample_time=SIMULATION_TIME_STEP,
            trajectory_config_file=(
                f"{dir_path}/../sara-shield/safety_shield/config/trajectory_parameters_{robot_name}.yaml"
            ),
            robot_config_file=f"{dir_path}/../sara-shield/safety_shield/config/robot_parameters_{robot_name}.yaml",
            mocap_config_file=dir_path + "/../sara-shield/safety_shield/config/mujoco_mocap.yaml",
            init_x=base_pos[0],
            init_y=base_pos[1],
            init_z=base_pos[2],
            init_roll=rpy[0],
            init_pitch=rpy[1],
            init_yaw=rpy[2],
            init_qpos=init_qpos,
            shield_type=self.shield_type,
        )
        self.desired_motion = self.safety_shield.step(0.0)
        # Place holder for dynamic variables
        self.command_vel = [0.0 for i in init_qpos]
        self.robot_cap_in = []
        self.human_cap_in = []
        self.control_dim = len(init_qpos)

    def reset(
        self,
        init_qpos: Union[List[float], np.ndarray],
        time: float,
        base_pos: Union[List[float], np.ndarray] = [0.0, 0.0, 0.0],
        base_orientation: Union[List[float], np.ndarray] = [0.0, 0.0, 0.0, 1.0],
        shield_type: str = "SSM",
    ):
        """Reset sara-shield.

        Args:
            init_qpos (list[float]): Initial joint angles
            base_pos (list[float]): position of base [x, y, z]
            base_orientation (list[float]): orientation of base as quaternion [x, y, z, w]
            shield_type (str): Shield type to use. Valid options are: "OFF", "SSM", and "PFL"
        """
        self.goal_qpos = None
        # Torques being outputted by the controller
        self.torques = None
        # Update flag to prevent redundant update calls
        self.new_update = True
        self.joint_pos = np.array(init_qpos)
        self.initial_joint = self.joint_pos
        # Control dimension
        rot = Rotation.from_quat(
            [
                base_orientation[0],
                base_orientation[1],
                base_orientation[2],
                base_orientation[3],
            ]
        )
        rpy = rot.as_euler("XYZ")
        # Unfortunately, all other native python enum functions seem to fail.
        self.shield_type = eval("ShieldType." + shield_type)
        self.safety_shield.reset(
            init_x=base_pos[0],
            init_y=base_pos[1],
            init_z=base_pos[2],
            init_roll=rpy[0],
            init_pitch=rpy[1],
            init_yaw=rpy[2],
            init_qpos=self.joint_pos,
            current_time=time,
            shield_type=self.shield_type,
        )

    def set_goal(self, desired_qpos: Union[List[float], np.ndarray]):
        """Set the joint position goal to move to.

        Args:
            desired_qpos (list[float]): Desired joint position absolute joint position.

        Raises:
            AssertionError: [Invalid action dimension size]
        """
        self.goal_qpos = np.array(desired_qpos)
        self.safety_shield.newLongTermTrajectory(self.goal_qpos, self.command_vel)

    def set_human_measurement(self, human_measurement: Union[List[List[float]], np.ndarray], time):
        """Set the human measurement of the safety shield.

        Args:
            human_measurement (list[list[float]]): List of human measurements [x, y, z]-joint positions.
                The order of joints is defined in the motion capture config file.
            time (float): Time of the human measurement
        """
        self.safety_shield.humanMeasurement(human_measurement, time)

    def step(
        self, joint_pos: Union[List[float], np.ndarray], joint_vel: Union[List[float], np.ndarray], time: float
    ) -> np.ndarray:
        """Calculate the torques required to reach the desired setpoint.

        Args:
            joint_pos (list[float]): Current joint position
            joint_vel (list[float]): Current joint velocity
            time (float): Current time
        Returns:
             np.array: Command accelerations
        """
        # Make sure goal has been set
        if self.goal_qpos is None:
            self.set_goal(np.zeros(self.control_dim))

        self.joint_pos = joint_pos
        self.joint_vel = joint_vel

        current_time = time
        self.desired_motion = self.safety_shield.step(current_time)
        desired_qpos = self.desired_motion.getAngle()
        desired_qvel = self.desired_motion.getVelocity()
        desided_qacc = self.desired_motion.getAcceleration()
        # Debug path following -> How well is the robot following the desired trajectory.
        # You can use this to tune your PID values

        # self.desired_pos_dbg[self.dbg_c] = desired_qpos
        # self.joint_pos_dbg[self.dbg_c] = self.joint_pos
        # self.dbg_c+=1

        # if self.dbg_c == 1000:
        #     fig, axs = plt.subplots(self.control_dim)
        #     for joint in range(self.control_dim):
        #         ax = axs[joint]
        #         ax.plot(np.arange(0, self.dbg_c), self.desired_pos_dbg[0:self.dbg_c, joint], label='desired pos')
        #         ax.plot(np.arange(0, self.dbg_c), self.joint_pos_dbg[0:self.dbg_c, joint], label='joint pos')
        #         # ax.set_xlabel("Step")
        #         # ax.set_ylabel("Angle [rad]")
        #         # ax.set_title(f"Joint {joint+1}")
        #         ax.grid()
        #     ax.legend()
        #     fig.suptitle("Joint Angles, PD+ Controller with Grav. Comp.")
        #     plt.show()
        #     self.dbg_c=0

        # torques = pos_err * kp + vel_err * kd
        position_error = desired_qpos - self.joint_pos
        vel_pos_error = desired_qvel - self.joint_vel
        ddq_control = (
            np.multiply(np.array(position_error), np.array(self.kp))
            + np.multiply(vel_pos_error, self.kd)
            + desided_qacc
        )

        # TODO: clip ddq_control

        return ddq_control

    def get_safety(self):
        """Return if the failsafe controller intervened in this step or not.

        Returns:
          bool: True: Safe, False: Unsafe
        """
        return self.safety_shield.getSafety()

    def get_robot_spheres(self) -> List[List[float]]:
        """Return the robot spheres in the correct format to plot them in pybullet.

        Returns:
            list[pos]
        """
        self.robot_cap_in = self.safety_shield.getRobotReachCapsules()
        robot_spheres = []
        for cap in self.robot_cap_in:
            robot_spheres.append(cap[0:3])
            robot_spheres.append(cap[3:6])
        return robot_spheres

    def get_human_spheres(self) -> List[List[float]]:
        """Return the human spheres in the correct format to plot them in pybullet.

        Returns:
            list[pos]
        """
        self.human_cap_in = self.safety_shield.getHumanReachCapsules()
        human_spheres = []
        for cap in self.human_cap_in:
            human_spheres.append(cap[0:3])
            human_spheres.append(cap[3:6])
        return human_spheres
