#!/usr/bin/env python3
import rospy
import numpy as np
from copy import deepcopy
import time
from threading import Lock
from std_msgs.msg import Duration, Header
import pinocchio as pin
import example_robot_data
from linear_feedback_controller_msgs.msg import Control, Sensor

from agimus_controller.utils.ros_np_multiarray import to_multiarray_f64
from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP


def get_robot_model(robot):
    locked_joints = [
        robot.model.getJointId("panda_finger_joint1"),
        robot.model.getJointId("panda_finger_joint2"),
    ]

    urdf_path = (
        "/home/gepetto/ros_ws/src/agimus_controller/agimus_controller_ros/robot.urdf"
    )
    srdf_path = (
        "/home/gepetto/ros_ws/src/agimus_controller/agimus_controller_ros/demo.srdf"
    )

    model = pin.Model()
    pin.buildModelFromUrdf(urdf_path, model)
    pin.loadReferenceConfigurations(model, srdf_path, False)
    q0 = model.referenceConfigurations["default"]
    return pin.buildReducedModel(model, locked_joints, q0)


class HppAgimusController:
    def __init__(self) -> None:
        self.dt = 1e-2
        self.q_goal = [1.9542, -1.1679, -2.0741, -1.8046, 0.0149, 2.1971, 2.0056]
        self.horizon_size = 100

        robot = example_robot_data.load("panda")
        self.rmodel = get_robot_model(robot)
        self.pandawrapper = PandaWrapper(auto_col=True)
        _, self.cmodel, self.vmodel = self.pandawrapper.create_robot()
        self.ee_frame_name = self.pandawrapper.get_ee_frame_name()
        self.hpp_interface = HppInterface()
        self.ocp = OCPCrocoHPP(self.rmodel, self.cmodel, use_constraints=False)
        self.ocp.set_weights(10**4, 1, 10**-3, 0)

        self.rate = rospy.Rate(100, reset=True)
        self.mutex = Lock()
        self.sensor_msg = Sensor()
        self.control_msg = Control()
        self.ocp_solve_time = Duration()
        self.x0 = np.zeros(self.rmodel.nq + self.rmodel.nv)
        self.x_guess = np.zeros(self.rmodel.nq + self.rmodel.nv)
        self.u_guess = np.zeros(self.rmodel.nv)
        self.state_subscriber = rospy.Subscriber(
            "robot_sensors",
            Sensor,
            self.sensor_callback,
        )
        self.control_publisher = rospy.Publisher(
            "motion_server_control", Control, queue_size=1
        )
        self.ocp_solve_time_pub = rospy.Publisher(
            "ocp_solve_time", Duration, queue_size=1
        )
        self.start_time = 0.0
        self.first_solve = False
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True

    def sensor_callback(self, sensor_msg):
        with self.mutex:
            self.sensor_msg = deepcopy(sensor_msg)
            if not self.first_robot_sensor_msg_received:
                self.first_robot_sensor_msg_received = True

    def wait_first_sensor_msg(self):
        wait_for_input = True
        while not rospy.is_shutdown() and wait_for_input:
            wait_for_input = (
                not self.first_robot_sensor_msg_received
                or not self.first_pose_ref_msg_received
            )
            if wait_for_input:
                rospy.loginfo_throttle(3, "Waiting until we receive a sensor message.")
                with self.mutex:
                    sensor_msg = deepcopy(self.sensor_msg)
                    self.start_time = sensor_msg.header.stamp.to_sec()
            rospy.loginfo_once("Start controller")
            self.rate.sleep()
        return wait_for_input

    def plan_and_first_solve(self):
        sensor_msg = self.get_sensor_msg()

        # Plan
        q_init = [*sensor_msg.joint_state.position]
        self.ps = self.hpp_interface.set_panda_planning(q_init, self.q_goal)
        whole_x_plan, whole_a_plan, _ = self.hpp_interface.get_hpp_x_a_planning(
            self.dt,
            self.rmodel.nq,
            self.ps.client.problem.getPath(self.ps.numberPaths() - 1),
        )

        # First solve
        self.mpc = MPC(self.ocp, whole_x_plan, whole_a_plan, self.rmodel, self.cmodel)
        self.x_plan = self.mpc.whole_x_plan[: self.horizon_size, :]
        self.a_plan = self.mpc.whole_a_plan[: self.horizon_size, :]
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        self.mpc.mpc_first_step(self.x_plan, self.a_plan, x0, self.horizon_size)
        np.save("ros_x_plan.npy", whole_x_plan)
        np.save("ros_xs.npy", self.mpc.ocp.solver.xs)
        self.next_node_idx = self.horizon_size
        whole_traj_T = whole_x_plan.shape[0]
        self.mpc_xs = np.zeros([whole_traj_T, self.horizon_size, 2 * self.rmodel.nq])
        self.mpc_us = np.zeros([whole_traj_T, self.horizon_size - 1, self.rmodel.nq])
        self.state_refs = np.zeros([whole_traj_T, 2 * self.rmodel.nq])
        self.translation_refs = np.zeros([whole_traj_T, 3])
        self.control_refs = np.zeros([whole_traj_T, self.rmodel.nq])

        self.mpc_xs[0, :, :] = np.array(self.mpc.ocp.solver.xs)
        self.mpc_us[0, :, :] = np.array(self.mpc.ocp.solver.us)
        x_ref, p_ref, u_ref = self.mpc.get_reference()
        self.state_refs[0, :] = x_ref
        self.translation_refs[0, :] = p_ref
        self.control_refs[0, :] = u_ref

        _, u, k = self.mpc.get_mpc_output()
        self.control_msg.header = Header()
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.feedback_gain = to_multiarray_f64(k)
        self.control_msg.feedforward = to_multiarray_f64(u)
        self.control_msg.initial_state = sensor_msg
        self.control_publisher.publish(self.control_msg)

    def solve_and_send(self):
        sensor_msg = self.get_sensor_msg()
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )

        new_x_ref = self.mpc.whole_x_plan[self.next_node_idx, :]
        new_a_ref = self.mpc.whole_a_plan[self.next_node_idx, :]

        mpc_start_time = time.time()
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref)
        mpc_duration = time.time() - mpc_start_time
        rospy.loginfo_throttle(1, "mpc_duration = %s", str(mpc_duration))
        if self.next_node_idx < self.mpc.whole_x_plan.shape[0] - 1:
            self.next_node_idx += 1
            self.mpc_xs[self.next_node_idx - self.horizon_size, :, :] = np.array(
                self.mpc.ocp.solver.xs
            )
            self.mpc_us[self.next_node_idx - self.horizon_size, :, :] = np.array(
                self.mpc.ocp.solver.us
            )
            x_ref, p_ref, u_ref = self.mpc.get_reference()
            self.state_refs[self.next_node_idx - self.horizon_size, :] = x_ref
            self.translation_refs[self.next_node_idx - self.horizon_size, :] = p_ref
            self.control_refs[self.next_node_idx - self.horizon_size, :] = u_ref
        if self.next_node_idx == self.mpc.whole_x_plan.shape[0] - 2:
            np.save("mpc_xs.npy", self.mpc_xs)
            np.save("mpc_us.npy", self.mpc_us)
            np.save("state_refs.npy", self.state_refs)
            np.save("translation_refs.npy", self.translation_refs)
            np.save("control_refs.npy", self.control_refs)
        _, u, k = self.mpc.get_mpc_output()

        self.control_msg.header = Header()
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.feedback_gain = to_multiarray_f64(k)
        self.control_msg.feedforward = to_multiarray_f64(u)
        self.control_msg.initial_state = sensor_msg
        self.control_publisher.publish(self.control_msg)

    def get_sensor_msg(self):
        with self.mutex:
            sensor_msg = deepcopy(self.sensor_msg)
        return sensor_msg

    def run(self):
        self.wait_first_sensor_msg()
        self.plan_and_first_solve()
        self.rate.sleep()
        while not rospy.is_shutdown():
            start_compute_time = rospy.Time.now()
            self.solve_and_send()
            self.ocp_solve_time.data = rospy.Time.now() - start_compute_time
            self.ocp_solve_time_pub.publish(self.ocp_solve_time)
            self.rate.sleep()


def crocco_motion_server_node():
    rospy.init_node("croccodyl_motion_server_node_py", anonymous=True)
    node = HppAgimusController()
    node.run()


if __name__ == "__main__":
    try:
        crocco_motion_server_node()
    except rospy.ROSInterruptException:
        pass
