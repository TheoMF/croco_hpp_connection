import rospy
import numpy as np
from copy import deepcopy
from threading import Lock
from std_msgs.msg import Duration, Header
from linear_feedback_controller_msgs.msg import Control, Sensor

from agimus_controller.utils.ros_np_multiarray import to_multiarray_f64
from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.mpc import MPC
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.trajectory_point import PointAttribute

from agimus_controller_ros.hpp_subscriber import HPPSubscriber


class AgimusControllerNodeParameters:
    def __init__(self) -> None:
        self.rate = rospy.get_param("~rate", 100)
        self.horizon_size = rospy.get_param("~horizon_size", 100)


class AgimusControllerNode:
    def __init__(self) -> None:
        rospy.loginfo("Load parameters")
        self.params = AgimusControllerNodeParameters()

        self.pandawrapper = PandaWrapper(auto_col=False)
        self.rmodel, self.cmodel, self.vmodel = self.pandawrapper.create_robot()
        self.ee_frame_name = self.pandawrapper.get_ee_frame_name()
        self.ocp = OCPCrocoHPP(self.rmodel, self.cmodel, use_constraints=False)
        self.ocp.set_weights(10**4, 1, 10**-3, 0)
        self.mpc = MPC(self.ocp, None, None, self.rmodel, self.cmodel)

        rospy.loginfo("Create the rate object.")
        self.rate = rospy.Rate(self.params.rate)
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
        self.hpp_subscriber = HPPSubscriber()
        self.start_time = 0.0
        self.first_solve = False
        self.first_robot_sensor_msg_received = False
        self.first_pose_ref_msg_received = True
        self.point_attributes = [PointAttribute.Q]

    def sensor_callback(self, sensor_msg):
        with self.mutex:
            self.sensor_msg = deepcopy(sensor_msg)
            if not self.first_robot_sensor_msg_received:
                self.first_robot_sensor_msg_received = True

    def get_sensor_msg(self):
        with self.mutex:
            sensor_msg = deepcopy(self.sensor_msg)
        return sensor_msg

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

    # def wait_twice_control_horizon_from_plan(self):
    # for _ in range(self.params.horizon_size * 2):
    #   tp = self.hpp_subscriber.get_trajectory_point()
    # self.buffer.append

    def first_solve(self):
        sensor_msg = self.get_sensor_msg()

        # Get 1 horizon from the plan.
        horizon_points = self.traj_buffer.get_points(
            self.params.horizon_size, self.point_attributes
        )
        self.x_plan = np.zeros([self.params.horizon_size, self.mpc.nx])
        self.a_plan = np.zeros([self.params.horizon_size, self.mpc.nv])
        for idx_point, point in enumerate(horizon_points):
            self.x_plan[idx_point, :] = point.get_x_as_q_v()
            self.a_plan[idx_point, :] = point.a

        # First solve
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        self.mpc.mpc_first_step(self.x_plan, self.a_plan, x0, self.params.horizon_size)

    def solve_and_send(self):
        sensor_msg = self.get_sensor_msg()
        x0 = np.concatenate(
            [sensor_msg.joint_state.position, sensor_msg.joint_state.velocity]
        )
        point = self.traj_buffer.get_points(1, self.point_attributes)[0]
        new_x_ref = point.get_x_as_q_v()
        new_a_ref = point.a

        mpc_duration = rospy.Time.now()
        self.mpc.mpc_step(x0, new_x_ref, new_a_ref)
        _, u, k = self.mpc.get_mpc_output()
        mpc_duration = rospy.Time.now() - mpc_duration
        rospy.loginfo_throttle(1, "mpc_duration = %s", str(mpc_duration))

        self.control_msg.header = Header()
        self.control_msg.header.stamp = rospy.Time.now()
        self.control_msg.feedback_gain = to_multiarray_f64(k)
        self.control_msg.feedforward = to_multiarray_f64(u)
        self.control_msg.initial_state = sensor_msg
        self.control_publisher.publish(self.control_msg)

    def run(self):
        self.wait_first_sensor_msg()
        self.wait_twice_control_horizon_from_plan()
        self.first_solve()
        input("Press Enter to continue...")
        while not rospy.is_shutdown():
            start_compute_time = rospy.Time.now()
            self.solve_and_send()
            self.ocp_solve_time.data = rospy.Time.now() - start_compute_time
            self.ocp_solve_time_pub.publish(self.ocp_solve_time)
            self.rate.sleep()
