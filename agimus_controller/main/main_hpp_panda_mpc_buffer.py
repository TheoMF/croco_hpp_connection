from math import pi
import numpy as np
import os
import example_robot_data
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.utils.plots import MPCPlots
from agimus_controller.utils.build_models import get_robot_model, get_collision_model

from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.trajectory_buffer import TrajectoryBuffer
from agimus_controller.trajectory_point import TrajectoryPoint, PointAttribute

if __name__ == "__main__":
    nq = 7
    nv = 7
    pandawrapper = PandaWrapper(auto_col=True)
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    urdf_path = os.path.join(current_dir_path, "../../urdf/robot.urdf")
    srdf_path = os.path.join(current_dir_path, "../../srdf/demo.srdf")
    yaml_path = os.path.join(current_dir_path, "../../config/param.yaml")
    robot = example_robot_data.load("panda")
    rmodel = get_robot_model(robot, urdf_path, srdf_path)
    cmodel = get_collision_model(rmodel, urdf_path, yaml_path)
    rmodel, cmodel, vmodel = pandawrapper.create_robot()
    ee_frame_name = pandawrapper.get_ee_frame_name()
    hpp_interface = HppInterface()
    ps = hpp_interface.get_panda_planner()
    q_init = [pi / 6, -pi / 2, pi / 2, 0, 0, 0, -0.2, 0, 0.02, 0, 0, 0, 1]
    whole_x_plan, whole_a_plan, whole_traj_T = hpp_interface.get_hpp_plan(
        1e-2, 7, ps.client.problem.getPath(ps.numberPaths() - 1)
    )
    ocp = OCPCrocoHPP(rmodel, cmodel, use_constraints=False)
    mpc = MPC(ocp, whole_x_plan, whole_a_plan, rmodel, cmodel)
    mpc.ocp.set_weights(10**4, 1, 10**-3, 0)

    point_attributes = [PointAttribute.Q]  # PointAttribute.V, PointAttribute.A
    traj_buffer = TrajectoryBuffer()
    first_point = TrajectoryPoint(nq=nq, nv=nv)
    first_point.q = whole_x_plan[0, :nq]
    first_point.v = whole_x_plan[0, nq:]
    first_point.a = whole_a_plan[0, :]
    traj_buffer.initialize(first_point)
    whole_traj_T = whole_x_plan.shape[0]
    T = 100
    mpc_xs = np.zeros([whole_traj_T, 2 * nq])
    mpc_us = np.zeros([whole_traj_T - 1, nq])
    x = first_point.get_x_as_q_v()
    mpc_xs[0, :] = x
    first_step_done = False
    for idx in range(1, whole_traj_T + 2 * T):
        traj_idx = min(idx, whole_x_plan.shape[0] - 1)
        point = TrajectoryPoint(nq=nq, nv=nv)
        point.q = whole_x_plan[traj_idx, :nq]
        point.v = whole_x_plan[traj_idx, nq:]
        point.a = whole_a_plan[traj_idx, :]
        traj_buffer.add_trajectory_point(point)
        if not first_step_done:
            buffer_size = traj_buffer.get_size(point_attributes)
            if buffer_size < 2 * T:
                print(f"buffer size is {buffer_size}, waiting for {2*T} points.")
            else:
                horizon_points = traj_buffer.get_points(T, point_attributes)
                x_plan = np.zeros([T, mpc.nx])
                a_plan = np.zeros([T, mpc.nv])
                for idx_point, point in enumerate(horizon_points):
                    x_plan[idx_point, :] = point.get_x_as_q_v()
                    a_plan[idx_point, :] = point.a
                next_node_idx = T
                x, u = mpc.mpc_first_step(x_plan, a_plan, x, T)
                mpc_xs[idx - 2 * T, :] = x
                mpc_us[idx - 2 * T - 1, :] = u
                first_step_done = True
        else:
            point = traj_buffer.get_points(1, point_attributes)[0]
            new_x_ref = point.get_x_as_q_v()
            new_a_ref = point.a
            x, u = mpc.mpc_step(x, new_x_ref, new_a_ref)
            mpc_xs[idx - 2 * T, :] = x
            mpc_us[idx - 2 * T - 1, :] = u

    u_plan = mpc.ocp.get_u_plan(whole_x_plan, whole_a_plan)
    mpc_plots = MPCPlots(
        mpc_xs,
        mpc_us,
        whole_x_plan,
        u_plan,
        rmodel,
        mpc.ocp.DT,
        ee_frame_name=ee_frame_name,
        v=hpp_interface.planner._v,
        ball_init_pose=[-0.2, 0, 0.02, 0, 0, 0, 1],
    )
    mpc_plots.plot_traj()
