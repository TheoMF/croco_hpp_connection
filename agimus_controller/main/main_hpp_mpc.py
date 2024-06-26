#!/usr/bin/env python
from math import pi
import time
import example_robot_data

from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP
from agimus_controller.mpc import MPC
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.utils.plots import MPCPlots


if __name__ == "__main__":
    robot = example_robot_data.load("ur3")
    rmodel = robot.model
    hpp_interface = HppInterface()
    q_init = [pi / 6, -pi / 2, pi / 2, 0, 0, 0, -0.2, 0, 0.02, 0, 0, 0, 1]
    hpp_interface.set_ur3_problem_solver(q_init)
    ps, viewer = hpp_interface.get_problem_solver_and_viewer()
    hpp_path = ps.client.basic.problem.getPath(hpp_interface.ps.numberPaths() - 1)
    x_plan, a_plan, whole_traj_T = hpp_interface.get_hpp_x_a_planning(1e-2, 6, hpp_path)
    ocp = OCPCrocoHPP(rmodel=rmodel, cmodel=None, use_constraints=False)
    mpc = MPC(ocp, x_plan, a_plan, rmodel)
    start = time.time()
    mpc.ocp.set_weights(10**4, 1, 10**-3, 0)
    mpc.simulate_mpc(T=100, save_predictions=False)
    end = time.time()
    u_plan = mpc.ocp.get_u_plan(x_plan, a_plan)
    mpc_plots = MPCPlots(
        croco_xs=mpc.croco_xs,
        croco_us=mpc.croco_us,
        whole_x_plan=x_plan,
        whole_u_plan=u_plan,
        rmodel=rmodel,
        DT=mpc.ocp.DT,
        ee_frame_name="wrist_3_joint",
        viewer=viewer,
    )
    print("mpc duration ", end - start)
