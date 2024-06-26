import time
import os
import example_robot_data
from agimus_controller.hpp_interface import HppInterface
from agimus_controller.mpc import MPC
from agimus_controller.utils.plots import MPCPlots
from agimus_controller.utils.build_models import get_robot_model, get_collision_model

from agimus_controller.utils.wrapper_panda import PandaWrapper
from agimus_controller.ocps.ocp_croco_hpp import OCPCrocoHPP

if __name__ == "__main__":
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
    q_init = [
        0.13082259440720514,
        -1.150735366655217,
        -0.6975751204881672,
        -2.835918304210108,
        -0.02303564961006244,
        2.51523530644841,
        0.33466451573454664,
    ]
    q_goal = [1.9542, -1.1679, -2.0741, -1.8046, 0.0149, 2.1971, 2.0056]

    hpp_interface = HppInterface()
    ps = hpp_interface.get_panda_planner(q_init, q_goal)
    x_plan, a_plan, whole_traj_T = hpp_interface.get_hpp_plan(
        1e-2, 7, ps.client.problem.getPath(ps.numberPaths() - 1)
    )
    ocp = OCPCrocoHPP(rmodel, cmodel, use_constraints=False)

    mpc = MPC(ocp, x_plan, a_plan, rmodel, cmodel)
    start = time.time()
    mpc.ocp.set_weights(10**4, 1, 10**-3, 0)
    mpc.simulate_mpc(100, save_predictions=True)
    end = time.time()
    u_plan = mpc.ocp.get_u_plan(x_plan, a_plan)
    mpc_plots = MPCPlots(
        mpc.croco_xs,
        mpc.croco_us,
        x_plan,
        u_plan,
        rmodel,
        mpc.ocp.DT,
        ee_frame_name=ee_frame_name,
        v=hpp_interface.get_viewer(),
    )
