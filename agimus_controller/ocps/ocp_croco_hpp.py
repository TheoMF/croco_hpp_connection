import crocoddyl
import pinocchio as pin
import numpy as np
import mim_solvers

from colmpc import ResidualDistanceCollision

from agimus_controller.utils.pin_utils import (
    get_ee_pose_from_configuration,
    get_last_joint,
)


class OCPCrocoHPP:
    def __init__(
        self, rmodel: pin.Model, cmodel: pin.GeometryModel = None, use_constraints=False
    ) -> None:
        """Class to define the OCP linked witha HPP generated trajectory.

        Args:
            rmodel (pin.Model): Pinocchio model of the robot.
            cmodel (pin.GeometryModel): Pinocchio geometry model of the robot. Must have been convexified for the collisions to work.

        Raises:
            Exception: Unkown robot.
        """
        # Robot models
        self._rmodel = rmodel
        self._cmodel = cmodel

        # Data of the model
        self._rdata = self._rmodel.createData()

        # Obtaining the last joint name and joint id
        self._last_joint_name, self._last_joint_id, self._last_joint_frame_id = (
            get_last_joint(self._rmodel)
        )

        # Weights of the different costs
        self._weight_x_reg = 1e-1
        self._weight_u_reg = 1e-4
        self._weight_ee_placement = 1e6
        self._weight_vel_reg = 0

        # Using the constraints ?
        self.use_constraints = use_constraints

        # Safety marging for the collisions
        self._safety_margin = 1e-3

        # Creating the state and actuation models
        self.state = crocoddyl.StateMultibody(self._rmodel)
        self.actuation = crocoddyl.ActuationModelFull(self.state)

        # Setting up variables necessary for the OCP
        self.DT = 1e-2  # Time step of the OCP
        self.nq = self._rmodel.nq  # Number of joints of the robot
        self.nv = self._rmodel.nv  # Dimension of the speed of the robot

        # Creating HPP variables
        self.x_plan = None
        self.a_plan = None
        self.u_plan = None
        self.T = None

        # Creating the running and terminal models
        self.running_models = None
        self.terminal_model = None

        # Solver used for the OCP
        self.solver = None

    def get_u_plan(
        self, x_plan: np.ndarray, a_plan: np.ndarray, using_gravity=False
    ) -> np.ndarray:
        """Return the reference of control u_plan that compensates gravity.

        Args:
            x_plan (np.ndarray): Array of (q,v) for each node, describing the trajectory found by the planner.
            a_plan (np.ndarray): Array of (dv/dt) for each node, describing the trajectory found by the planner.
            using_gravity (bool, optional): . Defaults to False.

        Returns:
            np.ndarray: Array of (u) for each node, found by either RNEA of Generalized Gravity.
        """
        u_plan = np.zeros([x_plan.shape[0] - 1, self.nv])
        if using_gravity:  ### TODO for Théo
            pass
            # for x in self.hpp_paths[path_idx].x_plan:
            #     pin.computeGeneralizedGravity(
            #         self._rmodel,
            #         self.robot_data,
            #         x[: self.nq],
            #     )
            #     u_plan.append(self.robot_data.g.copy())
        else:
            for idx in range(x_plan.shape[0] - 1):
                x = x_plan[idx, :]
                a = a_plan[idx, :]
                tau = self.get_inverse_dynamic_control(x, a)
                u_plan[idx, :] = tau[: self.nq]
        return u_plan

    def set_weights(
        self,
        weight_ee_placement: float,
        weight_x_reg: float,
        weight_u_reg: float,
        weight_vel_reg: float,
    ):
        """Set weights of the ocp.

        Args:
            weight_ee_placement (float): Weight of the placement of the end effector with regards to the target.
            weight_x_reg (float): Weight of the state regularization.
            weight_u_reg (float): Weight of the control regularization.
            weight_vel_reg (float): Weight of the velocity regularization.
        """
        self._weight_ee_placement = weight_ee_placement
        self._weight_x_reg = weight_x_reg
        self._weight_u_reg = weight_u_reg
        self._weight_vel_reg = weight_vel_reg

    def set_models(self, x_plan: np.ndarray, a_plan: np.ndarray):
        """Set running models and terminal model for the ocp.

        Args:
            x_plan (np.ndarray): Array of (q,v) for each node, describing the trajectory found by the planner.
            a_plan (np.ndarray): Array of (dv/dt) for each node, describing the trajectory found by the planner.
        """
        self.x_plan = x_plan
        self.a_plan = a_plan
        self.T = x_plan.shape[0]
        self.u_plan = self.get_u_plan(x_plan, a_plan)
        goal_placement_residual = self.get_placement_residual(
            self.x_plan[-1, : self.nq]
        )
        self.set_running_models()
        self.set_terminal_model(goal_placement_residual)

    def set_running_models(self):
        """Set running models based on state and acceleration reference trajectory."""

        running_models = []
        for idx in range(self.T - 1):
            running_cost_model = crocoddyl.CostModelSum(self.state)
            x_ref = self.x_plan[idx, :]
            x_residual = self.get_state_residual(x_ref)
            u_residual = self.get_control_residual(self.u_plan[idx, :])
            frame_velocity_residual = self.get_velocity_residual(self._last_joint_name)
            placemment_residual = self.get_placement_residual(x_ref[: self.nq])
            running_cost_model.addCost("xReg", x_residual, self._weight_x_reg)
            running_cost_model.addCost("uReg", u_residual, self._weight_u_reg)
            running_cost_model.addCost(
                "velReg", frame_velocity_residual, self._weight_vel_reg
            )
            running_cost_model.addCost(
                "gripperPose", placemment_residual, 0
            )  # useful for mpc to reset ocp

            if self.use_constraints:
                constraints = self.get_constraints()

                running_models.append(
                    crocoddyl.IntegratedActionModelEuler(
                        crocoddyl.DifferentialActionModelFreeFwdDynamics(
                            self.state, self.actuation, running_cost_model, constraints
                        ),
                        self.DT,
                    )
                )
            else:
                running_models.append(
                    crocoddyl.IntegratedActionModelEuler(
                        crocoddyl.DifferentialActionModelFreeFwdDynamics(
                            self.state, self.actuation, running_cost_model
                        ),
                        self.DT,
                    )
                )
        self.running_models = running_models
        return self.running_models

    def get_constraints(self):
        constraint_model_manager = crocoddyl.ConstraintModelManager(self.state, self.nq)
        if len(self._cmodel.collisionPairs) != 0:
            for col_idx in range(len(self._cmodel.collisionPairs)):
                collision_constraint = self._get_collision_constraint(
                    col_idx, self._safety_margin
                )
                # Adding the constraint to the constraint manager
                constraint_model_manager.addConstraint(
                    "col_term_" + str(col_idx), collision_constraint
                )
        return constraint_model_manager

    def set_terminal_model(self, goal_placement_residual):
        """Set terminal model."""
        if self.use_constraints:
            last_model = self.get_terminal_model_with_constraints(
                goal_placement_residual, self.x_plan[-1, :], self.u_plan[-1, :]
            )
        else:
            last_model = self.get_terminal_model_without_constraints(
                goal_placement_residual, self.x_plan[-1, :], self.u_plan[-1, :]
            )
        self.terminal_model = last_model

    def get_terminal_model_without_constraints(
        self, goal_placement_residual, x_ref: np.ndarray, u_plan: np.ndarray
    ):
        """Return last model without constraints."""
        running_cost_model = crocoddyl.CostModelSum(self.state)
        running_cost_model.addCost(
            "gripperPose", goal_placement_residual, self._weight_ee_placement
        )
        vel_cost = self.get_velocity_residual(self._last_joint_name)
        if np.linalg.norm(x_ref[self.nq :]) < 1e-9:
            running_cost_model.addCost("velReg", vel_cost, self._weight_ee_placement)
        else:
            running_cost_model.addCost("velReg", vel_cost, 0)
        x_residual = self.get_state_residual(x_ref)
        running_cost_model.addCost("xReg", x_residual, 0)

        u_reg_cost = self.get_control_residual(u_plan)
        running_cost_model.addCost("uReg", u_reg_cost, 0)
        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state, self.actuation, running_cost_model
            ),
            self.DT,
        )

    def get_terminal_model_with_constraints(
        self, placement_residual, x_ref: np.ndarray, u_plan: np.ndarray
    ):
        """Return terminal model with constraints for mim_solvers."""
        terminal_cost_model = crocoddyl.CostModelSum(self.state)
        x_residual = self.get_state_residual(x_ref)
        u_reg_cost = self.get_control_residual(u_plan)
        vel_cost = self.get_velocity_residual(self._last_joint_name)
        terminal_cost_model.addCost("xReg", x_residual, 0)
        terminal_cost_model.addCost("velReg", vel_cost, 0)
        terminal_cost_model.addCost("gripperPose", placement_residual, 0)
        terminal_cost_model.addCost("uReg", u_reg_cost, 0)

        # Add torque constraint
        # Joints torque limits given by the manufactor
        constraints = self.get_constraints()

        return crocoddyl.IntegratedActionModelEuler(
            crocoddyl.DifferentialActionModelFreeFwdDynamics(
                self.state,
                self.actuation,
                terminal_cost_model,
                constraints,
            ),
            self.DT,
        )

    def _get_collision_constraint(
        self, col_idx: int, safety_margin: float
    ) -> "crocoddyl.ConstraintModelResidual":
        """Returns the collision constraint that will be in the constraint model manager.

        Args:
            col_idx (int): index of the collision pair.
            safety_margin (float): Lower bound of the constraint, ie the safety margin.

        Returns:
            _type_: _description_
        """
        obstacleDistanceResidual = ResidualDistanceCollision(
            self.state, 7, self._cmodel, col_idx
        )

        # Creating the inequality constraint
        constraint = crocoddyl.ConstraintModelResidual(
            self.state,
            obstacleDistanceResidual,
            np.array([safety_margin]),
            np.array([np.inf]),
        )
        return constraint

    def get_placement_residual(self, q):
        """Return placement residual to the last position of the sub path."""
        target = get_ee_pose_from_configuration(
            self._rmodel, self._rdata, self._last_joint_frame_id, q
        )
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ResidualModelFramePlacement(
                self.state, self._last_joint_frame_id, target
            ),
        )

    def get_velocity_residual(self, joint_name):
        """Return velocity residual of desired joint."""
        vref = pin.Motion.Zero()
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ResidualModelFrameVelocity(
                self.state,
                self._rmodel.getFrameId(joint_name),
                vref,
                pin.WORLD,
            ),
        )

    def get_control_residual(self, uref):
        """Return control residual with uref the control reference."""
        return crocoddyl.CostModelResidual(
            self.state, crocoddyl.ResidualModelControl(self.state, uref)
        )

    def get_state_residual(self, xref):
        """Return state residual with xref the state reference."""
        return crocoddyl.CostModelResidual(
            self.state,  # x_reg_weights,
            crocoddyl.ResidualModelState(self.state, xref, self.actuation.nu),
        )

    def get_xlimit_residual(self):
        """Return state limit residual."""
        return crocoddyl.CostModelResidual(
            self.state,
            crocoddyl.ActivationModelQuadraticBarrier(
                crocoddyl.ActivationBounds(self.state.lb, self.state.ub)
            ),
            crocoddyl.ResidualModelState(
                self.state,
                np.array([0] * (self.nq + self.nv)),
                self.actuation.nu,
            ),
        )

    def get_translation_residual(self):
        """Return translation residual to the last position of the sub path."""
        q_final = self.x_plan[-1, : self.nq]
        target = get_ee_pose_from_configuration(
            self._rmodel, self._rdata, self._last_joint_frame_id, q_final
        )
        return crocoddyl.ResidualModelFrameTranslation(
            self.state,
            self._last_joint_id,
            target.translation,
        )

    def get_inverse_dynamic_control(self, x, a):
        """Return inverse dynamic control for a given state and acceleration."""
        return pin.rnea(self._rmodel, self._rdata, x[: self.nq], x[self.nq :], a).copy()

    def update_cost(self, model, new_model, cost_name, update_weight=True):
        """Update model's cost reference and weight by copying new_model's cost."""
        model.differential.costs.costs[
            cost_name
        ].cost.residual.reference = new_model.differential.costs.costs[
            cost_name
        ].cost.residual.reference.copy()
        if update_weight:
            new_weight = new_model.differential.costs.costs[cost_name].weight
            model.differential.costs.costs[cost_name].weight = new_weight

    def update_model(self, model, new_model, update_weight):
        """update model's costs by copying new_model's costs."""
        self.update_cost(model, new_model, "xReg", update_weight)
        self.update_cost(model, new_model, "gripperPose", update_weight)
        self.update_cost(model, new_model, "velReg", update_weight)
        self.update_cost(model, new_model, "uReg", update_weight)

    def reset_ocp(self, x, x_ref: np.ndarray, u_plan: np.ndarray):
        """Reset ocp problem using next reference in state and control."""
        self.solver.problem.x0 = x
        runningModels = list(self.solver.problem.runningModels)
        for node_idx in range(len(runningModels) - 1):
            self.update_model(
                runningModels[node_idx], runningModels[node_idx + 1], False
            )
        self.update_model(runningModels[-1], self.solver.problem.terminalModel, False)
        if self.use_constraints:
            terminal_model = self.get_terminal_model_with_constraints(
                self.get_placement_residual(x_ref[: self.nq]), x_ref, u_plan
            )
        else:
            terminal_model = self.get_terminal_model_without_constraints(
                self.get_placement_residual(x_ref[: self.nq]), x_ref, u_plan
            )
        self.update_model(self.solver.problem.terminalModel, terminal_model, True)

    def build_ocp_from_plannif(self, x_plan, a_plan, x0):
        """Set models based on state and acceleration planning, create crocoddyl problem from it."""
        self.set_models(x_plan, a_plan)
        return crocoddyl.ShootingProblem(x0, self.running_models, self.terminal_model)

    def run_solver(self, problem, xs_init, us_init, max_iter, set_callback=False):
        """
        Run FDDP or CSQP solver
        problem : crocoddyl ocp problem.
        xs_init : xs warm start.
        us_init : us warm start.
        max_iter : max number of iteration for the solver
        set_callback : activate solver callback
        """
        # Creating the solver for this OC problem, defining a logger
        if self.use_constraints:
            solver = mim_solvers.SolverCSQP(problem)
            solver.use_filter_line_search = True
            solver.termination_tolerance = 1e-3
            solver.max_qp_iters = 100
        else:
            solver = crocoddyl.SolverFDDP(problem)
            solver.use_filter_line_search = True
            solver.termination_tolerance = 1e-3
        if set_callback:
            solver.setCallbacks([crocoddyl.CallbackVerbose()])
        solver.solve(xs_init, us_init, max_iter)
        self.solver = solver
