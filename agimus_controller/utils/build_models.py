import yaml
import pinocchio as pin
import example_robot_data
from hppfcl import Sphere, Box, Cylinder, Capsule
import numpy as np
import os
from panda_torque_mpc_pywrap.panda_torque_mpc_pywrap import ReduceCollisionModel


class ObstacleParamsParser:
    def __init__(self, yaml_file, collision_model):
        with open(yaml_file, "r") as file:
            self.params = yaml.safe_load(file)
        self.collision_model = collision_model

    def add_collisions(self):
        obs_idx = 1

        while f"obstacle{obs_idx}" in self.params:
            obstacle_name = f"obstacle{obs_idx}"
            obstacle_config = self.params[obstacle_name]

            type_ = obstacle_config.get("type")
            translation_vect = obstacle_config.get("translation", [])

            if not translation_vect:
                print(
                    f"No obstacle translation declared for the obstacle named: {obstacle_name}"
                )
                return

            translation = np.array(translation_vect).reshape(3)

            rotation_vect = obstacle_config.get("rotation", [])
            if not rotation_vect:
                print(
                    f"No obstacle rotation declared for the obstacle named: {obstacle_name}"
                )
                return

            rotation = np.array(rotation_vect).reshape(4)

            geometry = None
            if type_ == "sphere":
                radius = obstacle_config.get("radius")
                if radius:
                    geometry = Sphere(radius)
                else:
                    print("No dimension or wrong dimensions in the obstacle config.")
                    return
            elif type_ == "box":
                x = obstacle_config.get("x")
                y = obstacle_config.get("y")
                z = obstacle_config.get("z")
                if x and y and z:
                    geometry = Box(x, y, z)
                else:
                    print("No dimension or wrong dimensions in the obstacle config.")
                    return
            elif type_ == "cylinder":
                radius = obstacle_config.get("radius")
                half_length = obstacle_config.get("halfLength")
                if radius and half_length:
                    geometry = Cylinder(radius, half_length)
                else:
                    print("No dimension or wrong dimensions in the obstacle config.")
                    return
            elif type_ == "capsule":
                radius = obstacle_config.get("radius")
                half_length = obstacle_config.get("halfLength")
                if radius and half_length:
                    geometry = Capsule(radius, half_length)
                else:
                    print("No dimension or wrong dimensions in the obstacle config.")
                    return
            else:
                print("No type or wrong type in the obstacle config.")
                return
            obstacle_pose = pin.XYZQUATToSE3(np.concatenate([translation, rotation]))
            obstacle_pose.translation = translation
            obstacle = pin.GeometryObject(obstacle_name, 0, 0, geometry, obstacle_pose)
            self.collision_model.addGeometryObject(obstacle)
            obs_idx += 1

        collision_pairs = self.params.get("collision_pairs", [])
        if collision_pairs:
            for pair in collision_pairs:
                if len(pair) == 2:
                    name_object1, name_object2 = pair
                    if self.collision_model.existGeometryName(
                        name_object1
                    ) and self.collision_model.existGeometryName(name_object2):
                        self.add_collision_pair(name_object1, name_object2)
                    else:
                        print(
                            f"Object {name_object1} or {name_object2} does not exist in the collision model."
                        )
                else:
                    print(f"Invalid collision pair: {pair}")
        else:
            print("No collision pairs.")

    def add_collision_pair(self, name_object1, name_object2):
        object1_id = self.collision_model.getGeometryId(name_object1)
        object2_id = self.collision_model.getGeometryId(name_object2)
        if object1_id is not None and object2_id is not None:
            self.collision_model.addCollisionPair(
                pin.CollisionPair(object1_id, object2_id)
            )
        else:
            print(
                f"Object ID not found for collision pair: {object1_id} and {object2_id}"
            )


def get_robot_model(robot, urdf_path, srdf_path):
    locked_joints = [
        robot.model.getJointId("panda_finger_joint1"),
        robot.model.getJointId("panda_finger_joint2"),
    ]

    model = pin.Model()
    pin.buildModelFromUrdf(urdf_path, model)
    pin.loadReferenceConfigurations(model, srdf_path, False)
    q0 = model.referenceConfigurations["default"]
    return pin.buildReducedModel(model, locked_joints, q0)


def get_collision_model(rmodel, urdf_path, yaml_file):
    collision_model = pin.buildGeomFromUrdf(rmodel, urdf_path, pin.COLLISION)
    reduce_collision_model = ReduceCollisionModel()
    collision_model = reduce_collision_model.reduce_capsules_robot(collision_model)
    parser = ObstacleParamsParser(yaml_file, collision_model)
    parser.add_collisions()
    return parser.collision_model


if __name__ == "__main__":
    robot = example_robot_data.load("panda")
    urdf_path = "/home/gepetto/ros_ws/src/agimus_controller/urdf/robot.urdf"
    srdf_path = "/home/gepetto/ros_ws/src/agimus_controller/srdf/demo.srdf"
    rmodel = get_robot_model(robot, urdf_path, srdf_path)
    yaml_file = os.path.join("../main", "param.yaml")
    cmodel = get_collision_model(rmodel, urdf_path)
