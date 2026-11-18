#!/usr/bin/env python3
"""Analytic IK + manipulability for a 6R arm, via IKDH (https://github.com/Lab-CORO/IKDH).

ROS-free (see reachability_engine.py / workspace_evaluation.py for the same convention);
`workspace_reachability_node.py` is the only caller. Requires ``pip install ikdh``.

IKDH's pip wheel (1.2.0) exposes IK/FK (`Solver`, `forward_kin`) but not the
jacobian/manipulability methods described in its README -- those are C++/web-only
in this version. Each bundled robot YAML does however ship a precomputed analytic
Jacobian (as sympy-parseable strings in q1..q6, degenerate trig shorthand s1=sin(q1)
etc.), so manipulability is computed here instead: lambdify that expression once
per robot, evaluate sqrt(det(J @ J.T)) per sample. Cross-checked against a
finite-difference Jacobian of `forward_kin` at several configurations (max element
error ~1e-6) before trusting it.
"""

import os

import ikdh
import numpy as np
from scipy.spatial.transform import Rotation
import sympy as sp
import yaml


def pose_to_transform(position, orientation):
    """geometry_msgs Point/Quaternion (metres) -> 4x4 transform for IKDH.

    IKDH's DH tables (a/d) are in metres, matching ROS geometry_msgs directly --
    no unit conversion needed (unlike `pose_from_xyzrpw`, a RoboDK-pose convenience
    that takes mm and isn't used here).
    """
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_quat(
        [orientation.x, orientation.y, orientation.z, orientation.w]).as_matrix()
    transform[:3, 3] = [position.x, position.y, position.z]
    return transform


class IkdhModel:
    """Wraps one IKDH robot preset: all-solutions IK and analytic manipulability."""

    def __init__(self, robot_yaml='doosan_robotics_m1013.yaml'):
        self._robot = ikdh.load_robot(robot_yaml)
        self._solver = ikdh.Solver(self._robot.dh, self._robot.limits)
        self._jacobian_fn = self._load_jacobian_fn(robot_yaml)

    def inverse_kinematics(self, position, orientation):
        """All analytic IK solutions reaching `pose`, each a (6,) array in degrees.

        Up to 8 solutions for a 6R spherical-wrist arm (shoulder/elbow/wrist
        branches); fewer if some branches are out of joint limits or unreachable.
        """
        transform = pose_to_transform(position, orientation)
        return self._solver.solve(transform)

    def manipulability(self, q_deg):
        """Yoshikawa manipulability index sqrt(det(J J^T)) at joint config q_deg (degrees)."""
        jacobian = np.array(self._jacobian_fn(*np.radians(q_deg)), dtype=float)
        return float(np.sqrt(max(np.linalg.det(jacobian @ jacobian.T), 0.0)))

    @staticmethod
    def _load_jacobian_fn(robot_yaml):
        path = robot_yaml
        if not os.path.isabs(path) and not os.path.exists(path):
            path = os.path.join(ikdh.robots_dir(), os.path.basename(path))
        with open(path) as f:
            rows = yaml.safe_load(f)['jacobian']['rows']

        joint_syms = sp.symbols('q1 q2 q3 q4 q5 q6')
        local_syms = {'pi': sp.pi, 'sin': sp.sin, 'cos': sp.cos}
        for i, q in enumerate(joint_syms, start=1):
            local_syms[f'q{i}'] = q
            local_syms[f's{i}'] = sp.sin(q)
            local_syms[f'c{i}'] = sp.cos(q)

        jacobian = sp.Matrix(
            [[sp.sympify(cell, locals=local_syms) for cell in row] for row in rows])
        return sp.lambdify(joint_syms, jacobian, modules='numpy')
