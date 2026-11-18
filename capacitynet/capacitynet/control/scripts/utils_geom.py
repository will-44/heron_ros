#!/usr/bin/env python3
"""Pure geometry helpers: quaternion/vector math with no ROS node, topic, or
domain (marker/grasp/state) dependency — plain functions on
geometry_msgs.msg.Pose/Quaternion-shaped objects (anything with .x/.y/.z/.w
or .position/.orientation attributes), reusable from any node.
"""

from geometry_msgs.msg import Pose


def quat_to_axis(q, axis: str):
    """Return local `axis` ('x', 'y', or 'z') of orientation q, expressed in
    the parent frame, as an (x, y, z) unit vector."""
    n = (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w) ** 0.5
    if n == 0.0:
        n = 1.0
    qx, qy, qz, qw = q.x / n, q.y / n, q.z / n, q.w / n
    if axis == 'x':
        return (
            1.0 - 2.0 * (qy * qy + qz * qz),
            2.0 * (qx * qy + qw * qz),
            2.0 * (qx * qz - qw * qy),
        )
    if axis == 'y':
        return (
            2.0 * (qx * qy - qw * qz),
            1.0 - 2.0 * (qx * qx + qz * qz),
            2.0 * (qy * qz + qw * qx),
        )
    if axis == 'z':
        return (
            2.0 * (qx * qz + qw * qy),
            2.0 * (qy * qz - qw * qx),
            1.0 - 2.0 * (qx * qx + qy * qy),
        )
    raise ValueError(f"Unknown axis '{axis}' (expected 'x', 'y', or 'z')")


def quat_mul(a, b):
    """Hamilton product a (x) b of two (x, y, z, w) quaternion tuples."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return (
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    )


def unit_vector_between(from_pose, to_pose):
    """Unit vector from one Pose's position to another's, in their shared
    frame. Returns None if the two coincide."""
    dx = to_pose.position.x - from_pose.position.x
    dy = to_pose.position.y - from_pose.position.y
    dz = to_pose.position.z - from_pose.position.z
    norm = (dx * dx + dy * dy + dz * dz) ** 0.5
    if norm == 0.0:
        return None
    return (dx / norm, dy / norm, dz / norm)


def lift_pose(base_pose, distance: float):
    """Vertical lift from base_pose along the frame's own Z axis (not any
    local/marker Z) — same orientation as base_pose. Returns a new
    geometry_msgs.msg.Pose."""
    p = Pose()
    p.position.x = base_pose.position.x
    p.position.y = base_pose.position.y
    p.position.z = base_pose.position.z + distance
    p.orientation = base_pose.orientation
    return p
