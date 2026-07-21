"""ROS2 entry point."""

from __future__ import annotations

import rclpy
from rclpy.executors import MultiThreadedExecutor

from sysnav.sysnav_node import SysNavNode


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SysNavNode()
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        executor.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
