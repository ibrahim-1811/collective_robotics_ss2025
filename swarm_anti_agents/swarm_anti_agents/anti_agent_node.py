import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from math import hypot
import random

class AntiAgent(Node):
    def __init__(self):
        super().__init__('anti_agent_node')
        self.total_robots = 7
        self.cluster_radius = 1.0
        self.cluster_threshold = 3
        self.command_interval = 2.0
        self.robot_positions = {}

        for i in range(self.total_robots):
            robot_ns = f'/robot_{i}'
            self.create_subscription(Odometry, f'{robot_ns}/odom', self.odom_callback_factory(i), 10)

        self.cmd_publishers = {
            i: self.create_publisher(Twist, f'/robot_{i}/cmd_vel', 10) for i in range(self.total_robots)
        }

        self.create_timer(self.command_interval, self.evaluate_clusters)
        self.get_logger().info("Anti-agent node initialized.")

    def odom_callback_factory(self, robot_id):
        def callback(msg):
            pos = msg.pose.pose.position
            self.robot_positions[robot_id] = (pos.x, pos.y)
        return callback

    def evaluate_clusters(self):
        for target_id, (x, y) in self.robot_positions.items():
            neighbor_count = sum(
                1 for other_id, (ox, oy) in self.robot_positions.items()
                if other_id != target_id and self.euclidean_distance(x, y, ox, oy) < self.cluster_radius
            )
            if neighbor_count >= self.cluster_threshold:
                self.send_leave_command(target_id)

    def send_leave_command(self, robot_id):
        twist = Twist()
        twist.linear.x = random.uniform(-0.2, 0.2)
        twist.angular.z = random.uniform(-1.0, 1.0)
        self.cmd_publishers[robot_id].publish(twist)
        self.get_logger().info(f"Anti-agent: Sent leave command to robot_{robot_id}")

    @staticmethod
    def euclidean_distance(x1, y1, x2, y2):
        return hypot(x1 - x2, y1 - y2)

def main(args=None):
    rclpy.init(args=args)
    node = AntiAgent()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()