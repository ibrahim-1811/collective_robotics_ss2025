#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from sensor_msgs.msg import LaserScan
import math
import random
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

# Helper function for distance
def euclidean_distance(pos1: Point, pos2: Point):
    return math.sqrt((pos1.x - pos2.x)**2 + (pos1.y - pos2.y)**2)

class RobotControllerNode(Node):
    def __init__(self):
        super().__init__('robot_controller')

        # === Parameter Declarations ===
        self.declare_parameter('robot_id', 'robot_0')
        self.declare_parameter('total_robots', 7)
        self.declare_parameter('designated_anti_agent_id', 'robot_0')
        self.declare_parameter('robot_prefix_for_others', 'robot_')
        
        self.declare_parameter('linear_speed', 0.3)
        self.declare_parameter('angular_speed', 0.5)
        self.declare_parameter('attraction_radius', 3.0)
        self.declare_parameter('clustering_distance', 0.8) # Target distance for forming a cluster
        self.declare_parameter('anti_agent_view_radius', 3.0) 
        self.declare_parameter('min_cluster_size_for_disperse', 3)
        self.declare_parameter('leave_command_duration_s', 10.0)
        self.declare_parameter('update_frequency', 10.0)

        # Frontal obstacle avoidance
        self.declare_parameter('obstacle_danger_threshold_m', 0.35) 
        self.declare_parameter('obstacle_turn_speed_rad_s', 0.75)  
        self.declare_parameter('frontal_scan_angle_rad', 0.3) 

        # Side obstacle adjustment
        self.declare_parameter('side_check_angle_rad', math.pi / 3.0) 
        self.declare_parameter('side_check_sector_width_rad', 0.35) 
        self.declare_parameter('side_caution_distance_m', 0.55)       
        self.declare_parameter('side_avoidance_turn_factor', 0.6)   
        self.declare_parameter('side_avoidance_speed_factor', 0.6)  

        # Recovery behavior
        self.declare_parameter('stuck_avoidance_threshold_cycles', 25) 
        self.declare_parameter('recovery_backup_duration_s', 1.8)    
        self.declare_parameter('recovery_turn_duration_s', 2.2)      
        self.declare_parameter('recovery_backup_speed_mps', -0.12)   
        self.declare_parameter('recovery_turn_speed_rads', 0.75)   
        
        # New parameters for refined clustering behavior
        self.declare_parameter('approach_speed_reduction_factor', 0.6) # Reduce speed more significantly
        self.declare_parameter('min_approach_speed_mps', 0.05)       
        # Personal space: if another robot is this close, consider self as clustered/stop
        self.declare_parameter('personal_space_stop_distance_m', 0.5) 


        # === Parameter Retrieval (ensure all new ones are retrieved) ===
        self.robot_id = self.get_parameter('robot_id').get_parameter_value().string_value
        self.total_robots = self.get_parameter('total_robots').get_parameter_value().integer_value
        designated_anti_agent_id_str = self.get_parameter('designated_anti_agent_id').get_parameter_value().string_value
        self.robot_prefix_for_others = self.get_parameter('robot_prefix_for_others').get_parameter_value().string_value
        self.is_anti_agent = (self.robot_id == designated_anti_agent_id_str)

        self.linear_speed = self.get_parameter('linear_speed').get_parameter_value().double_value
        self.angular_speed = self.get_parameter('angular_speed').get_parameter_value().double_value
        self.attraction_radius = self.get_parameter('attraction_radius').get_parameter_value().double_value
        self.clustering_distance = self.get_parameter('clustering_distance').get_parameter_value().double_value
        self.anti_agent_view_radius = self.get_parameter('anti_agent_view_radius').get_parameter_value().double_value
        self.min_cluster_size_for_disperse = self.get_parameter('min_cluster_size_for_disperse').get_parameter_value().integer_value
        self.leave_command_duration_s = self.get_parameter('leave_command_duration_s').get_parameter_value().double_value
        self.update_frequency_val = self.get_parameter('update_frequency').get_parameter_value().double_value
        self.obstacle_danger_threshold_m = self.get_parameter('obstacle_danger_threshold_m').get_parameter_value().double_value
        self.obstacle_turn_speed_rad_s = self.get_parameter('obstacle_turn_speed_rad_s').get_parameter_value().double_value
        self.frontal_scan_angle_rad = self.get_parameter('frontal_scan_angle_rad').get_parameter_value().double_value
        self.SIDE_CHECK_ANGLE_RAD = self.get_parameter('side_check_angle_rad').get_parameter_value().double_value
        self.SIDE_CHECK_SECTOR_WIDTH_RAD = self.get_parameter('side_check_sector_width_rad').get_parameter_value().double_value
        self.SIDE_CAUTION_DISTANCE_M = self.get_parameter('side_caution_distance_m').get_parameter_value().double_value
        self.SIDE_AVOIDANCE_TURN_FACTOR = self.get_parameter('side_avoidance_turn_factor').get_parameter_value().double_value
        self.SIDE_AVOIDANCE_SPEED_FACTOR = self.get_parameter('side_avoidance_speed_factor').get_parameter_value().double_value
        self.STUCK_AVOIDANCE_THRESHOLD_CYCLES = self.get_parameter('stuck_avoidance_threshold_cycles').get_parameter_value().integer_value
        self.RECOVERY_BACKUP_DURATION_S = self.get_parameter('recovery_backup_duration_s').get_parameter_value().double_value
        self.RECOVERY_TURN_DURATION_S = self.get_parameter('recovery_turn_duration_s').get_parameter_value().double_value
        self.RECOVERY_BACKUP_SPEED_MPS = self.get_parameter('recovery_backup_speed_mps').get_parameter_value().double_value
        self.RECOVERY_TURN_SPEED_RADS = self.get_parameter('recovery_turn_speed_rads').get_parameter_value().double_value

        self.APPROACH_SPEED_REDUCTION_FACTOR = self.get_parameter('approach_speed_reduction_factor').get_parameter_value().double_value
        self.MIN_APPROACH_SPEED_MPS = self.get_parameter('min_approach_speed_mps').get_parameter_value().double_value
        self.PERSONAL_SPACE_STOP_DISTANCE_M = self.get_parameter('personal_space_stop_distance_m').get_parameter_value().double_value
        
        # Sanity check for personal space vs clustering distance
        if self.PERSONAL_SPACE_STOP_DISTANCE_M >= self.clustering_distance:
            self.PERSONAL_SPACE_STOP_DISTANCE_M = self.clustering_distance * 0.7 # Ensure it's smaller
            self.get_logger().warn(
                f"Personal space stop distance was too large, adjusted to: {self.PERSONAL_SPACE_STOP_DISTANCE_M:.2f} m"
            )
        
        self.get_logger().info(f"Robot ID: {self.robot_id}, Anti-agent: {self.is_anti_agent}")
        self.get_logger().info(f"Clustering dist: {self.clustering_distance:.2f}, Personal space: {self.PERSONAL_SPACE_STOP_DISTANCE_M:.2f}")


        # State variables, QoS, Pub/Sub setup (mostly as before)
        # ... (identical to previous version) ...
        self.current_pose = None; self.other_robot_poses = {}; self.latest_scan = None
        self.state = "EXPLORING" if not self.is_anti_agent else "PATROLLING"
        self.leave_timer_end_time = None; self.anti_agent_cmd_pub = {}
        self.consecutive_avoidance_maneuvers = 0; self.recovery_phase_timer_end_time = None
        self.current_recovery_phase = None; self.recovery_turn_direction_multiplier = 1

        odom_qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        scan_qos_profile = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT, history=HistoryPolicy.KEEP_LAST, depth=1)
        self.cmd_vel_pub = self.create_publisher(Twist, f'/{self.robot_id}/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, f'/{self.robot_id}/ground_truth', self.own_odom_callback, odom_qos_profile)
        self.scan_sub = self.create_subscription(LaserScan, f'/{self.robot_id}/base_scan', self.laser_scan_callback, scan_qos_profile)
        self.other_odom_subs = []
        for i in range(self.total_robots):
            other_robot_name = f"{self.robot_prefix_for_others}{i}"
            if other_robot_name != self.robot_id:
                sub = self.create_subscription(Odometry, f'/{other_robot_name}/ground_truth', lambda msg, rn=other_robot_name: self.other_odom_callback(msg, rn), odom_qos_profile)
                self.other_odom_subs.append(sub); self.other_robot_poses[other_robot_name] = None
        if self.is_anti_agent:
            for i in range(self.total_robots):
                target_robot_name = f"{self.robot_prefix_for_others}{i}"
                if target_robot_name != self.robot_id:
                    pub = self.create_publisher(Empty, f'/{target_robot_name}/leave_command', 10); self.anti_agent_cmd_pub[target_robot_name] = pub
        else: self.leave_sub = self.create_subscription(Empty, f'/{self.robot_id}/leave_command', self.leave_command_callback, 10)
        self.timer = self.create_timer(1.0 / self.update_frequency_val, self.update_logic)


    def own_odom_callback(self, msg: Odometry): self.current_pose = msg.pose.pose
    def other_odom_callback(self, msg: Odometry, robot_name: str): self.other_robot_poses[robot_name] = msg.pose.pose
    def laser_scan_callback(self, msg: LaserScan): self.latest_scan = msg
    def leave_command_callback(self, msg: Empty):
        if not self.is_anti_agent: self.get_logger().info(f"{self.robot_id} received LEAVE!"); self.state = "LEAVING_CLUSTER"; self.leave_timer_end_time = self.get_clock().now() + rclpy.duration.Duration(seconds=self.leave_command_duration_s)

    def get_critical_frontal_obstacle_maneuver(self) -> Twist | None: # Unchanged
        # ... (logic from previous version for critical frontal stop & random default turn)
        if self.latest_scan is None: return None
        min_dist_in_frontal_cone = float('inf'); angle_of_min_dist_in_frontal_cone = 0.0
        any_critical_obstacle_in_cone = False
        for i, dist in enumerate(self.latest_scan.ranges):
            if not (self.latest_scan.range_min < dist < self.latest_scan.range_max and math.isfinite(dist)): continue
            angle = self.latest_scan.angle_min + i * self.latest_scan.angle_increment
            if -self.frontal_scan_angle_rad < angle < self.frontal_scan_angle_rad:
                if dist < min_dist_in_frontal_cone:
                    min_dist_in_frontal_cone = dist; angle_of_min_dist_in_frontal_cone = angle
                if dist < self.obstacle_danger_threshold_m: any_critical_obstacle_in_cone = True
        if any_critical_obstacle_in_cone:
            self.get_logger().warn(f"{self.robot_id} CRITICAL frontal obstacle ({min_dist_in_frontal_cone:.2f}m at {angle_of_min_dist_in_frontal_cone:.2f}rad). STOP & TURN.")
            avoid_twist = Twist(); avoid_twist.linear.x = 0.0
            if abs(angle_of_min_dist_in_frontal_cone) < 0.1: avoid_twist.angular.z = self.obstacle_turn_speed_rad_s * random.choice([-1, 1])
            elif angle_of_min_dist_in_frontal_cone > 0: avoid_twist.angular.z = -self.obstacle_turn_speed_rad_s
            else: avoid_twist.angular.z = self.obstacle_turn_speed_rad_s
            return avoid_twist
        return None

    def adjust_twist_for_side_obstacles(self, desired_twist: Twist, scan: LaserScan) -> Twist: # Unchanged
        # ... (logic from previous version) ...
        if scan is None: return desired_twist
        adjusted_twist = Twist(); adjusted_twist.linear.x = desired_twist.linear.x; adjusted_twist.angular.z = desired_twist.angular.z
        min_dist_left_shoulder = float('inf'); min_dist_right_shoulder = float('inf')
        left_shoulder_min_angle = self.SIDE_CHECK_ANGLE_RAD - self.SIDE_CHECK_SECTOR_WIDTH_RAD / 2.0; left_shoulder_max_angle = self.SIDE_CHECK_ANGLE_RAD + self.SIDE_CHECK_SECTOR_WIDTH_RAD / 2.0
        right_shoulder_min_angle = -self.SIDE_CHECK_ANGLE_RAD - self.SIDE_CHECK_SECTOR_WIDTH_RAD / 2.0; right_shoulder_max_angle = -self.SIDE_CHECK_ANGLE_RAD + self.SIDE_CHECK_SECTOR_WIDTH_RAD / 2.0
        for i, dist in enumerate(scan.ranges):
            if not (scan.range_min < dist < scan.range_max and math.isfinite(dist)): continue
            angle = scan.angle_min + i * scan.angle_increment
            if left_shoulder_min_angle < angle < left_shoulder_max_angle and angle < scan.angle_max:
                if dist < min_dist_left_shoulder: min_dist_left_shoulder = dist
            elif right_shoulder_min_angle < angle < right_shoulder_max_angle and angle > scan.angle_min:
                 if dist < min_dist_right_shoulder: min_dist_right_shoulder = dist
        turn_adjustment = 0.0; speed_was_adjusted_by_side = False
        if min_dist_left_shoulder < self.SIDE_CAUTION_DISTANCE_M:
            self.get_logger().info(f"{self.robot_id} Caution: Left shoulder obs at {min_dist_left_shoulder:.2f}m. Nudging right."); turn_adjustment -= self.angular_speed * self.SIDE_AVOIDANCE_TURN_FACTOR
            if adjusted_twist.linear.x > 0.05 : adjusted_twist.linear.x *= self.SIDE_AVOIDANCE_SPEED_FACTOR; speed_was_adjusted_by_side = True
        if min_dist_right_shoulder < self.SIDE_CAUTION_DISTANCE_M:
            self.get_logger().info(f"{self.robot_id} Caution: Right shoulder obs at {min_dist_right_shoulder:.2f}m. Nudging left."); turn_adjustment += self.angular_speed * self.SIDE_AVOIDANCE_TURN_FACTOR
            if adjusted_twist.linear.x > 0.05 and not speed_was_adjusted_by_side: adjusted_twist.linear.x *= self.SIDE_AVOIDANCE_SPEED_FACTOR
        adjusted_twist.angular.z += turn_adjustment; adjusted_twist.angular.z = max(-self.angular_speed, min(self.angular_speed, adjusted_twist.angular.z))
        return adjusted_twist

    def execute_recovery_maneuver(self) -> Twist: # Unchanged
        # ... (logic from previous version) ...
        recovery_twist = Twist(); now = self.get_clock().now()
        if self.current_recovery_phase == "BACKUP":
            if now < self.recovery_phase_timer_end_time: recovery_twist.linear.x = self.RECOVERY_BACKUP_SPEED_MPS; self.get_logger().info(f"{self.robot_id} Recovery: BACKUP.")
            else: self.get_logger().info(f"{self.robot_id} Recovery: BACKUP finished. Starting random TURN."); self.current_recovery_phase = "TURN"; self.recovery_phase_timer_end_time = now + rclpy.duration.Duration(seconds=self.RECOVERY_TURN_DURATION_S); self.recovery_turn_direction_multiplier = random.choice([-1, 1]); recovery_twist.angular.z = self.RECOVERY_TURN_SPEED_RADS * self.recovery_turn_direction_multiplier
        elif self.current_recovery_phase == "TURN":
            if now < self.recovery_phase_timer_end_time: recovery_twist.angular.z = self.RECOVERY_TURN_SPEED_RADS * self.recovery_turn_direction_multiplier; self.get_logger().info(f"{self.robot_id} Recovery: TURN (dir: {self.recovery_turn_direction_multiplier}).")
            else: self.get_logger().info(f"{self.robot_id} Recovery: TURN finished."); self.state = "EXPLORING" if not self.is_anti_agent else "PATROLLING"; self.current_recovery_phase = None; self.consecutive_avoidance_maneuvers = 0 
        return recovery_twist

    def update_logic(self): # Unchanged
        # ... (logic from previous version) ...
        if self.current_pose is None: return
        final_twist = Twist() 
        if self.state == "RECOVERING":
            final_twist = self.execute_recovery_maneuver()
            self.cmd_vel_pub.publish(final_twist)
            return 
        critical_frontal_maneuver = self.get_critical_frontal_obstacle_maneuver() 
        if critical_frontal_maneuver:
            final_twist = critical_frontal_maneuver
            if self.state not in ["LEAVING_CLUSTER", "RECOVERING"]: self.consecutive_avoidance_maneuvers += 1
            if self.consecutive_avoidance_maneuvers >= self.STUCK_AVOIDANCE_THRESHOLD_CYCLES and \
               self.state not in ["LEAVING_CLUSTER", "RECOVERING"]:
                self.get_logger().warn(f"{self.robot_id} Stuck! ({self.consecutive_avoidance_maneuvers} avoids). Recovery next."); self.state = "RECOVERING"; self.current_recovery_phase = "BACKUP"; self.recovery_phase_timer_end_time = self.get_clock().now() + rclpy.duration.Duration(seconds=self.RECOVERY_BACKUP_DURATION_S)
        else: 
            self.consecutive_avoidance_maneuvers = 0 
            desired_twist_from_behavior = Twist()
            if self.is_anti_agent: desired_twist_from_behavior = self.calculate_anti_agent_twist()
            else: desired_twist_from_behavior = self.calculate_normal_robot_twist()
            final_twist = self.adjust_twist_for_side_obstacles(desired_twist_from_behavior, self.latest_scan)
        self.cmd_vel_pub.publish(final_twist)


    def calculate_normal_robot_twist(self) -> Twist:
        twist = Twist() 

        if self.state == "LEAVING_CLUSTER": # Handles LEAVING_CLUSTER first
            if self.get_clock().now() < self.leave_timer_end_time:
                twist.linear.x = -self.linear_speed * 0.6 
                twist.angular.z = self.angular_speed * (random.random() - 0.5) * 1.5 
                return twist 
            else: # Finished leaving
                self.state = "EXPLORING"
                self.get_logger().info(f"{self.robot_id} finished leaving, now EXPLORING.")
                # Fall through to EXPLORING logic in the same tick for immediate action
        
        # Collect nearby robots for all subsequent states
        nearby_robots_positions = []
        min_dist_to_any_robot_in_group = float('inf')
        if self.current_pose: 
            for r_id, r_pose in self.other_robot_poses.items():
                if r_pose: 
                    dist = euclidean_distance(self.current_pose.position, r_pose.position)
                    if dist < min_dist_to_any_robot_in_group:
                        min_dist_to_any_robot_in_group = dist
                    if dist < self.attraction_radius: # Only consider robots within attraction radius for clustering behavior
                        nearby_robots_positions.append(r_pose.position)

        # State: CLUSTERED (Check this before MOVING_TO_CLUSTER decision, to stay clustered if already so)
        if self.state == "CLUSTERED":
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            # Condition to break from cluster
            # If no robots are nearby OR all previously nearby robots are now too far
            # (using nearby_robots_positions which is based on attraction_radius)
            if not nearby_robots_positions: # No robots even in attraction radius
                 self.state = "EXPLORING"
                 self.get_logger().info(f"{self.robot_id} CLUSTERED but no robots in attraction_radius, -> EXPLORING.")
            elif self.current_pose:
                # Check avg distance to those still in attraction radius
                avg_dist_to_nearby_group = sum(euclidean_distance(self.current_pose.position, p) for p in nearby_robots_positions) / len(nearby_robots_positions) if nearby_robots_positions else float('inf')
                if avg_dist_to_nearby_group > self.clustering_distance * 1.2: # Hysteresis
                    self.state = "EXPLORING"
                    self.get_logger().info(f"{self.robot_id} CLUSTERED but avg_dist ({avg_dist_to_nearby_group:.2f}) > threshold*1.2, -> EXPLORING.")
            return twist # Return stopped twist

        # State: EXPLORING
        if self.state == "EXPLORING":
            if nearby_robots_positions: # Found robots to move towards
                self.state = "MOVING_TO_CLUSTER"
                self.get_logger().info(f"{self.robot_id} EXPLORING -> MOVING_TO_CLUSTER.")
                # Fall through to MOVING_TO_CLUSTER logic in the same tick
            else: # Continue random walk
                twist.linear.x = self.linear_speed * 0.5
                twist.angular.z = self.angular_speed * (random.random() - 0.5) * 1.8 
                return twist # Return exploring twist
        
        # State: MOVING_TO_CLUSTER
        if self.state == "MOVING_TO_CLUSTER":
            if not nearby_robots_positions or not self.current_pose: # Guard: if lost all targets or self.pose
                self.state = "EXPLORING"
                self.get_logger().info(f"{self.robot_id} MOVING_TO_CLUSTER but lost targets/pose -> EXPLORING.")
                twist.linear.x = self.linear_speed * 0.5 
                twist.angular.z = self.angular_speed * (random.random() - 0.5) * 1.8
                return twist

            # Calculate centroid of robots within attraction_radius
            avg_x = sum(p.x for p in nearby_robots_positions) / len(nearby_robots_positions)
            avg_y = sum(p.y for p in nearby_robots_positions) / len(nearby_robots_positions)
            target_centroid = Point(x=avg_x, y=avg_y)
            dist_to_centroid = euclidean_distance(self.current_pose.position, target_centroid)

            # --- Conditions to transition to CLUSTERED state ---
            avg_dist_to_nearby_group = sum(euclidean_distance(self.current_pose.position, p) for p in nearby_robots_positions) / len(nearby_robots_positions)

            transition_to_clustered = False
            if min_dist_to_any_robot_in_group < self.PERSONAL_SPACE_STOP_DISTANCE_M:
                self.get_logger().info(f"{self.robot_id} Personal space breach ({min_dist_to_any_robot_in_group:.2f}m < {self.PERSONAL_SPACE_STOP_DISTANCE_M:.2f}m)! -> CLUSTERED.")
                transition_to_clustered = True
            elif avg_dist_to_nearby_group < self.clustering_distance:
                self.get_logger().info(f"{self.robot_id} Avg group dist ({avg_dist_to_nearby_group:.2f}m < {self.clustering_distance:.2f}m). -> CLUSTERED.")
                transition_to_clustered = True
            elif dist_to_centroid < self.clustering_distance * 0.5 and min_dist_to_any_robot_in_group < (self.clustering_distance + self.PERSONAL_SPACE_STOP_DISTANCE_M)/2.0 :
                self.get_logger().info(f"{self.robot_id} Near centroid ({dist_to_centroid:.2f}m) & close to others ({min_dist_to_any_robot_in_group:.2f}m). -> CLUSTERED.")
                transition_to_clustered = True
            
            if transition_to_clustered:
                self.state = "CLUSTERED"
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                return twist # Stop immediately

            # --- If not yet clustered, calculate movement towards centroid ---
            target_angle = math.atan2(target_centroid.y - self.current_pose.position.y, target_centroid.x - self.current_pose.position.x)
            q = self.current_pose.orientation
            current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            angle_diff = (target_angle - current_yaw + math.pi) % (2 * math.pi) - math.pi

            current_linear_speed = self.linear_speed # Default speed
            if abs(angle_diff) > 0.15: # Turning significantly
                twist.angular.z = self.angular_speed * (1 if angle_diff > 0 else -1)
                current_linear_speed = self.linear_speed * 0.3 # Slow down when turning sharply
            else: # Moving relatively straight towards target
                twist.angular.z = angle_diff * self.angular_speed # Proportional small turn correction
                # Scale speed based on distance to centroid
                if dist_to_centroid < self.clustering_distance: # Very close to target point
                    current_linear_speed = self.MIN_APPROACH_SPEED_MPS
                elif dist_to_centroid < self.attraction_radius * self.APPROACH_SPEED_REDUCTION_FACTOR:
                    # Proportional speed reduction:
                    # Start reducing from linear_speed when dist_to_centroid < attraction_radius * factor
                    # Down to MIN_APPROACH_SPEED_MPS when dist_to_centroid approaches clustering_distance
                    range_for_scaling = (self.attraction_radius * self.APPROACH_SPEED_REDUCTION_FACTOR) - self.clustering_distance
                    if range_for_scaling <= 0: # Avoid division by zero if distances are too close
                        current_linear_speed = self.MIN_APPROACH_SPEED_MPS
                    else:
                        proximity_factor = (dist_to_centroid - self.clustering_distance) / range_for_scaling
                        current_linear_speed = self.MIN_APPROACH_SPEED_MPS + (self.linear_speed - self.MIN_APPROACH_SPEED_MPS) * max(0, min(1, proximity_factor))
                # else: current_linear_speed remains self.linear_speed (further away)
            
            twist.linear.x = max(self.MIN_APPROACH_SPEED_MPS if dist_to_centroid < self.clustering_distance * 1.5 else 0, current_linear_speed) # Ensure min speed if very close, else calculated
            if dist_to_centroid < self.clustering_distance * 0.5 : # If practically at centroid
                 twist.linear.x = 0.0 # Stop linear if very close to centroid and not yet clustered by other rules

            return twist
        
        return twist # Should be covered by states, but as a fallback

    def calculate_anti_agent_twist(self) -> Twist: # Unchanged
        # ... (logic from previous version) ...
        twist = Twist(); twist.linear.x = self.linear_speed * 0.6; twist.angular.z = self.angular_speed * (random.random() - 0.5) * 1.5
        potential_targets = []; all_poses_map = {r_id: pose for r_id, pose in self.other_robot_poses.items() if pose}
        if self.current_pose: 
            for r_id, r_pose in all_poses_map.items():
                if r_id == self.robot_id: continue
                neighbors_of_r_id_count = 0
                for other_r_id, other_r_pose in all_poses_map.items():
                    if other_r_id == r_id or other_r_id == self.robot_id: continue
                    if euclidean_distance(r_pose.position, other_r_pose.position) < self.clustering_distance * 1.5: neighbors_of_r_id_count +=1
                current_robot_cluster_size = neighbors_of_r_id_count + 1
                if current_robot_cluster_size >= self.min_cluster_size_for_disperse:
                    if euclidean_distance(self.current_pose.position, r_pose.position) < self.anti_agent_view_radius:
                        potential_targets.append({'id': r_id, 'size': current_robot_cluster_size, 'pose': r_pose})
            if potential_targets:
                potential_targets.sort(key=lambda x: x['size'], reverse=True); target_robot = potential_targets[0]
                dist_to_target = euclidean_distance(self.current_pose.position, target_robot['pose'].position)
                if dist_to_target < self.anti_agent_view_radius * 0.9: 
                    if target_robot['id'] in self.anti_agent_cmd_pub:
                        self.get_logger().info(f"Anti-agent SENDING LEAVE to {target_robot['id']}"); self.anti_agent_cmd_pub[target_robot['id']].publish(Empty())
                        twist.linear.x = -self.linear_speed * 0.3; twist.angular.z = self.obstacle_turn_speed_rad_s 
                else: 
                    target_angle = math.atan2(target_robot['pose'].position.y - self.current_pose.position.y, target_robot['pose'].position.x - self.current_pose.position.x)
                    q = self.current_pose.orientation; current_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
                    angle_diff = (target_angle - current_yaw + math.pi) % (2 * math.pi) - math.pi
                    if abs(angle_diff) > 0.15: twist.angular.z = self.angular_speed * (1 if angle_diff > 0 else -1); twist.linear.x = self.linear_speed * 0.3 
                    else: twist.linear.x = self.linear_speed * 0.7 
        return twist

# main function
def main(args=None): # Unchanged
    rclpy.init(args=args)
    node = RobotControllerNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: node.get_logger().info(f"Ctrl-C, shutting down {node.robot_id}...")
    finally:
        stop_twist = Twist(); node.cmd_vel_pub.publish(stop_twist)
        node.get_logger().info(f"{node.robot_id} published stop cmd."); node.destroy_node(); rclpy.shutdown()

if __name__ == '__main__':
    main()