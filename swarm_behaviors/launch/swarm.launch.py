import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, TextSubstitution

def generate_launch_description():
    num_robots_arg = DeclareLaunchArgument(
        'num_robots', 
        default_value='7', 
        description='Total number of robots'
    )
    robot_prefix_arg = DeclareLaunchArgument(
        'robot_prefix', 
        default_value='robot_', 
        description='Prefix for robot names (e.g., robot_ for robot_0)'
    )
    # This will be the ID of the robot designated as the anti-agent, e.g., "robot_0"
    designated_anti_agent_id_arg = DeclareLaunchArgument(
        'designated_anti_agent_id', 
        default_value='robot_0', 
        description='Full ID of the robot to be an anti-agent (e.g., robot_0)'
    )

    # These are LaunchConfiguration objects, not direct strings yet.
    num_robots_lc = LaunchConfiguration('num_robots')
    robot_prefix_lc = LaunchConfiguration('robot_prefix')
    designated_anti_agent_id_lc = LaunchConfiguration('designated_anti_agent_id')
    
    robot_node_actions = [] # Changed variable name to avoid conflict
    
    # For iterating and creating nodes. The 'num_robots' parameter is also passed to nodes
    # for their internal logic if they need to know the total count.
    # This loop creates 'default_num_robots_for_loop' node configurations.
    # If you change 'num_robots' via launch argument, this loop here won't change,
    # but the 'total_robots' parameter inside each node WILL reflect the launch argument.
    # This is a common pattern: launch a fixed number of potential nodes or use OpaqueFunction for fully dynamic.
    default_num_robots_for_loop = 7 

    for i in range(default_num_robots_for_loop):
        # current_robot_id_for_param will be a list of substitutions.
        # The ROS 2 launch system will resolve this list into a single string like "robot_0"
        # when it processes the parameters for the Node.
        current_robot_id_for_param = [robot_prefix_lc, str(i)]
        
        # The namespace for the node should also be constructed this way.
        current_robot_namespace = [robot_prefix_lc, str(i)]

        robot_node_actions.append(
            Node(
                package='swarm_behaviors', # Make sure this is your package name
                executable='robot_controller', # Your Python script executable
                # Node name can be simple, as it will be under a namespace
                name=['controller_', str(i)], # e.g. controller_0 (becomes /robot_0/controller_0)
                namespace=current_robot_namespace, # e.g. robot_0, robot_1
                parameters=[
                    # The 'robot_id' parameter in your Python node will receive "robot_0", "robot_1", etc.
                    {'robot_id': current_robot_id_for_param}, 
                    {'total_robots': num_robots_lc}, # Node receives the configured total number
                    {'designated_anti_agent_id': designated_anti_agent_id_lc}, # Node uses this to see if it's the one
                    {'robot_prefix_for_others': robot_prefix_lc}, # Pass the prefix for constructing other robot names

                    # Other parameters from your node's declarations
                    {'linear_speed': 0.3},
                    {'angular_speed': 0.5},
                    {'attraction_radius': 3.0}, 
                    {'clustering_distance': 0.8}, 
                    {'anti_agent_view_radius': 4.0}, 
                    {'min_cluster_size_for_disperse': 3}, 
                    {'leave_command_duration_s': 10.0},
                    {'update_frequency': 5.0} # As in your previous file
                ],
                output='screen',
                emulate_tty=True, # Good for seeing Python's print() or direct logging output
            )
        )

    ld = LaunchDescription([
        num_robots_arg,
        robot_prefix_arg,
        designated_anti_agent_id_arg,
    ])

    for node_action in robot_node_actions:
        ld.add_action(node_action)
        
    return ld