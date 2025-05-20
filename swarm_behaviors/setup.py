from setuptools import find_packages, setup
import os

package_name = 'swarm_behaviors'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Add this line to install all launch files
        ('share/' + package_name + '/launch', ['launch/' + f for f in os.listdir('launch') if f.endswith('.py')]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ujjwal',
    maintainer_email='ujjwalpatil20@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'robot_controller = swarm_behaviors.robot_controller:main',
        ],
    },
)
