from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'heron_experiment'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml') + glob('config/*.yml') + glob('config/*.xml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='will-44',
    maintainer_email='todo@todo.todo',
    description='Heron on the robot: reachability inference, gradient base positioning, '
                'pick-and-place experiments and their analysis',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'reachability_node=heron_experiment.nodes.reachability_node:main',
            'metrics_recorder=heron_experiment.nodes.metrics_recorder:main',
            'workspace_reachability_node=heron_experiment.nodes.workspace_reachability_node:main',
            'gradient_base_controller=heron_experiment.nodes.gradient_base_controller:main',
            'workspace_probe=heron_experiment.nodes.workspace_probe:main',
            'rm_noise_probe=heron_experiment.nodes.rm_noise_probe:main',
            'grasp=heron_experiment.nodes.grasp:main',
            'release=heron_experiment.nodes.release:main',
            'brain=heron_experiment.nodes.brain:main',
            'replay_maps=heron_experiment.nodes.replay_maps:main',
        ]
    },
)
