from glob import glob

from setuptools import find_packages, setup

package_name = 'streamvln_policy'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/config', glob('config/*.yaml')),
        ('share/' + package_name + '/launch', glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='amilearning',
    maintainer_email='hojin.projects@gmail.com',
    description='ROS 2 policy requester for StreamVLN (image + instruction -> cmd_vel).',
    license='CC-BY-NC-SA-4.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = streamvln_policy.policy_node:main',
        ],
    },
)
