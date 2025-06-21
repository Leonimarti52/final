from setuptools import find_packages, setup

package_name = 'leoni'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/odo.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='leoni',
    maintainer_email='leoni@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cam=leoni.cam:main',
            'point=leoni.point:main',
            'oakd=leoni.do:main',
            'duo1=leoni.du:main',
            'imu=leoni.imu:main',
            'fu=leoni.fution:main',
            'vi=leoni.view:main'
        ],
    },
)
