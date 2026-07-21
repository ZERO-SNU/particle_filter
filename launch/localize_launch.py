# MIT License

# Copyright (c) 2020 Hongrui Zheng

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.actions import DeclareLaunchArgument
from ament_index_python.packages import get_package_share_directory
import os
import yaml

def generate_launch_description():
    # config and args
    localize_config = os.path.join(
        get_package_share_directory('particle_filter'),
        'config',
        'localize.yaml'
    )
    localize_la = DeclareLaunchArgument(
        'localize_config',
        default_value=localize_config,
        description='Localization configs')
    sim_mode_la = DeclareLaunchArgument(
        'sim_mode',
        default_value='false',
        description='Disable 180-degree scan/TF correction in simulation')

    # 트랙 SSOT: 맵은 track_assets 패키지에서 가져온다.
    # (localize.yaml의 map 키와 particle_filter/maps/ 는 더 이상 사용하지 않음)
    # track:=<이름> 인자를 생략하면 track_assets/config/active_track.yaml 값 사용.
    assets_share = get_package_share_directory('track_assets')
    with open(os.path.join(assets_share, 'config', 'active_track.yaml'), 'r') as f:
        default_track = yaml.safe_load(f)['active_track']
    track_la = DeclareLaunchArgument(
        'track',
        default_value=default_track,
        description='track_assets/maps/<track> 폴더 이름 (기본값: active_track.yaml)')
    map_yaml = PathJoinSubstitution([assets_share, 'maps', LaunchConfiguration('track'), 'map.yaml'])

    ld = LaunchDescription([localize_la, sim_mode_la, track_la])

    # nodes
    pf_node = Node(
        package='particle_filter',
        executable='particle_filter',
        name='particle_filter',
        parameters=[
            LaunchConfiguration('localize_config'),
            {'sim_mode': LaunchConfiguration('sim_mode')}
        ]
    )
    map_server_node = Node(
        package='nav2_map_server',
        executable='map_server',
        name='map_server',
        parameters=[{'yaml_filename': map_yaml},
                    {'topic': 'map'},
                    {'frame_id': 'map'},
                    {'output': 'screen'},
                    {'use_sim_time': True}]
    )
    nav_lifecycle_node = Node(
        package='nav2_lifecycle_manager',
        executable='lifecycle_manager',
        name='lifecycle_manager_localization',
        output='screen',
        parameters=[{'use_sim_time': True},
                    {'autostart': True},
                    {'node_names': ['map_server']}]
    )

    # finalize
    ld.add_action(nav_lifecycle_node)
    ld.add_action(map_server_node)
    ld.add_action(pf_node)

    return ld