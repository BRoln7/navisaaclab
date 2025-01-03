import torch
import yaml
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from dataclasses import dataclass
import random

@dataclass
class MapInfo:
    binary_map : np.array 
    res : float
    origin : np.array

def load_map(_png_path, _yaml_path):
    yaml_path = _yaml_path
    png_path = _png_path

    with open(yaml_path, 'r') as file:
        map_yaml = yaml.safe_load(file)
    resolution = map_yaml['resolution'] # type: ignore
    origin = map_yaml['origin'] # type: ignore
    
    map_image = Image.open(png_path)
    map_image = map_image.convert('L')
    map_image = map_image.transpose(method=Image.Transpose.FLIP_TOP_BOTTOM)
    binary_map = np.array(map_image)
    binary_map[binary_map < 128] = 1
    binary_map[binary_map >= 128] = 0
    structure_element = np.ones((25, 25))
    binary_map = ndimage.binary_dilation(binary_map, 
                                         structure=structure_element).astype(binary_map.dtype)

    return binary_map, resolution, origin

def publish_random_goal(_start_points:torch.tensor, _map_info:MapInfo):
    '''
    set a annular region centered at start point
    then randomly select from this region
    '''
    min_radius = 4.5
    max_radius = 7.0

    rows, cols = _map_info.binary_map.shape
    sampled_goals = []
    robot_poses = _start_points

    for robot_pos in robot_poses:
        x_world, y_world = robot_pos
        x_idx = int((x_world - _map_info.origin[0]) / _map_info.res)
        y_idx = int((y_world - _map_info.origin[1]) / _map_info.res)
        
        while True:
            angle = random.uniform(0, 2 * np.pi)
            distance = random.uniform(min_radius, max_radius)
            
            dx = int(distance * np.cos(angle) / _map_info.res)
            dy = int(distance * np.sin(angle) / _map_info.res)
            sample_x = x_idx + dx
            sample_y = y_idx + dy
            
            if 0 <= sample_x < cols and 0 <= sample_y < rows:
                if _map_info.binary_map[sample_y, sample_x] == 0:
                    sampled_goals.append(torch.tensor((sample_x*_map_info.res+_map_info.origin[0], 
                                                       sample_y*_map_info.res+_map_info.origin[1]),
                                                       device='cuda:0'))
                break
        
    sampled_goals = torch.stack(sampled_goals)

    return sampled_goals