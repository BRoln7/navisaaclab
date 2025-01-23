import torch
import yaml
import random
import cv2
import numpy as np
from PIL import Image
import scipy.ndimage as ndimage
from dataclasses import dataclass

@dataclass
class MapInfo:
    dilated_img: np.array
    binary_img : np.array
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

def load_image_scaled(_img_path, _yaml_path, scale=1/3, dilation_size=5):
    '''
    scale: 1080 figure ---> 1080/3
    '''
    with open(_yaml_path, 'r') as file:
        map_yaml = yaml.safe_load(file)
    resolution = map_yaml['resolution'] # type: ignore
    res = resolution / scale
    origin = map_yaml['origin'] # type: ignore    

    img = cv2.imread(_img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.flip(img, 0)
    print(img.shape)

    img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    
    _, binary_img = cv2.threshold(img, 126, 1, cv2.THRESH_BINARY_INV)
    _, binary_img2 = cv2.threshold(img, 126, 1, cv2.THRESH_BINARY)
    kernel = np.ones((dilation_size, dilation_size), np.uint8)
    dilated_img = cv2.dilate(binary_img, kernel, iterations=1)

    return dilated_img, binary_img, res, origin

def random_spawn(_map_info:MapInfo, _num_agent:int):
    
    free_space_coord = np.argwhere(_map_info.dilated_img == 0)
    random_pos = free_space_coord[np.random.choice(free_space_coord.shape[0], _num_agent, replace=False)]

    spawn_pos_m = torch.tensor(random_pos, device='cuda:0')
    spawn_pos_m = torch.flip(spawn_pos_m, dims=[-1])

    spawn_pos_w = transform_map_to_world(spawn_pos_m.float(), _map_info)

    return spawn_pos_m.to(torch.int), spawn_pos_w.to(torch.float)

def publish_random_goal(_start_points:torch.tensor, _map_info:MapInfo, _min_length, _max_lenght):
    '''
    set a annular region centered at start point
    then randomly select from this region
    '''
    min_radius = _min_length
    max_radius = _max_lenght

    rows, cols = _map_info.dilated_img.shape
    sampled_goals_w = []
    sampled_goals_m = []
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
                if _map_info.dilated_img[sample_y, sample_x] == 0:
                    sampled_goals_w.append(torch.tensor((sample_x*_map_info.res+_map_info.origin[0], 
                                                         sample_y*_map_info.res+_map_info.origin[1]),
                                                         device='cuda:0'))
                    sampled_goals_m.append(torch.tensor([sample_x, sample_y], device='cuda:0'))
                    break
        
    sampled_goals_w = torch.stack(sampled_goals_w)
    sampled_goals_m = torch.stack(sampled_goals_m).to(torch.int)

    return sampled_goals_m, sampled_goals_w

def transform_map_to_world(_pos_map:torch.tensor, _map_info:MapInfo):
    pos_map = _pos_map.to(torch.float)
    pos_world = torch.zeros_like(_pos_map, device='cuda:0', dtype=torch.float)
    pos_world[:, 0] = pos_map[:, 0] * _map_info.res + _map_info.origin[0]
    pos_world[:, 1] = pos_map[:, 1] * _map_info.res + _map_info.origin[1]
    
    return pos_world

def transform_world_to_map(_pos_world:torch.tensor, _map_info:MapInfo):
    pos_map = torch.zeros_like(_pos_world, device='cuda:0', dtype=torch.int)
    
    pos_map[:, 0] = (_pos_world[:, 0] - _map_info.origin[0]) / _map_info.res
    pos_map[:, 1] = (_pos_world[:, 1] - _map_info.origin[1]) / _map_info.res

    pos_map = pos_map.to(torch.int)
    
    return pos_map

def get_spawn_and_goal(num_robots:int, num_store:int, num_garden:int, num_station:int, 
                       joint_store, joint_garden, map1_info, map2_info, map3_info):
    # get spawn for robot, get goals for robot in small scene

    goals_w = torch.ones((num_robots, 2),device='cuda:0')
    goals_m = torch.ones((num_robots, 2),device='cuda:0', dtype=torch.int)

    env_ids = torch.arange(num_robots, device='cuda:0')
    store_id_0 = env_ids[env_ids < num_store]
    garden_id_0 = env_ids[(env_ids >= num_store) & (env_ids < (num_store + num_garden))]  

    path_store_m = joint_store[torch.randperm(joint_store.size(0))[:num_store].numpy().tolist()]
    spawn_store_m, goal_store_m = path_store_m[:, :2], path_store_m[:, 2:]
    spawn_store_w, goal_store_w = transform_map_to_world(spawn_store_m, map1_info), \
                                  transform_map_to_world(goal_store_m, map1_info)
    goals_w[store_id_0], goals_m[store_id_0] = goal_store_w, goal_store_m

    path_garden_m = joint_garden[torch.randperm(joint_garden.size(0))[:num_garden].numpy().tolist()]
    spawn_garden_m, goal_garden_m = path_garden_m[:, :2], path_garden_m[:, 2:]
    spawn_garden_w, goal_garden_w = transform_map_to_world(spawn_garden_m, map2_info), \
                                    transform_map_to_world(goal_garden_m, map2_info)
    goals_w[garden_id_0], goals_m[garden_id_0] = goal_garden_w, goal_garden_m

    spawn_station_m, spawn_station_w = random_spawn(map3_info, num_station)

    spawn_m = torch.cat([spawn_store_m, spawn_garden_m, spawn_station_m], dim=0).to(torch.int)
    spawn_w = torch.cat([spawn_store_w, spawn_garden_w, spawn_station_w], dim=0)

    return spawn_m, spawn_w, goals_w, goals_m



'''
# Example usage:
# Assuming the center coordinates (x, y) are provided and valid
center_x, center_y = 500, 500  # example coordinates in pixels
pixel_extent = int(90*5)  # 90 meters converted to pixels
scale_factor = 4/12
img_path = './datasets/new_maps/warehouse/map.png'
dilated_img, binary_img = load_image(img_path, scale=scale_factor)


height_m, width_m  = (90.0, 90.0)
resolution = 12*scale_factor
height_px, width_px = round(height_m * resolution), round(width_m * resolution)
# Transformation from world coordinates [m] to map coordinates [px]
map_from_world: np.ndarray = np.array(
    [[resolution, 0.0, resolution * (width_m / 2.0)],
    [0.0, resolution, resolution * (height_m / 2.0)],
    [0.0, 0.0, 1.0]]
)

plt.plot(resolution * (width_m / 2.0), resolution * (height_m / 2.0), 'x')

time = 11
trackTime = np.arange(0, time, 0.1)

map_pos = []
map_vel = []

start = (3, 9, 1)
start_map = map_from_world@np.array(start)
start_map = tuple(start_map[:2].astype(int).tolist())
goal_map = (start_map[0] - 40, start_map[1] - 30,)
print(start_map, goal_map)

path = get_dijkstra_path(dilated_img, binary_img, start_map, goal_map, False)

vel = 3
map_pos.append(path[:vel*time*10:vel])

start = (170, 230)
goal = (0, 0, 1)
goal_map = map_from_world@np.array(goal)
goal_map = tuple(goal_map[:2].astype(int).tolist())

path1 = get_dijkstra_path(dilated_img, binary_img, start, goal_map, True)
vel = 2
map_pos.append(path1[:vel*time*10:vel])


trackPos = []

for i in range(2):
    column_of_ones = np.ones((map_pos[i].shape[0], 1))
    expanded_pos = np.hstack((map_pos[i], column_of_ones))
    gpos = np.linalg.inv(map_from_world) @ expanded_pos.T
    trackPos.append(gpos[:2, :].T)

trackPos = np.array(trackPos)

trackVel = trackPos[:, 1:, :] - trackPos[:, :-1, :]
trackVel = trackVel*10

trackPos = trackPos[:,:-1,:]

rad = np.array([0.4,0.4])
print(trackPos.shape, trackVel.shape)
sim_data = {
    'trackTime':trackTime[:-1], 'trackPos':trackPos, 'trackVel':trackVel, "radius":rad
    }
np.savez('./datasets/new_maps/sim.npz', **sim_data)



# height_m, width_m  = (90.0, 90.0)
# resolution = 12
# height_px, width_px = round(height_m * resolution), round(width_m * resolution)
# # Transformation from world coordinates [m] to map coordinates [px]
# map_from_world: np.ndarray = np.array(
#         [[resolution, 0.0, resolution * (width_m / 2.0)],
#         [0.0, resolution, resolution * (height_m / 2.0)],
#         [0.0, 0.0, 1.0]]
# )
'''

# def get_map_mask(map_data, layer_name, map_from_world, canvas_size):
#     '''
#     Rasterizes an ORCA sim map.
#     - obstacles: ORCA map data (list of obstacle polygons)
#     - layer_name: one of ["walkable", "obstacles"] supported.
#     - map_from_world: 3x3 transformation matrix from world coordinates [m] to map coordinates [px]
#     - canvas_size tuple: (H, W) pixels tuple which determines the rasterized image size
#     '''
#     init_val = 1 if layer_name == "walkable" else 0
#     map_mask = np.ones(canvas_size, np.uint8) * init_val
#     for obstacles in map_data:
#         cur_obs = np.array(obstacles)
#         # convert to pixel coords
#         poly_pts = np.concatenate([cur_obs, np.ones((cur_obs.shape[0], 1))], axis=1)
#         poly_pts = map_from_world @ poly_pts.T
#         poly_pts = poly_pts[:-1, :].T
#         # draw polygon
#         coords = poly_pts.round().astype(np.int32)
#         cv2.fillPoly(map_mask, [coords], 0 if layer_name == "walkable" else 1)

#     return map_mask

# import pickle
# map_path = './datasets/orca_sim/maps/scene_000000/map.pkl'
# with open(map_path, "rb") as mapf:
#     map_obj = pickle.load(mapf)

# map_mask = get_map_mask(map_obj, 'walkable', map_from_world, [height_px, width_px])
# img = cv2.resize(map_mask, (0, 0), fx=1/12, fy=1/12, interpolation=cv2.INTER_NEAREST)

# sim_path = './datasets/orca_sim/maps/scene_000000/sim.npz'
# sim_data = np.load(sim_path)
# sim_dict = {
#     "ts" : sim_data["trackTime"],
#     "pos" : sim_data["trackPos"],
#     "vel" : sim_data["trackVel"],
#     "radius" : sim_data["radius"].reshape((-1,)),
# }


# # ag = sim_dict['pos'][1]
# # ones_column = np.ones((ag.shape[0], 1))

# # trackPos = np.dot(np.hstack((ag,ones_column)) , map_from_world)
# plt.imshow(img, origin='lower')
# ag0 = sim_dict['pos'][0]
# ag0_vel = sim_dict['vel'][0]
# ag0_head = np.arctan2(ag0_vel[31,1], ag0_vel[31,0])
# trans_mean = ag0[31,0:2]
# trans_rotmat: np.ndarray = np.array(
#                 [
#                     [np.cos(ag0_head), -np.sin(ag0_head)],
#                     [np.sin(ag0_head), np.cos(ag0_head)],
#                 ]
#             )
# print(trans_rotmat, trans_mean)
# plt.figure()
# ag1 = sim_dict['pos'][1][:31, 0:2]
# ag1ag0 = (ag1-trans_mean)@trans_rotmat
# plt.plot(ag1ag0[:,0], ag1ag0[:,1])


# # for ag in range(2):#sim_dict['pos'].shape[0]
# #     ag_org = sim_dict['pos'][ag] + np.array([45, 45])
# #     # ag_ext = np.hstack((ag_org, np.ones((ag_org.shape[0], 1))))
# #     # ag_map = np.dot(map_from_world, ag_ext.T).T
# #     plt.plot(ag_org[31:,0], ag_org[31:,1])
# #     # frame31 = ag_org[31]-  np.array([45, 45])
# #     # frame30 =ag_org[30]-  np.array([45, 45])
# #     # vel = sim_dict['vel'][ag][31]
# #     # print(frame31, frame30+ vel*0.1, frame30)

# plt.show()    
