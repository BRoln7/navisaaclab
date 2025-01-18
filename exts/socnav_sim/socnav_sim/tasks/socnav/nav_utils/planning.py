import torch
import numpy as np
import heapq
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import cm

# define 8 direction for searching
DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)]

def get_dijkstra_cost(grid, start, goal):
    rows, cols = grid.shape
    distance = np.full(grid.shape, np.inf)
    distance[start] = 0
    pq = [(0, start)]
    parent = np.full((rows, cols, 2), -1)

    while pq:
        current_dist, current_node = heapq.heappop(pq)
        current_dist = current_dist - np.sqrt((current_node[0]-goal[0])**2+(current_node[1]-goal[1])**2)
        
        if current_node == goal:
            break
        
        for direction in DIRECTIONS:
            neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])
            
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols and grid[neighbor] == 0:
                weight = 1 if direction[0] == 0 or direction[1] == 0 else np.sqrt(2)
                distance_through_current = current_dist + weight + np.sqrt((neighbor[0]-goal[0])**2+(neighbor[1]-goal[1])**2)
                
                if distance_through_current < distance[neighbor]:
                    distance[neighbor] = distance_through_current
                    parent[neighbor] = current_node
                    heapq.heappush(pq, (distance_through_current, neighbor))

    return distance, parent

def reconstruct_path(parent, start, goal):
    path = []
    x, y = goal
    while parent[x, y][0] != -1:
        path.append((x, y))
        x, y = parent[x, y]
    path.append(start)
    path.reverse()
    return np.array(path)

def calculate_curvature(path):
    x = path[:, 0]
    y = path[:, 1]

    # (x3 - x1)*(y2 - y1) - (y3 - y1)*(x2 - x1)
    dx1 = x[2:] - x[:-2]  # x3 - x1
    dy1 = y[2:] - y[:-2]  # y3 - y1
    dx2 = x[1:-1] - x[:-2]  # x2 - x1
    dy2 = y[1:-1] - y[:-2]  # y2 - y1

    area = np.abs(dx1 * dy2 - dy1 * dx2)

    # (x2 - x1)^2 + (y2 - y1)^2
    length_sq = (dx2 ** 2 + dy2 ** 2) ** 1.5

    curvatures = np.divide(2 * area, length_sq, where=length_sq != 0)

    return curvatures

def select_control_points(path, curvature_threshold=0.2):
    curvatures = calculate_curvature(path)
    control_points = [path[0]]
    j = 0 
    for i, curv in enumerate(curvatures):
        if curv > curvature_threshold:
            control_points.append(path[i + 1])
            j = 0
        else:
            j += 1
            if j > 6:
                control_points.append(path[i])
                j = 0
    control_points.append(path[-1])
    return np.array(control_points)


from scipy.interpolate import splprep, splev

def smooth_with_b_spline(control_points, num):

    tck, u = splprep(control_points.T, s=0)
    x = np.linspace(0, 1, num)
    new_points = splev(x, tck)
    return np.array(new_points).T

def visualize_cost_and_path(grid, distance, start, goal, path=None):
    color_map = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.float32)

    color_map[grid == 1] = [0, 0, 0]

    color_map[(grid == 0)] = [0.6, 0.6, 0.6]

    finite_mask = np.isfinite(distance)
    if finite_mask.any():
        norm = Normalize(vmin=np.min(distance[finite_mask]), vmax=np.max(distance[finite_mask]))
        cmap = cm.get_cmap('viridis')
        color_map[finite_mask] = cmap(norm(distance[finite_mask]))[:, :3] # type: ignore

    color_map[start] = [0, 1, 0]
    color_map[goal] = [1, 0, 0]

    plt.imshow(color_map, origin='lower')
    # plt.show()


def get_dijkstra_path(grid, img, start, goal, viz=False, traj_optim=False):
    '''
        grid: ndarray binary map for planning
        img: ndarray origin figure for visualization
        start: tuple 
        goal: tuple 
        viz: bool flag for visiualization
        traj_optim: bool flag for b_spline trajctory optimization  
        
        return: trajectory
    '''
    start = start[::-1]
    goal = goal[::-1]
    distance, parents= get_dijkstra_cost(grid, start, goal)

    if distance[goal] != float('inf'):
        path = reconstruct_path(parents, start, goal)
        # visualize_cost_and_path(img, distance, start, goal, path)
        if traj_optim:
            control_points = select_control_points(path)
            smoothed_path = smooth_with_b_spline(control_points, 10*path.shape[0])
            # print('*** Found b-spline path with: ', smoothed_path.shape)
        else:
            smoothed_path = path

        if viz:
            visualize_cost_and_path(img, distance, start, goal, path)
            plt.plot(start[1], start[0],'rx')
            plt.plot(goal[1], goal[0], 'bx')
            plt.plot( path[:, 1], path[:, 0], color='b')
            plt.plot( smoothed_path[:, 1], smoothed_path[:, 0], linestyle='--', color='r')
            plt.scatter(control_points[:,1], control_points[:,0], color='g')
            plt.show()
        return smoothed_path[:, ::-1]
    else:
        print("No path found from start to goal.")
        if viz:
            visualize_cost_and_path(img, distance, start, goal)

def path_align_length(_path:np.array, traj_length:int)->torch.tensor:

    path = torch.cat([torch.tensor(_path.copy(), device='cuda:0'),
                      torch.ones((traj_length - _path.shape[0]), 2, device='cuda:0') *\
                      torch.tensor(_path[-1].copy(), device='cuda:0')], dim=0)
    
    return path