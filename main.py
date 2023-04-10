import os
import numpy as np
import pyvista as pv
import multiprocessing as mp
import imageio

def raycast_worker(mesh, x_range, y_range, max_bound, resolution, output, height):
    for x in x_range:
        for y in y_range:
            origin = np.array([x, y, max_bound[2]])
            direction = np.array([0, 0, -1])
            intersections = mesh.ray_trace(origin, direction)
            if intersections[0].size > 0:
                index = int((y - y_range[0]) * len(x_range) + (x - x_range[0]))
                output[index] = max_bound[2] - intersections[0][0][2]

def generate_top_view_depth_image(file_path, output_path, resolution=1.0):
    pv_mesh = pv.read(file_path)
    mesh = pv.PolyData(pv_mesh)
    
    bounds = mesh.bounds
    max_bound = np.array([bounds[1], bounds[3], bounds[5]])
    min_bound = np.array([bounds[0], bounds[2], bounds[4]])
    width, height = int((max_bound[0] - min_bound[0]) // resolution), int((max_bound[1] - min_bound[1]) // resolution)
    
    depth_image = np.zeros((height, width))
    
    num_processes = os.cpu_count()
    x_ranges = np.array_split(np.arange(min_bound[0], max_bound[0], resolution), num_processes)
    y_range = np.arange(min_bound[1], max_bound[1], resolution)

    manager = mp.Manager()
    shared_output = manager.Array('d', [0.0] * (width * height))
    
    processes = []
    for i in range(num_processes):
        process = mp.Process(target=raycast_worker, args=(mesh, x_ranges[i], y_range, max_bound, resolution, shared_output, height))
        process.start()
        processes.append(process)
    
    for process in processes:
        process.join()
    
    shared_output_np = np.array(shared_output).reshape(height, width)

    for i in range(num_processes):
        start_idx = int((x_ranges[i][0] - min_bound[0]) // resolution)
        end_idx = int((x_ranges[i][-1] - min_bound[0]) // resolution) + 1
        depth_image[:, start_idx:end_idx] = shared_output_np[:, start_idx:end_idx]

    depth_image = np.flip(depth_image, axis=0)
    depth_image = 255 * (depth_image - depth_image.min()) / (depth_image.max() - depth_image.min())
    depth_image = depth_image.astype(np.uint8)
    
    imageio.imwrite(output_path, depth_image)

if __name__ == '__main__':
    generate_top_view_depth_image("2d_chess.obj", "2DChess_depth_image.png", resolution=0.1)
