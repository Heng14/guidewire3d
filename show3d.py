import open3d as o3d
import sys
import numpy as np

def vis():
    # xyz = np.asarray(np.load(npy))[:, :3]
    xyz = np.random.rand(10000, 3) * 2000

    np.random.shuffle(xyz)
    # xyz = xyz[:80000]
    # xyz = xyz[xyz[:,2] < 1300]

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(xyz)

    print(pc)
    print(np.asarray(pc.points))

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pc)

    while (True):
    # for i in range(100):
        # geometry = np.array([
        #     [ 0.99985 , 0.017452, 0, 0.0],
        #     [-0.017452,  0.99985 , 0, 0.0],
        #     [0, 0, 1, -1.4],
        #     [0.0, 0.0, 0.0, 1.0]])
        # pc.transform(geometry)
        vis.update_geometry(pc)
        vis.poll_events()
        vis.update_renderer()

if __name__ == '__main__':
    vis()