# import torch
# import torch.nn as nn

import numpy as np
import trimesh
import copy

from sklearn.neighbors import KDTree
import warnings

class ICP_T_S():
    def __init__(self, mesh_source, mesh_target):
        self.mesh_source = mesh_source
        self.mesh_target = mesh_target

        self.points_source = self.mesh_source.vertices.copy()
        self.points_target = self.mesh_target.vertices.copy()

    def sample_mesh(self, n=30000, mesh_id='both'):
        if mesh_id == 'source' or mesh_id == 'both':
            self.points_source, _ = trimesh.sample.sample_surface(self.mesh_source, n)
        if mesh_id == 'target' or mesh_id == 'both':
            self.points_target, _ = trimesh.sample.sample_surface(self.mesh_target, n)

        self.offset_source = self.points_source.mean(0)
        self.scale_source = np.sqrt(((self.points_source - self.offset_source)**2).sum() / len(self.points_source))
        self.offset_target = self.points_target.mean(0)
        self.scale_target = np.sqrt(((self.points_target - self.offset_target)**2).sum() / len(self.points_target))

        self.points_source = (self.points_source - self.offset_source) / self.scale_source * self.scale_target + self.offset_target

    def run_icp_f(self, max_iter = 10, stop_error = 1e-3, stop_improvement = 1e-5, verbose=0):
        # Difference with run_icp():
            # run_icp_1 build KDTree only once
            # run_icp() build KDTree in every iteration.
        # Build KDTree for both original target and source point cloud
        self.target_KDTree = KDTree(self.points_target)
        self.source_KDTree = KDTree(self.points_source)

        self.trans = np.zeros((1,3), dtype = np.float)
        self.scale = 1.0
        self.A_c123 = []

        error = 1e8
        previous_error = error
        for i in range(0, max_iter):
            
            # Find closest target point for each source point:
            query_source_points = self.points_source*self.scale + self.trans
            _, closest_target_points_index = self.target_KDTree.query(query_source_points)
            closest_target_points = self.points_target[closest_target_points_index[:,0], :]

            # Find closest source point for each target point:
            query_target_points = (self.points_target - self.trans)/self.scale
            _, closest_source_points_index = self.source_KDTree.query(query_target_points)
            closest_source_points = self.points_source[closest_source_points_index[:,0], :]
            closest_source_points = closest_source_points*self.scale + self.trans
            query_target_points = self.points_target

            # Compute current error:
            error = (((query_source_points-closest_target_points)**2).sum() +\
            ((query_target_points-closest_source_points)**2).sum())/(query_source_points.shape[0] + query_target_points.shape[0])
            error = error**0.5
            if verbose >= 1:
                print(i, "th iter, error: ", error)

            if previous_error - error < stop_improvement:
                break
            else:
                previous_error = error

            if error < stop_error:
                break

            ''' 
            Build lsq linear system:
            / x1 1 0 0 \  / scale \     / x_t1 \
            | y1 0 1 0 |  |  t_x  |  =  | y_t1 |
            | z1 0 0 1 |  |  t_y  |     | z_t1 | 
            | x2 1 0 0 |  \  t_z  /     | x_t2 |
            | ...      |                | .... |
            \ zn 0 0 1 /                \ z_tn /
            '''
            A_c0 = np.vstack([self.points_source.reshape(-1,1),
                              self.points_source[closest_source_points_index[:,0], :].reshape(-1,1)])
            if i == 0:
                A_c1 = np.zeros((self.points_source.shape[0] + self.points_target.shape[0], 3),\
                            dtype = np.float) + np.array([1.0, 0.0, 0.0])
                A_c1 = A_c1.reshape(-1, 1)
                A_c2 = np.zeros_like(A_c1)
                A_c2[1:,0] = A_c1[0:-1, 0]
                A_c3 = np.zeros_like(A_c1)
                A_c3[2:,0] = A_c1[0:-2, 0]

                self.A_c123 = np.hstack([A_c1, A_c2, A_c3])

            A = np.hstack([A_c0, self.A_c123])
            # print(closest_target_points.reshape(-1,1).shape)
            # print(query_target_points.reshape(-1,1).shape)
            b = np.vstack([closest_target_points.reshape(-1,1),
                                query_target_points.reshape(-1, 1)])
            # print(A.shape)
            # print(b.shape)
            x = np.linalg.lstsq(A,b)
            self.scale = x[0][0]
            self.trans = (x[0][1:]).transpose()

            # query_source_points = self.points_source*self.scale + self.trans
            # closest_source_points = self.points_source[closest_source_points_index[:,0], :]*self.scale + self.trans
            # error = (((query_source_points-closest_target_points)**2).sum() +\
            # ((query_target_points-closest_source_points)**2).sum())/(query_source_points.shape[0] + query_target_points.shape[0])
            # error = error**0.5
            # print(i, "th iter, error: ", error)
    
    def run_icp(self, max_iter = 10, stop_error = 1e-3):
        # Build KDTree for both original target and source point cloud
        self.target_KDTree = KDTree(self.points_target)
        self.source_KDTree = KDTree(self.points_source)

        self.trans = np.zeros((1,3), dtype = np.float)
        self.scale = 1.0
        self.A_c123 = []

        error = 1e8
        for i in range(0, max_iter):
            
            # Find closest target point for each source point:
            self.source_KDTree = KDTree(self.points_source*self.scale + self.trans)
            query_source_points = self.points_source*self.scale + self.trans
            _, closest_target_points_index = self.target_KDTree.query(query_source_points)
            closest_target_points = self.points_target[closest_target_points_index[:,0], :]

            # Find closest source point for each target point:
            query_target_points = self.points_target
            _, closest_source_points_index = self.source_KDTree.query(query_target_points)
            closest_source_points = self.points_source[closest_source_points_index[:,0], :]*self.scale + self.trans

            # Compute current error:
            error = (((query_source_points-closest_target_points)**2).sum() +\
                    ((query_target_points-closest_source_points)**2).sum())\
                    /(query_source_points.shape[0] + query_target_points.shape[0])
            error = error**0.5
            # print(i, "th iter, error: ", error)

            if error < stop_error:
                break

            ''' 
            Build lsq linear system:
            / x1 1 0 0 \  / scale \     / x_t1 \
            | y1 0 1 0 |  |  t_x  |  =  | y_t1 |
            | z1 0 0 1 |  |  t_y  |     | z_t1 | 
            | x2 1 0 0 |  \  t_z  /     | x_t2 |
            | ...      |                | .... |
            \ zn 0 0 1 /                \ z_tn /
            '''
            A_c0 = np.vstack([self.points_source.reshape(-1,1),
                              self.points_source[closest_source_points_index[:,0], :].reshape(-1,1)])
            if i == 0:
                A_c1 = np.zeros((self.points_source.shape[0] + self.points_target.shape[0], 3),\
                            dtype = np.float) + np.array([1.0, 0.0, 0.0])
                A_c1 = A_c1.reshape(-1, 1)
                A_c2 = np.zeros_like(A_c1)
                A_c2[1:,0] = A_c1[0:-1, 0]
                A_c3 = np.zeros_like(A_c1)
                A_c3[2:,0] = A_c1[0:-2, 0]

                self.A_c123 = np.hstack([A_c1, A_c2, A_c3])

            A = np.hstack([A_c0, self.A_c123])
            # print(closest_target_points.reshape(-1,1).shape)
            # print(query_target_points.reshape(-1,1).shape)
            b = np.vstack([closest_target_points.reshape(-1,1),
                                query_target_points.reshape(-1, 1)])
            # print(A.shape)
            # print(b.shape)
            x = np.linalg.lstsq(A,b)
            self.scale = x[0][0]
            self.trans = (x[0][1:]).transpose()

            query_source_points = self.points_source*self.scale + self.trans
            closest_source_points = self.points_source[closest_source_points_index[:,0], :]*self.scale + self.trans
            error = (((query_source_points-closest_target_points)**2).sum() +\
            ((query_target_points-closest_source_points)**2).sum())/(query_source_points.shape[0] + query_target_points.shape[0])
            error = error**0.5
            # print(i, "th iter, error: ", error)
    
    def get_trans_scale(self):
        all_scale = self.scale_target * self.scale / self.scale_source 
        all_trans = self.trans + self.offset_target * self.scale - self.offset_source * self.scale_target * self.scale / self.scale_source
        return all_trans, all_scale

    def export_source_mesh(self, output_name):
        self.mesh_source.vertices = (self.mesh_source.vertices - self.offset_source) / self.scale_source * self.scale_target + self.offset_target
        self.mesh_source.vertices = self.mesh_source.vertices * self.scale + self.trans
        self.mesh_source.export(output_name)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    target_mesh = trimesh.load('icp_tmp_data/00000004_gt.obj', process=False)
    # source_mesh = trimesh.load('00000004_mano.obj', process=False)
    source_mesh = trimesh.load('icp_tmp_data/00000004_hand.ply', process=False)

    icp_solver = ICP_T_S(source_mesh, target_mesh)
    icp_solver.sample_mesh(30000, 'source') # mesh_id = ['source', 'target', 'both'] Please make sure both mesh have similar number of point samples
    icp_solver.run_icp_f(max_iter = 100)
    icp_solver.export_source_mesh('icp_tmp_data/result.obj')
    trans, scale = icp_solver.get_trans_scale()
    print(trans)
    print(scale)
