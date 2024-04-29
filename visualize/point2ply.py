import numpy as np
import torch


def labeled_point2ply(point, point_label, ply_filename,colorMap):  #
    """Save labeled points to disk in colored-point cloud format: x y z r g b, with '.ply' suffix
        vox_labeled.shape: (W, H, D)
    """  
    
    if type(point) is torch.Tensor:
        point = point.numpy()  
        point = point.astype(np.float32) 
    if type(point_label) is torch.Tensor:
        point_label = point_label.numpy()  
        point_label = point_label.astype(np.int32)             
        
            
    # ---- Check data type, numpy ndarray
    if type(point) is not np.ndarray:
        raise Exception("Oops! Type of vox_labeled should be 'numpy.ndarray', not {}.".format(type(point_label)))
    # ---- Check data validation
    if np.amax(point_label) == 0:
        print('Oops! All voxel is labeled empty.')
        return
    
    # ---- Convert to list
    point_label = point_label.flatten()
    # ---- Get X Y Z
    _x = point[:,0].flatten()
    _y = point[:,1].flatten()
    _z = point[:,2].flatten()
    # print('_x.shape', _x.shape)
    # ---- Get R G B
    point_label[point_label == 255] = 0  # empty
    # vox_labeled[vox_labeled == 255] = 12  # ignore
    _rgb = colorMap[point_label[:]]
    # print('_rgb.shape:', _rgb.shape)
    # ---- Get X Y Z R G B
    xyz_rgb = zip(_x, _y, _z, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # python2.7
    xyz_rgb = list(xyz_rgb)  # python3
    # print('xyz_rgb.shape-1', xyz_rgb.shape)
    # xyz_rgb = zip(_z, _y, _x, _rgb[:, 0], _rgb[:, 1], _rgb[:, 2])  # 将X轴和Z轴交换，用于meshlab显示
    # ---- Get ply data without empty voxel

    xyz_rgb = np.array(xyz_rgb)
    # print('xyz_rgb.shape-1', xyz_rgb.shape)
    ply_data = xyz_rgb[np.where(point_label > -1)]

    if len(ply_data) == 0:
        raise Exception("Oops!  That was no valid ply data.")
    ply_head = 'ply\n' \
                'format ascii 1.0\n' \
                'element vertex %d\n' \
                'property float x\n' \
                'property float y\n' \
                'property float z\n' \
                'property uchar red\n' \
                'property uchar green\n' \
                'property uchar blue\n' \
                'end_header' % len(ply_data)
    # ---- Save ply data to disk
    np.savetxt(ply_filename, ply_data, fmt="%f %f %f %d %d %d", header=ply_head, comments='')  # It takes 20s
    del point_label, _x, _y, _z, _rgb, xyz_rgb, ply_data, ply_head
    # print('Saved-->{}'.format(ply_filename))