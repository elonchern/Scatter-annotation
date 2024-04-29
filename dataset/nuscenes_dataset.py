import torch
import numpy as np
import os
import yaml
from PIL import Image
from torch.utils import data
from torchvision import transforms as T
from pathlib import Path
from nuscenes.utils import splits
from torchvision import transforms as T
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points

def absoluteFilePaths(directory, num_vote):
    for dirpath, _, filenames in os.walk(directory):
        filenames.sort()
        for f in filenames:
            for _ in range(num_vote):
                yield os.path.abspath(os.path.join(dirpath, f))


class SemanticKITTI(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            semkittiyaml = yaml.safe_load(stream)

        self.config = config
        self.num_vote = num_vote
        self.learning_map = semkittiyaml['learning_map']
        self.imageset = imageset

        if imageset == 'train':
            split = semkittiyaml['split']['train']
            if config['train_params'].get('trainval', False):
                split += semkittiyaml['split']['valid']
        elif imageset == 'val':
            split = semkittiyaml['split']['valid']
        elif imageset == 'test':
            split = semkittiyaml['split']['test']
        else:
            raise Exception('Split must be train/val/test')

        self.im_idx = []
        self.proj_matrix = {}

        for i_folder in split:
            self.im_idx += absoluteFilePaths('/'.join([data_path, str(i_folder).zfill(2), 'velodyne']), num_vote)
            calib_path = os.path.join(data_path, str(i_folder).zfill(2), "calib.txt")
            calib = self.read_calib(calib_path)
            proj_matrix = np.matmul(calib["P2"], calib["Tr"])
            self.proj_matrix[i_folder] = proj_matrix

        # seg_num_per_class = config['dataset_params']['seg_labelweights']
        # seg_labelweights = seg_num_per_class / np.sum(seg_num_per_class)
        # self.seg_labelweights = np.power(np.amax(seg_labelweights) / seg_labelweights, 1 / 3.0)

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.im_idx)

    @staticmethod
    def read_calib(calib_path):
        """
        :param calib_path: Path to a calibration text file.
        :return: dict with calibration matrices.
        """
        calib_all = {}
        with open(calib_path, 'r') as f:
            for line in f.readlines():
                if line == '\n':
                    break
                key, value = line.split(':', 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        # reshape matrices
        calib_out = {}
        calib_out['P2'] = calib_all['P2'].reshape(3, 4)  # 3x4 projection matrix for left camera
        calib_out['Tr'] = np.identity(4)  # 4x4 matrix
        calib_out['Tr'][:3, :4] = calib_all['Tr'].reshape(3, 4)

        return calib_out

    def __getitem__(self, index):
        raw_data = np.fromfile(self.im_idx[index], dtype=np.float32).reshape((-1, 4))
        origin_len = len(raw_data)
        points = raw_data[:, :3]

        if self.imageset == 'test':
            annotated_data = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
            instance_label = np.expand_dims(np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            annotated_data = np.fromfile(self.im_idx[index].replace('velodyne', 'labels')[:-3] + 'label',
                                         dtype=np.uint32).reshape((-1, 1))
            instance_label = annotated_data >> 16
            annotated_data = annotated_data & 0xFFFF  # delete high 16 digits binary
            annotated_data = np.vectorize(self.learning_map.__getitem__)(annotated_data) # 函数向量化

            if self.config['dataset_params']['ignore_label'] != 0:
                annotated_data -= 1
                annotated_data[annotated_data == -1] = self.config['dataset_params']['ignore_label']

        image_file = self.im_idx[index].replace('velodyne', 'image_2').replace('.bin', '.png')
        image = Image.open(image_file)
        proj_matrix = self.proj_matrix[int(self.im_idx[index][-22:-20])]

        data_dict = {}
        data_dict['xyz'] = points
        data_dict['labels'] = annotated_data.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = raw_data[:, 3:4]
        data_dict['origin_len'] = origin_len
        data_dict['img'] = image
        data_dict['proj_matrix'] = proj_matrix

        return data_dict, self.im_idx[index]
    
    
    
class point_image_dataset_semkitti(data.Dataset):
    def __init__(self, in_dataset, config, loader_config, num_vote=1, trans_std=[0.1, 0.1, 0.1], max_dropout_ratio=0.2):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
        self.rotate_aug = loader_config['rotate_aug']
        self.flip_aug = loader_config['flip_aug']
        self.transform = loader_config['transform_aug']
        self.scale_aug = loader_config['scale_aug']
        self.dropout = loader_config['dropout_aug']
        self.instance_aug = loader_config.get('instance_aug', False)
        self.max_volume_space = config['dataset_params']['max_volume_space']
        self.min_volume_space = config['dataset_params']['min_volume_space']
        self.num_vote = num_vote
        self.trans_std = trans_std
        self.max_dropout_ratio = max_dropout_ratio
        self.debug = config['debug']

        self.bottom_crop = config['dataset_params']['bottom_crop']
        color_jitter = config['dataset_params']['color_jitter']
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None
        self.flip2d = config['dataset_params']['flip2d']
        self.image_normalizer = config['dataset_params']['image_normalizer']

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    @staticmethod
    def select_points_in_frustum(points_2d, x1, y1, x2, y2):
        """
        Select points in a 2D frustum parametrized by x1, y1, x2, y2 in image coordinates
        :param points_2d: point cloud projected into 2D
        :param points_3d: point cloud
        :param x1: left bound
        :param y1: upper bound
        :param x2: right bound
        :param y2: lower bound
        :return: points (2D and 3D) that are in the frustum
        """
        keep_ind = (points_2d[:, 0] > x1) * \
                   (points_2d[:, 1] > y1) * \
                   (points_2d[:, 0] < x2) * \
                   (points_2d[:, 1] < y2)

        return keep_ind

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label']
        sig = data['signal']
        origin_len = data['origin_len']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        point_num = len(xyz)

        # load 2D data
        image = data['img']
        proj_matrix = data['proj_matrix']

        # project points into image
        keep_idx = xyz[:, 0] > 0  # only keep point in front of the vehicle
        points_hcoords = np.concatenate([xyz[keep_idx], np.ones([keep_idx.sum(), 1], dtype=np.float32)], axis=1)
        img_points = (proj_matrix @ points_hcoords.T).T
        img_points = img_points[:, :2] / np.expand_dims(img_points[:, 2], axis=1)  # scale 2D points
        keep_idx_img_pts = self.select_points_in_frustum(img_points, 0, 0, *image.size)
        keep_idx[keep_idx] = keep_idx_img_pts

        # fliplr so that indexing is row, col and not col, row
        img_points = np.fliplr(img_points)
        points_img = img_points[keep_idx_img_pts]

        img_label = labels[keep_idx]
        point2img_index = np.arange(len(labels))[keep_idx]
        feat = np.concatenate((xyz, sig), axis=1)

        img_indices = points_img.astype(np.int64)


        # PIL to numpy
        image = np.array(image, dtype=np.float32, copy=False) / 255.

        # ------------ 可视化点投影到图像上 ----------- #
        
        proj_label = np.zeros((image.shape[0], image.shape[1],1), dtype=np.float32)
        proj_label[img_indices[:,0],img_indices[:,1]] = labels[point2img_index]
        proj_instance_label = np.zeros((image.shape[0], image.shape[1],1), dtype=np.float32)
        proj_instance_label[img_indices[:,0],img_indices[:,1]] = instance_label[point2img_index]
        
        # -------------------------------------------- #           

        data_dict = {}
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        data_dict['img'] = image
        data_dict['img_indices'] = img_indices
        data_dict['img_label'] = img_label
        data_dict['point2img_index'] = point2img_index
        data_dict['proj_label'] = proj_label
        data_dict['proj_instance_label'] = proj_instance_label
        return data_dict
    
    
def collate_fn_default(data):
    point_num = [d['point_num'] for d in data]
    batch_size = len(point_num)
    ref_labels = data[0]['ref_label']
    origin_len = data[0]['origin_len']
    ref_indices = [torch.from_numpy(d['ref_index']) for d in data]
    path = [d['root'] for d in data]
    
    point2img_index_0 = [torch.from_numpy(d['point2img_index_0']).long() for d in data]
    point2img_index_1 = [torch.from_numpy(d['point2img_index_1']).long() for d in data]
    point2img_index_2 = [torch.from_numpy(d['point2img_index_2']).long() for d in data]
    point2img_index_3 = [torch.from_numpy(d['point2img_index_3']).long() for d in data]
    point2img_index_4 = [torch.from_numpy(d['point2img_index_4']).long() for d in data]
    point2img_index_5 = [torch.from_numpy(d['point2img_index_5']).long() for d in data]
    
    img_0 = [torch.from_numpy(d['img_0']) for d in data]
    img_1 = [torch.from_numpy(d['img_1']) for d in data]
    img_2 = [torch.from_numpy(d['img_2']) for d in data]
    img_3 = [torch.from_numpy(d['img_3']) for d in data]
    img_4 = [torch.from_numpy(d['img_4']) for d in data]
    img_5 = [torch.from_numpy(d['img_5']) for d in data]
    
    img_indices_0 = [d['img_indices_0'] for d in data]
    img_indices_1 = [d['img_indices_1'] for d in data]
    img_indices_2 = [d['img_indices_2'] for d in data]
    img_indices_3 = [d['img_indices_3'] for d in data]
    img_indices_4 = [d['img_indices_4'] for d in data]
    img_indices_5 = [d['img_indices_5'] for d in data]
    
    img_label_0 = [torch.from_numpy(d['img_label_0']) for d in data]
    img_label_1 = [torch.from_numpy(d['img_label_1']) for d in data]
    img_label_2 = [torch.from_numpy(d['img_label_2']) for d in data]
    img_label_3 = [torch.from_numpy(d['img_label_3']) for d in data]
    img_label_4 = [torch.from_numpy(d['img_label_4']) for d in data]
    img_label_5 = [torch.from_numpy(d['img_label_5']) for d in data]

    b_idx = []
    for i in range(batch_size):
        b_idx.append(torch.ones(point_num[i]) * i)
    points = [torch.from_numpy(d['point_feat']) for d in data]
    ref_xyz = [torch.from_numpy(d['ref_xyz']) for d in data]
    labels = [torch.from_numpy(d['point_label']) for d in data]
    
    proj_label_0 = [torch.from_numpy(d['proj_label_0']) for d in data]
    proj_label_1 = [torch.from_numpy(d['proj_label_1']) for d in data]
    proj_label_2 = [torch.from_numpy(d['proj_label_2']) for d in data]
    proj_label_3 = [torch.from_numpy(d['proj_label_3']) for d in data]
    proj_label_4 = [torch.from_numpy(d['proj_label_4']) for d in data]
    proj_label_5 = [torch.from_numpy(d['proj_label_5']) for d in data]
    
    proj_instance_label_0 = [torch.from_numpy(d['proj_instance_label_0']) for d in data]
    proj_instance_label_1 = [torch.from_numpy(d['proj_instance_label_1']) for d in data]
    proj_instance_label_2 = [torch.from_numpy(d['proj_instance_label_2']) for d in data]
    proj_instance_label_3 = [torch.from_numpy(d['proj_instance_label_3']) for d in data]
    proj_instance_label_4 = [torch.from_numpy(d['proj_instance_label_4']) for d in data]
    proj_instance_label_5 = [torch.from_numpy(d['proj_instance_label_5']) for d in data]
    
    return {
        'points': torch.cat(points).float(),
        'ref_xyz': torch.cat(ref_xyz).float(),
        'batch_idx': torch.cat(b_idx).long(),
        'batch_size': batch_size,
        'labels': torch.cat(labels).long().squeeze(1),
        'raw_labels': torch.from_numpy(ref_labels).long(),
        'origin_len': origin_len,
        'indices': torch.cat(ref_indices).long(),
        'point2img_index_0': point2img_index_0,
        'point2img_index_1': point2img_index_1,
        'point2img_index_2': point2img_index_2,
        'point2img_index_3': point2img_index_3,
        'point2img_index_4': point2img_index_4,
        'point2img_index_5': point2img_index_5,
        'img_0': torch.stack(img_0, 0).permute(0, 3, 1, 2),
        'img_1': torch.stack(img_1, 0).permute(0, 3, 1, 2),
        'img_2': torch.stack(img_2, 0).permute(0, 3, 1, 2),
        'img_3': torch.stack(img_3, 0).permute(0, 3, 1, 2),
        'img_4': torch.stack(img_4, 0).permute(0, 3, 1, 2),
        'img_5': torch.stack(img_5, 0).permute(0, 3, 1, 2),
        'img_indices_0': img_indices_0,
        'img_indices_1': img_indices_1,
        'img_indices_2': img_indices_2,
        'img_indices_3': img_indices_3,
        'img_indices_4': img_indices_4,
        'img_indices_5': img_indices_5,
        'img_label_0': torch.cat(img_label_0, 0).squeeze(1).long(),
        'img_label_1': torch.cat(img_label_1, 0).squeeze(1).long(),
        'img_label_2': torch.cat(img_label_2, 0).squeeze(1).long(),
        'img_label_3': torch.cat(img_label_3, 0).squeeze(1).long(),
        'img_label_4': torch.cat(img_label_4, 0).squeeze(1).long(),
        'img_label_5': torch.cat(img_label_5, 0).squeeze(1).long(),
        'path': path,
        'proj_label_0': torch.stack(proj_label_0,0).long(),
        'proj_label_1': torch.stack(proj_label_1,0).long(),
        'proj_label_2': torch.stack(proj_label_2,0).long(),
        'proj_label_3': torch.stack(proj_label_3,0).long(),
        'proj_label_4': torch.stack(proj_label_4,0).long(),
        'proj_label_5': torch.stack(proj_label_5,0).long(),
        'proj_instance_label_0': torch.stack(proj_instance_label_0,0).long(),
        'proj_instance_label_1': torch.stack(proj_instance_label_1,0).long(),
        'proj_instance_label_2': torch.stack(proj_instance_label_2,0).long(),
        'proj_instance_label_3': torch.stack(proj_instance_label_3,0).long(),
        'proj_instance_label_4': torch.stack(proj_instance_label_4,0).long(),
        'proj_instance_label_5': torch.stack(proj_instance_label_5,0).long(),
    }


class nuScenes(data.Dataset):
    def __init__(self, config, data_path, imageset='train', num_vote=1):
        if config.debug:
            version = 'v1.0-mini'
            scenes = splits.mini_train
        else:
            if imageset != 'test':
                version = 'v1.0-trainval'
                if imageset == 'train':
                    scenes = splits.train
                else:
                    scenes = splits.val
            else:
                version = 'v1.0-test'
                scenes = splits.test

        self.split = imageset
        with open(config['dataset_params']['label_mapping'], 'r') as stream:
            nuscenesyaml = yaml.safe_load(stream)
        self.learning_map = nuscenesyaml['learning_map']

        self.num_vote = num_vote
        self.data_path = data_path
        self.imageset = imageset
        self.img_view = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT',
                         'CAM_FRONT_LEFT']

        from nuscenes import NuScenes
        self.nusc = NuScenes(version=version, dataroot=data_path, verbose=True)

        self.get_available_scenes()
        available_scene_names = [s['name'] for s in self.available_scenes]
        scenes = list(filter(lambda x: x in available_scene_names, scenes))
        scenes = set([self.available_scenes[available_scene_names.index(s)]['token'] for s in scenes])
        self.get_path_infos_cam_lidar(scenes)

        print('Total %d scenes in the %s split' % (len(self.token_list), imageset))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.token_list)

    def loadDataByIndex(self, index):
        lidar_sample_token = self.token_list[index]['lidar_token']
        lidar_path = os.path.join(self.data_path,
                                  self.nusc.get('sample_data', lidar_sample_token)['filename'])
        raw_data = np.fromfile(lidar_path, dtype=np.float32).reshape((-1, 5))

        if self.split == 'test':
            self.lidarseg_path = None
            annotated_data = np.expand_dims(
                np.zeros_like(raw_data[:, 0], dtype=int), axis=1)
        else:
            lidarseg_path = os.path.join(self.data_path,
                                         self.nusc.get('lidarseg', lidar_sample_token)['filename'])
            annotated_data = np.fromfile(
                lidarseg_path, dtype=np.uint8).reshape((-1, 1))  # label

        pointcloud = raw_data[:, :4]
        sem_label = annotated_data
        # inst_label = np.zeros(pointcloud.shape[0], dtype=np.int32)
        inst_label = annotated_data
        
        return pointcloud, sem_label, inst_label, lidar_sample_token

    def labelMapping(self, sem_label):
        sem_label = np.vectorize(self.map_name_from_general_index_to_segmentation_index.__getitem__)(
            sem_label)  # n, 1
        assert sem_label.shape[-1] == 1
        sem_label = sem_label[:, 0]
        return sem_label

    def loadImage(self, index, image_id):
        cam_sample_token = self.token_list[index]['cam_token'][image_id]
        cam = self.nusc.get('sample_data', cam_sample_token)
        image = Image.open(os.path.join(self.nusc.dataroot, cam['filename']))
        return image, cam_sample_token

    def get_available_scenes(self):
        # only for check if all the files are available
        self.available_scenes = []
        for scene in self.nusc.scene:
            scene_token = scene['token']
            scene_rec = self.nusc.get('scene', scene_token)
            sample_rec = self.nusc.get('sample', scene_rec['first_sample_token'])
            sd_rec = self.nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
            has_more_frames = True
            scene_not_exist = False
            while has_more_frames:
                lidar_path, _, _ = self.nusc.get_sample_data(sd_rec['token'])
                if not Path(lidar_path).exists():
                    scene_not_exist = True
                    break
                else:
                    break

            if scene_not_exist:
                continue
            self.available_scenes.append(scene)

    def get_path_infos_cam_lidar(self, scenes):
        self.token_list = []

        for sample in self.nusc.sample:
            scene_token = sample['scene_token']
            lidar_token = sample['data']['LIDAR_TOP']  # 360 lidar

            if scene_token in scenes:
                for _ in range(self.num_vote):
                    cam_token = []
                    for i in self.img_view:
                        cam_token.append(sample['data'][i])
                    self.token_list.append(
                        {'lidar_token': lidar_token,
                         'cam_token': cam_token}
                    )

    def __getitem__(self, index):
        pointcloud, sem_label, instance_label, lidar_sample_token = self.loadDataByIndex(index)
        sem_label = np.vectorize(self.learning_map.__getitem__)(sem_label)

        # get image feature
        image_id = np.random.randint(6)
        
        image_0, cam_sample_token_0 = self.loadImage(index, 0)
        image_1, cam_sample_token_1 = self.loadImage(index, 1)
        image_2, cam_sample_token_2 = self.loadImage(index, 2)
        image_3, cam_sample_token_3 = self.loadImage(index, 3)
        image_4, cam_sample_token_4 = self.loadImage(index, 4)
        image_5, cam_sample_token_5 = self.loadImage(index, 5)
        
        
        cam_path, boxes_front_cam, cam_intrinsic_0 = self.nusc.get_sample_data(cam_sample_token_0)
        cam_path, boxes_front_cam, cam_intrinsic_1 = self.nusc.get_sample_data(cam_sample_token_1)
        cam_path, boxes_front_cam, cam_intrinsic_2 = self.nusc.get_sample_data(cam_sample_token_2)
        cam_path, boxes_front_cam, cam_intrinsic_3 = self.nusc.get_sample_data(cam_sample_token_3)
        cam_path, boxes_front_cam, cam_intrinsic_4 = self.nusc.get_sample_data(cam_sample_token_4)
        cam_path, boxes_front_cam, cam_intrinsic_5 = self.nusc.get_sample_data(cam_sample_token_5)
        
        
        pointsensor = self.nusc.get('sample_data', lidar_sample_token)
        cs_record_lidar = self.nusc.get('calibrated_sensor',
                                        pointsensor['calibrated_sensor_token'])
        pose_record_lidar = self.nusc.get('ego_pose', pointsensor['ego_pose_token'])
        
        cam_0 = self.nusc.get('sample_data', cam_sample_token_0)
        cam_1 = self.nusc.get('sample_data', cam_sample_token_1)
        cam_2 = self.nusc.get('sample_data', cam_sample_token_2)
        cam_3 = self.nusc.get('sample_data', cam_sample_token_3)
        cam_4 = self.nusc.get('sample_data', cam_sample_token_4)
        cam_5 = self.nusc.get('sample_data', cam_sample_token_5)
        cs_record_cam_0 = self.nusc.get('calibrated_sensor',
                                            cam_0['calibrated_sensor_token'])
        cs_record_cam_1 = self.nusc.get('calibrated_sensor',
                                            cam_1['calibrated_sensor_token'])       
        cs_record_cam_2 = self.nusc.get('calibrated_sensor',
                                            cam_2['calibrated_sensor_token'])
        cs_record_cam_3 = self.nusc.get('calibrated_sensor',
                                            cam_3['calibrated_sensor_token'])  
        cs_record_cam_4 = self.nusc.get('calibrated_sensor',
                                            cam_4['calibrated_sensor_token']) 
        cs_record_cam_5 = self.nusc.get('calibrated_sensor',
                                            cam_5['calibrated_sensor_token'])                                                              
        
        pose_record_cam_0 = self.nusc.get('ego_pose', cam_0['ego_pose_token'])
        pose_record_cam_1 = self.nusc.get('ego_pose', cam_1['ego_pose_token'])
        pose_record_cam_2 = self.nusc.get('ego_pose', cam_2['ego_pose_token'])
        pose_record_cam_3 = self.nusc.get('ego_pose', cam_3['ego_pose_token'])
        pose_record_cam_4 = self.nusc.get('ego_pose', cam_4['ego_pose_token'])
        pose_record_cam_5 = self.nusc.get('ego_pose', cam_5['ego_pose_token'])

        calib_infos_0 = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam_0['translation'],
            "ego2global_rotation_cam": pose_record_cam_0['rotation'],
            "cam2ego_translation": cs_record_cam_0['translation'],
            "cam2ego_rotation": cs_record_cam_0['rotation'],
            "cam_intrinsic": cam_intrinsic_0,
        }
        
        calib_infos_1 = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam_1['translation'],
            "ego2global_rotation_cam": pose_record_cam_1['rotation'],
            "cam2ego_translation": cs_record_cam_1['translation'],
            "cam2ego_rotation": cs_record_cam_1['rotation'],
            "cam_intrinsic": cam_intrinsic_1,
        }        

        calib_infos_2 = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam_2['translation'],
            "ego2global_rotation_cam": pose_record_cam_2['rotation'],
            "cam2ego_translation": cs_record_cam_2['translation'],
            "cam2ego_rotation": cs_record_cam_2['rotation'],
            "cam_intrinsic": cam_intrinsic_2,
        }
        
        calib_infos_3 = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam_3['translation'],
            "ego2global_rotation_cam": pose_record_cam_3['rotation'],
            "cam2ego_translation": cs_record_cam_3['translation'],
            "cam2ego_rotation": cs_record_cam_3['rotation'],
            "cam_intrinsic": cam_intrinsic_3,
        }

        calib_infos_4 = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam_4['translation'],
            "ego2global_rotation_cam": pose_record_cam_4['rotation'],
            "cam2ego_translation": cs_record_cam_4['translation'],
            "cam2ego_rotation": cs_record_cam_4['rotation'],
            "cam_intrinsic": cam_intrinsic_4,
        }
        calib_infos_5 = {
            "lidar2ego_translation": cs_record_lidar['translation'],
            "lidar2ego_rotation": cs_record_lidar['rotation'],
            "ego2global_translation_lidar": pose_record_lidar['translation'],
            "ego2global_rotation_lidar": pose_record_lidar['rotation'],
            "ego2global_translation_cam": pose_record_cam_5['translation'],
            "ego2global_rotation_cam": pose_record_cam_5['rotation'],
            "cam2ego_translation": cs_record_cam_5['translation'],
            "cam2ego_rotation": cs_record_cam_5['rotation'],
            "cam_intrinsic": cam_intrinsic_5,
        }

        data_dict = {}
        data_dict['xyz'] = pointcloud[:, :3]
        data_dict['img_0'] = image_0
        data_dict['img_1'] = image_1
        data_dict['img_2'] = image_2
        data_dict['img_3'] = image_3
        data_dict['img_4'] = image_4
        data_dict['img_5'] = image_5
         
        data_dict['calib_infos_0'] = calib_infos_0
        data_dict['calib_infos_1'] = calib_infos_1
        data_dict['calib_infos_2'] = calib_infos_2
        data_dict['calib_infos_3'] = calib_infos_3
        data_dict['calib_infos_4'] = calib_infos_4
        data_dict['calib_infos_5'] = calib_infos_5
        
        data_dict['labels'] = sem_label.astype(np.uint8)
        data_dict['instance_label'] = instance_label
        data_dict['signal'] = pointcloud[:, 3:4]
        data_dict['origin_len'] = len(pointcloud)

        return data_dict, lidar_sample_token
    
class point_image_dataset_nus(data.Dataset):
    def __init__(self, in_dataset, config):
        'Initialization'
        self.point_cloud_dataset = in_dataset
        self.config = config
        self.ignore_label = config['dataset_params']['ignore_label']
    
        self.debug = config['debug']


    def map_pointcloud_to_image(self, pc, im_shape, info):
        """
        Maps the lidar point cloud to the image.
        :param pc: (3, N)
        :param im_shape: image to check size and debug
        :param info: dict with calibration infos
        :param im: image, only for visualization
        :return:
        """
        pc = pc.copy().T

        # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
        # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
        pc = Quaternion(info['lidar2ego_rotation']).rotation_matrix @ pc
        pc = pc + np.array(info['lidar2ego_translation'])[:, np.newaxis]

        # Second step: transform to the global frame.
        pc = Quaternion(info['ego2global_rotation_lidar']).rotation_matrix @ pc
        pc = pc + np.array(info['ego2global_translation_lidar'])[:, np.newaxis]

        # Third step: transform into the ego vehicle frame for the timestamp of the image.
        pc = pc - np.array(info['ego2global_translation_cam'])[:, np.newaxis]
        pc = Quaternion(info['ego2global_rotation_cam']).rotation_matrix.T @ pc

        # Fourth step: transform into the camera.
        pc = pc - np.array(info['cam2ego_translation'])[:, np.newaxis]
        pc = Quaternion(info['cam2ego_rotation']).rotation_matrix.T @ pc

        # Fifth step: actually take a "picture" of the point cloud.
        # Grab the depths (camera frame z axis points away from the camera).
        depths = pc[2, :]

        # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
        points = view_points(pc, np.array(info['cam_intrinsic']), normalize=True)

        # Cast to float32 to prevent later rounding errors
        points = points.astype(np.float32)

        # Remove points that are either outside or behind the camera.
        mask = np.ones(depths.shape[0], dtype=bool)
        mask = np.logical_and(mask, depths > 0)
        mask = np.logical_and(mask, points[0, :] > 0)
        mask = np.logical_and(mask, points[0, :] < im_shape[1])
        mask = np.logical_and(mask, points[1, :] > 0)
        mask = np.logical_and(mask, points[1, :] < im_shape[0])

        return mask, pc.T, points.T[:, :2]

    def __len__(self):
        'Denotes the total number of samples'
        if self.debug:
            return 100 * self.num_vote
        else:
            return len(self.point_cloud_dataset)

    def __getitem__(self, index):
        'Generates one sample of data'
        data, root = self.point_cloud_dataset[index]

        xyz = data['xyz']
        labels = data['labels']
        instance_label = data['instance_label']
        sig = data['signal']
        sig = data['signal']
        origin_len = data['origin_len']

        # load 2D data
        image_0 = data['img_0']
        calib_infos_0 = data['calib_infos_0']
        image_1 = data['img_1']
        calib_infos_1 = data['calib_infos_1']
        image_2 = data['img_2']
        calib_infos_2 = data['calib_infos_2']
        image_3 = data['img_3']
        calib_infos_3 = data['calib_infos_3']
        image_4 = data['img_4']
        calib_infos_4 = data['calib_infos_4']
        image_5 = data['img_5']
        calib_infos_5 = data['calib_infos_5']

        ref_pc = xyz.copy()
        ref_labels = labels.copy()
        ref_index = np.arange(len(ref_pc))

        point_num = len(xyz)
        
        feat = np.concatenate((xyz, sig), axis=1)


        keep_idx_0, _, points_img_0 = self.map_pointcloud_to_image(
            xyz, (image_0.size[1], image_0.size[0]), calib_infos_0)
        points_img_0 = np.ascontiguousarray(np.fliplr(points_img_0))

        keep_idx_1, _, points_img_1 = self.map_pointcloud_to_image(
            xyz, (image_1.size[1], image_1.size[0]), calib_infos_1)
        points_img_1 = np.ascontiguousarray(np.fliplr(points_img_1))

        keep_idx_2, _, points_img_2 = self.map_pointcloud_to_image(
            xyz, (image_2.size[1], image_2.size[0]), calib_infos_2)
        points_img_2 = np.ascontiguousarray(np.fliplr(points_img_2))
        
        keep_idx_3, _, points_img_3 = self.map_pointcloud_to_image(
            xyz, (image_3.size[1], image_3.size[0]), calib_infos_3)
        points_img_3 = np.ascontiguousarray(np.fliplr(points_img_3))
        
        keep_idx_4, _, points_img_4 = self.map_pointcloud_to_image(
            xyz, (image_4.size[1], image_4.size[0]), calib_infos_4)
        points_img_4 = np.ascontiguousarray(np.fliplr(points_img_4))                

        keep_idx_5, _, points_img_5 = self.map_pointcloud_to_image(
            xyz, (image_5.size[1], image_5.size[0]), calib_infos_5)
        points_img_5 = np.ascontiguousarray(np.fliplr(points_img_5))
        
        points_img_0 = points_img_0[keep_idx_0]
        img_label_0 = labels[keep_idx_0]
        point2img_index_0 = np.arange(len(keep_idx_0))[keep_idx_0]
        img_indices_0 = points_img_0.astype(np.int64)
        
        points_img_1 = points_img_1[keep_idx_1]
        img_label_1 = labels[keep_idx_1]
        point2img_index_1 = np.arange(len(keep_idx_1))[keep_idx_1]
        img_indices_1 = points_img_1.astype(np.int64)
        
        points_img_2 = points_img_2[keep_idx_2]
        img_label_2 = labels[keep_idx_2]
        point2img_index_2 = np.arange(len(keep_idx_2))[keep_idx_2]
        img_indices_2 = points_img_2.astype(np.int64)   
        
        points_img_3 = points_img_3[keep_idx_3]
        img_label_3 = labels[keep_idx_3]
        point2img_index_3 = np.arange(len(keep_idx_3))[keep_idx_3]
        img_indices_3 = points_img_3.astype(np.int64)  
        
        points_img_4 = points_img_4[keep_idx_4]
        img_label_4 = labels[keep_idx_4]
        point2img_index_4 = np.arange(len(keep_idx_4))[keep_idx_4]
        img_indices_4 = points_img_4.astype(np.int64)              

        points_img_5 = points_img_5[keep_idx_5]
        img_label_5 = labels[keep_idx_5]
        point2img_index_5 = np.arange(len(keep_idx_5))[keep_idx_5]
        img_indices_5 = points_img_5.astype(np.int64)                     

        # PIL to numpy
        image_0 = np.array(image_0, dtype=np.float32, copy=False) / 255.
        image_1 = np.array(image_1, dtype=np.float32, copy=False) / 255.
        image_2 = np.array(image_2, dtype=np.float32, copy=False) / 255.
        image_3 = np.array(image_3, dtype=np.float32, copy=False) / 255.
        image_4 = np.array(image_4, dtype=np.float32, copy=False) / 255.
        image_5 = np.array(image_5, dtype=np.float32, copy=False) / 255.

       
        # ------------ 可视化点投影到图像上 ----------- #
        
        proj_label_0 = np.zeros((image_0.shape[0], image_0.shape[1],1), dtype=np.float32)
        proj_label_0[img_indices_0[:,0],img_indices_0[:,1]] = labels[point2img_index_0]
        proj_instance_label_0 = np.zeros((image_0.shape[0], image_0.shape[1],1), dtype=np.float32)
        proj_instance_label_0[img_indices_0[:,0],img_indices_0[:,1]] = instance_label[point2img_index_0]
        
        proj_label_1 = np.zeros((image_1.shape[0], image_1.shape[1],1), dtype=np.float32)
        proj_label_1[img_indices_1[:,0],img_indices_1[:,1]] = labels[point2img_index_1]
        proj_instance_label_1 = np.zeros((image_1.shape[0], image_1.shape[1],1), dtype=np.float32)
        proj_instance_label_1[img_indices_1[:,0],img_indices_1[:,1]] = instance_label[point2img_index_1]        
        
        proj_label_2 = np.zeros((image_2.shape[0], image_2.shape[1],1), dtype=np.float32)
        proj_label_2[img_indices_2[:,0],img_indices_2[:,1]] = labels[point2img_index_2]
        proj_instance_label_2 = np.zeros((image_2.shape[0], image_2.shape[1],1), dtype=np.float32)
        proj_instance_label_2[img_indices_2[:,0],img_indices_2[:,1]] = instance_label[point2img_index_2]        
        
        proj_label_3 = np.zeros((image_3.shape[0], image_3.shape[1],1), dtype=np.float32)
        proj_label_3[img_indices_3[:,0],img_indices_3[:,1]] = labels[point2img_index_3]
        proj_instance_label_3 = np.zeros((image_3.shape[0], image_3.shape[1],1), dtype=np.float32)
        proj_instance_label_3[img_indices_3[:,0],img_indices_3[:,1]] = instance_label[point2img_index_3]        
        
        proj_label_4 = np.zeros((image_4.shape[0], image_4.shape[1],1), dtype=np.float32)
        proj_label_4[img_indices_4[:,0],img_indices_4[:,1]] = labels[point2img_index_4]
        proj_instance_label_4 = np.zeros((image_4.shape[0], image_4.shape[1],1), dtype=np.float32)
        proj_instance_label_4[img_indices_4[:,0],img_indices_4[:,1]] = instance_label[point2img_index_4]        
        
        proj_label_5 = np.zeros((image_5.shape[0], image_5.shape[1],1), dtype=np.float32)
        proj_label_5[img_indices_5[:,0],img_indices_5[:,1]] = labels[point2img_index_5]
        proj_instance_label_5 = np.zeros((image_5.shape[0], image_5.shape[1],1), dtype=np.float32)
        proj_instance_label_5[img_indices_5[:,0],img_indices_5[:,1]] = instance_label[point2img_index_5]        
        # -------------------------------------------- #                
        

        data_dict = {}
        data_dict['point_feat'] = feat
        data_dict['point_label'] = labels
        data_dict['ref_xyz'] = ref_pc
        data_dict['ref_label'] = ref_labels
        data_dict['ref_index'] = ref_index
        # data_dict['mask'] = mask
        data_dict['point_num'] = point_num
        data_dict['origin_len'] = origin_len
        data_dict['root'] = root

        data_dict['img_0'] = image_0
        data_dict['img_indices_0'] = img_indices_0
        data_dict['img_label_0'] = img_label_0
        data_dict['point2img_index_0'] = point2img_index_0
        data_dict['proj_label_0'] = proj_label_0
        data_dict['proj_instance_label_0'] = proj_instance_label_0
        
        data_dict['img_1'] = image_1
        data_dict['img_indices_1'] = img_indices_1
        data_dict['img_label_1'] = img_label_1
        data_dict['point2img_index_1'] = point2img_index_1
        data_dict['proj_label_1'] = proj_label_1
        data_dict['proj_instance_label_1'] = proj_instance_label_1     
        
        data_dict['img_2'] = image_2
        data_dict['img_indices_2'] = img_indices_2
        data_dict['img_label_2'] = img_label_2
        data_dict['point2img_index_2'] = point2img_index_2
        data_dict['proj_label_2'] = proj_label_2
        data_dict['proj_instance_label_2'] = proj_instance_label_2
        
        data_dict['img_3'] = image_3
        data_dict['img_indices_3'] = img_indices_3
        data_dict['img_label_3'] = img_label_3
        data_dict['point2img_index_3'] = point2img_index_3
        data_dict['proj_label_3'] = proj_label_3
        data_dict['proj_instance_label_3'] = proj_instance_label_3
        
        data_dict['img_4'] = image_4
        data_dict['img_indices_4'] = img_indices_4
        data_dict['img_label_4'] = img_label_4
        data_dict['point2img_index_4'] = point2img_index_4
        data_dict['proj_label_4'] = proj_label_4
        data_dict['proj_instance_label_4'] = proj_instance_label_4
        
        data_dict['img_5'] = image_5
        data_dict['img_indices_5'] = img_indices_5
        data_dict['img_label_5'] = img_label_5
        data_dict['point2img_index_5'] = point2img_index_5
        data_dict['proj_label_5'] = proj_label_5
        data_dict['proj_instance_label_5'] = proj_instance_label_5           

        return data_dict