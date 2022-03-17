# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import OrderedDict
import logging
import os
import json_tricks as json

import numpy as np
import pandas as pd

from dataset.JointsDataset import JointsDataset


logger = logging.getLogger(__name__)

HUMANEVA_KEYPOINTS = np.array([
    'pelvis',
    'thorax',
    'lsho',
    'lelb',
    'lwri',
    'rsho',
    'relb',
    'rwri',
    'lhip',
    'lkne',
    'lank',
    'rhip',
    'rkne',
    'rank',
    'head'
])

class HumanEva(JointsDataset):
    def __init__(self, cfg, root, image_set, is_train, transform=None):
        super().__init__(cfg, root, image_set, is_train, transform)

        self.num_joints = 15
        self.flip_pairs = [[13, 10], [12, 9], [11, 8], [7, 4], [6, 3], [5, 2]]    # ankle, knee, hip, wrist, elbow, shoulder [r,l]
        self.parent_ids = [12, 11, 0, 0, 8, 9, 0, 0, 1, 6, 5, 1, 1, 2, 3]

        self.db = self._get_db()

        if is_train and cfg.DATASET.SELECT_DATA:
            self.db = self.select_data(self.db)

        logger.info('=> load {} samples'.format(len(self.db)))

    def _get_db(self):
        # load dataset from .npz file and convert to dictionary
        file_name = os.path.join(self.root, 'annot', self.image_set + '.json')

        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image'].replace('\\', '/' )

            joints_3d = np.zeros((self.num_joints, 3), dtype=np.float)
            joints_3d_vis = np.zeros((self.num_joints, 3), dtype=np.float)
            if self.image_set != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                joints_vis = np.array([1 for i in range(self.num_joints)])
                assert len(joints) == self.num_joints, \
                    'joint num diff: {} vs {}'.format(len(joints),
                                                      self.num_joints)

                joints_3d[:, 0:2] = joints[:, 0:2]
                joints_3d_vis[:, 0] = joints_vis[:]
                joints_3d_vis[:, 1] = joints_vis[:]

            gt_db.append({
                'image': os.path.join(self.root, image_name),
                'joints_3d': joints_3d,
                'joints_3d_vis': joints_3d_vis,
                'filename': '',
                'imgnum': 0,
                })

        return gt_db

    def evaluate(self, epoch, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.json')
            with open(pred_file, 'w') as outfile:
                outfile.write(json.dumps(preds))

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        joint_num = 15
        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               '{}.json'.format(cfg.DATASET.TEST_SET))

        with open(gt_file) as valid_file:
            gt_dict = json.load(valid_file)

        gt = np.array([d['joints'] for d in gt_dict])

        head = np.where(HUMANEVA_KEYPOINTS == 'head')[0][0]
        pelv = np.where(HUMANEVA_KEYPOINTS == 'pelvis')[0][0]
        thor = np.where(HUMANEVA_KEYPOINTS == 'thorax')[0][0]
        lsho = np.where(HUMANEVA_KEYPOINTS == 'lsho')[0][0]
        lelb = np.where(HUMANEVA_KEYPOINTS == 'lelb')[0][0]
        lwri = np.where(HUMANEVA_KEYPOINTS == 'lwri')[0][0]
        lhip = np.where(HUMANEVA_KEYPOINTS == 'lhip')[0][0]
        lkne = np.where(HUMANEVA_KEYPOINTS == 'lkne')[0][0]
        lank = np.where(HUMANEVA_KEYPOINTS == 'lank')[0][0]

        rsho = np.where(HUMANEVA_KEYPOINTS == 'rsho')[0][0]
        relb = np.where(HUMANEVA_KEYPOINTS == 'relb')[0][0]
        rwri = np.where(HUMANEVA_KEYPOINTS == 'rwri')[0][0]
        rkne = np.where(HUMANEVA_KEYPOINTS == 'rkne')[0][0]
        rank = np.where(HUMANEVA_KEYPOINTS == 'rank')[0][0]
        rhip = np.where(HUMANEVA_KEYPOINTS == 'rhip')[0][0]

        # calculate torso diameters
        torsosizesX = np.subtract(gt[:, lsho, 0], gt[:, rhip, 0])
        torsosizesY = np.subtract(gt[:, lsho, 1], gt[:, rhip, 1])

        torsodiameter = np.sqrt(np.power(torsosizesX, 2) + np.power(torsosizesY, 2))
        torsosizes = np.sqrt(np.power(torsosizesX, 2) + np.power(torsosizesY, 2))

        # reshape torsosizes for comparison with distance
        for i in range(joint_num - 1):
            torsosizes = np.vstack((torsosizes, torsodiameter))

        torsosizes = np.swapaxes(torsosizes, 0, 1)

        # calculate pred and gt distance
        distanceX = np.subtract(preds[:, :, 0], gt[:, :, 0])
        distanceY = np.subtract(preds[:, :, 1], gt[:, :, 1])

        distance = np.sqrt(np.power(distanceX, 2) + np.power(distanceY, 2))

        # if distance within threshold is one
        pck = (distance <= 0.2 * torsosizes).astype(int)

        # sum of correct keypoint every joint
        lenPred = len(distance)
        ckAll = pck.sum(axis=0)
        pckAll = np.divide(100 * ckAll, lenPred)

        name_value = [
            ('Epoch', epoch),
            ('Head', pckAll[head]),
            ('Pelvis', pckAll[pelv]),
            ('Thorax', pckAll[thor]),
            ('Shoulder', pckAll[lsho] + pckAll[rsho]),
            ('Elbow', pckAll[lelb] + pckAll[relb]),
            ('Wrist', pckAll[lwri] + pckAll[rwri]),
            ('Hip', pckAll[lhip] + pckAll[rhip]),
            ('Knee', pckAll[lkne] + pckAll[rkne]),
            ('Ankle', pckAll[lank] + pckAll[rank]),
            ('Mean', np.divide(np.sum(pckAll), lenPred * joint_num))
        ]

        df = pd.DataFrame(name_value, columns=[
            'Epoch', 'Head', 'Pelvis', 'Thorax', 'Shoulder', 'Elbow', 'Wrist', 'Hip', 'Knee', 'Ankle', 'Mean'
        ])

        df.to_csv('validation.csv', mode='a', header=not os.path.exists('validation.csv'))

        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
