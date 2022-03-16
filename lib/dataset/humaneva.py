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
import re

import numpy as np
from scipy.io import loadmat, savemat

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
    'lknee',
    'lankl',
    'rhip',
    'rknee',
    'rankl',
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

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        # convert 0-based index to 1-based index
        preds = preds[:, :, 0:2] + 1.0

        if output_dir:
            pred_file = os.path.join(output_dir, 'pred.json')
            with open(pred_file, 'w') as outfile:
                outfile.write(json_tricks.dumps(preds))

        if 'test' in cfg.DATASET.TEST_SET:
            return {'Null': 0.0}, 0.0

        SC_BIAS = 0.6
        threshold = 0.5

        gt_file = os.path.join(cfg.DATASET.ROOT,
                               'annot',
                               '{}.json'.format(cfg.DATASET.TEST_SET))

        with open(file_name) as valid_file:
            gt_dict = json.load(valid_file)

        pos_gt_src = np.array([d['joints'] for d in gt_dict])
        print("<= PREDICTION")
        print(preds)
        pos_pred_src = np.transpose(preds, [1, 2, 0])

        head = np.where(HUMANEVA_KEYPOINTS == 'head')[0][0]
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

        torsosizes = pos_gt_src[lsho]
        uv_error = pos_pred_src - pos_gt_src
        uv_err = np.linalg.norm(uv_error, axis=1)
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= SC_BIAS
        scale = np.multiply(headsizes, np.ones((len(uv_err), 1)))
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        jnt_count = np.sum(jnt_visible, axis=1)
        less_than_threshold = np.multiply((scaled_uv_err <= threshold),
                                          jnt_visible)
        PCKh = np.divide(100.*np.sum(less_than_threshold, axis=1), jnt_count)

        # save
        rng = np.arange(0, 0.5+0.01, 0.01)
        pckAll = np.zeros((len(rng), 16))

        for r in range(len(rng)):
            threshold = rng[r]
            less_than_threshold = np.multiply(scaled_uv_err <= threshold,
                                              jnt_visible)
            pckAll[r, :] = np.divide(100.*np.sum(less_than_threshold, axis=1),
                                     jnt_count)

        PCKh = np.ma.array(PCKh, mask=False)
        PCKh.mask[6:8] = True

        jnt_count = np.ma.array(jnt_count, mask=False)
        jnt_count.mask[6:8] = True
        jnt_ratio = jnt_count / np.sum(jnt_count).astype(np.float64)

        name_value = [
            ('Head', PCKh[head]),
            ('Shoulder', 0.5 * (PCKh[lsho] + PCKh[rsho])),
            ('Elbow', 0.5 * (PCKh[lelb] + PCKh[relb])),
            ('Wrist', 0.5 * (PCKh[lwri] + PCKh[rwri])),
            ('Hip', 0.5 * (PCKh[lhip] + PCKh[rhip])),
            ('Knee', 0.5 * (PCKh[lkne] + PCKh[rkne])),
            ('Ankle', 0.5 * (PCKh[lank] + PCKh[rank])),
            ('Mean', np.sum(PCKh * jnt_ratio)),
            ('Mean@0.1', np.sum(pckAll[11, :] * jnt_ratio))
        ]
        name_value = OrderedDict(name_value)

        return name_value, name_value['Mean']
