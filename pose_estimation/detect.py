# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import cv2
import numpy as np

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.inference import get_final_preds
from utils.utils import create_logger
from utils.transforms import get_affine_transform
from utils.transforms import flip_back

import models


def _box2cs(box, image_width, image_height):
    x, y, w, h = box[:4]
    return _xywh2cs(x, y, w, h, image_width, image_height)


def _xywh2cs(x, y, w, h, image_width, image_height):
    center = np.zeros((2), dtype=np.float32)
    center[0] = x + w * 0.5
    center[1] = y + h * 0.5

    aspect_ratio = image_width * 1.0 / image_height
    pixel_std = 200

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    scale = np.array(
        [w * 1.0 / pixel_std, h * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def draw_skeleton(preds, img):
    pred = preds[0,:, 0:2] + 1.0
    pred = np.round(pred).astype(int)

    # get keypoint (15 keypoints)
    pelvis_point = tuple(pred[6])
    thorax_point = tuple(pred[7])
    left_shoulder_point = tuple(pred[13])
    left_elbow_point = tuple(pred[14])
    left_wrist_point = tuple(pred[15])
    right_shoulder_point = tuple(pred[12])
    right_elbow_point = tuple(pred[11])
    right_wrist_point = tuple(pred[10])
    left_hip_point = tuple(pred[3])
    left_knee_point = tuple(pred[4])
    left_ankle_point = tuple(pred[5])
    right_hip_point = tuple(pred[2])
    right_knee_point = tuple(pred[1])
    right_ankle_point = tuple(pred[0])
    head_point = tuple(pred[9])

    # draw line to make a skeleton
    # color (argument 4 is BGR)
    # thickness in px
    thickness = 5

    img_skel = cv2.line(img, pelvis_point, thorax_point, (203, 192, 255), thickness)
    img_skel = cv2.line(img_skel, thorax_point, left_shoulder_point, (0, 165, 255), thickness)
    img_skel = cv2.line(img_skel, left_shoulder_point, left_elbow_point, (128, 0, 128), thickness)
    img_skel = cv2.line(img_skel, left_elbow_point, left_wrist_point, (0, 75, 150), thickness)
    img_skel = cv2.line(img_skel, thorax_point, right_shoulder_point, (0, 255, 255), thickness)
    img_skel = cv2.line(img_skel, right_shoulder_point, right_elbow_point, (0, 255, 0), thickness)
    img_skel = cv2.line(img_skel, right_elbow_point, right_wrist_point, (0, 0, 255), thickness)
    img_skel = cv2.line(img_skel, pelvis_point, left_hip_point, (33, 0, 133), thickness)
    img_skel = cv2.line(img_skel, left_hip_point, left_knee_point, (0, 76, 255), thickness)
    img_skel = cv2.line(img_skel, left_knee_point, left_ankle_point, (0, 255, 0), thickness)
    img_skel = cv2.line(img_skel, pelvis_point, right_hip_point, (248, 0, 252), thickness)
    img_skel = cv2.line(img_skel, right_hip_point, right_knee_point, (0, 196, 92), thickness)
    img_skel = cv2.line(img_skel, right_knee_point, right_ankle_point, (0, 238, 255), thickness)
    img_skel = cv2.line(img_skel, head_point, thorax_point, (255, 0, 0), thickness)

    cv2.imwrite('predictions.jpg', img_skel)

def detect_cv2(model, imgfile, flip_pairs):
    img = cv2.imread(imgfile)

    data_numpy = cv2.imread(imgfile, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    if data_numpy is None:
        print('=> fail to read {}'.format(imgfile))
        raise ValueError('Fail to read {}'.format(imgfile))

    c = np.array([data_numpy.shape[1] / 2.0, data_numpy.shape[0] / 2.0], dtype='float32')
    max_wh = max([data_numpy.shape[1] / 200.0, data_numpy.shape[0] / 200.0])
    s = np.array([max_wh, max_wh], dtype='float32')
    r = 0

    # c = np.array([img.shape[0]/2.0, img.shape[1]/2.0], dtype='float32')
    # s = np.array([img.shape[1]/200.0, img.shape[1]/200.0], dtype='float32')
    # r = 0

    trans = get_affine_transform(c, s, r, config.MODEL.IMAGE_SIZE)
    input = cv2.warpAffine(
        data_numpy,
        trans,
        (int(config.MODEL.IMAGE_SIZE[0]), int(config.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)

    # vis transformed image
    cv2.imshow('image', input)
    cv2.waitKey(0)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    input = transform(input).unsqueeze(0)

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = model(input)

        # compute coordinate
        preds, maxvals = get_final_preds(
            config, output.clone().cpu().numpy(), np.asarray([c]), np.asarray([s]))

        # plot
        image = data_numpy.copy()
        for mat in preds[0]:
            x, y = int(mat[0]), int(mat[1])
            cv2.circle(image, (x, y), 2, (255, 0, 0), 2)

        # vis result
        cv2.imshow('res', image)
        cv2.waitKey(0)

    # # ===================================
    # convert_tensor = transforms.ToTensor()
    # input = convert_tensor(img_rgb)
    #
    # # add one more dimension for tensor input
    # input = input[None, :, :, :]
    # with torch.no_grad():
    # output = model(input)
    #
    # # this part is ugly, because pytorch has not supported negative index
    # # input_flipped = model(input[:, :, :, ::-1])
    # input_flipped = np.flip(input.cpu().numpy(), 3).copy()
    # input_flipped = torch.from_numpy(input_flipped).cuda()
    # output_flipped = model(input_flipped)
    # output_flipped = flip_back(output_flipped.cpu().numpy(),
    #                            flip_pairs)
    # output_flipped = torch.from_numpy(output_flipped.copy()).cuda()
    #
    # # feature is not aligned, shift flipped heatmap for higher accuracy
    # if config.TEST.SHIFT_HEATMAP:
    #     output_flipped[:, :, :, 1:] = \
    #         output_flipped.clone()[:, :, :, 0:-1]
    #     # output_flipped[:, :, :, 0] = 0
    #
    # output = (output + output_flipped) * 0.5
    #
    # c = np.array([[img.shape[0]/2.0, img.shape[1]/2.0]], dtype='float32')
    # s = np.array([[img.shape[1]/200.0, img.shape[1]/200.0]], dtype='float32')
    #
    # preds, maxvals = get_final_preds(
    #     config, output.clone().cpu().numpy(), c, s)

    draw_skeleton(preds, img)

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=True,
                        type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument('--frequent',
                        help='frequency of logging',
                        default=config.PRINT_FREQ,
                        type=int)
    parser.add_argument('--gpus',
                        help='gpus',
                        type=str)
    parser.add_argument('--workers',
                        help='num of dataloader workers',
                        type=int)
    parser.add_argument('--model-file',
                        help='model state file',
                        type=str)
    parser.add_argument('--use-detect-bbox',
                        help='use detect bbox',
                        action='store_true')
    parser.add_argument('--flip-test',
                        help='use flip test',
                        action='store_true')
    parser.add_argument('--post-process',
                        help='use post process',
                        action='store_true')
    parser.add_argument('--shift-heatmap',
                        help='shift heatmap',
                        action='store_true')
    parser.add_argument('--coco-bbox-file',
                        help='coco detection bbox file',
                        type=str)
    parser.add_argument('--imgfile',
                        help='cropped person image file',
                        type=str)

    args = parser.parse_args()

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.workers:
        config.WORKERS = args.workers
    if args.use_detect_bbox:
        config.TEST.USE_GT_BBOX = not args.use_detect_bbox
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.coco_bbox_file:
        config.TEST.COCO_BBOX_FILE = args.coco_bbox_file


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    model = eval('models.'+config.MODEL.NAME+'.get_pose_net')(
        config, is_train=False
    )
    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        model.load_state_dict(torch.load(config.TEST.MODEL_FILE))
    else:
        model_state_file = os.path.join(final_output_dir,
                                        'final_state.pth.tar')
        logger.info('=> loading model from {}'.format(model_state_file))
        model.load_state_dict(torch.load(model_state_file))

    gpus = [int(i) for i in config.GPUS.split(',')]
    model = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # get flip_pairs from dataset.___.flip_pairs
    # in this case took flip_pairs from mpii
    flip_pairs = [[0, 5], [1, 4], [2, 3], [10, 15], [11, 14], [12, 13]]
    detect_cv2(model, args.imgfile, flip_pairs)


if __name__ == '__main__':
    main()
