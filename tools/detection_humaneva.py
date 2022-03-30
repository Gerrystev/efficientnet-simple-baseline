from __future__ import absolute_import

import os
import numpy as np
import cv2
import json_tricks as json

# yolo library
import _init_paths
from yolov4.tool.utils import *
from yolov4.tool.torch_utils import *
from yolov4.tool.darknet2pytorch import Darknet
import torch

"""hyper parameters"""
use_cuda = True

if __name__ == "__main__":
    root_dir = '../data/humaneva/annot/'
    file_names = ['train.json', 'test.json', 'valid.json']     # change name to test, train, valid as requirement

    for file in file_names:
        file_name = root_dir + file
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        # load yolo weight
        cfgfile = './cfg/yolov4-tiny.cfg'
        weightfile = '../models/weights/yolov4/yolov4-tiny.weights'

        m = Darknet(cfgfile)

        m.print_network()
        m.load_weights(weightfile)
        print('Loading weights from %s... Done!' % (weightfile))

        if use_cuda:
            m.cuda()

        num_classes = m.num_classes
        namesfile = 'data/coco.names'
        class_names = load_class_names(namesfile)

        gt_db = []
        for a in anno:
            image_name = a['image'].replace('\\', '/' )
            imgfile = '../data/humaneva/' + image_name

            img = cv2.imread(imgfile)
            sized = cv2.resize(img, (m.width, m.height))
            sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

            for i in range(2):
                start = time.time()
                boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)
                finish = time.time()
                if i == 1:
                    print('%s: Predicted in %f seconds.' % (imgfile, (finish - start)))

            if len(boxes) > 1:
                raise Exception("Boxes more than 1")

            # calculate scaling and center
            width = img.shape[1]
            height = img.shape[0]

            box = boxes[0][0]
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            centerx, centery = (np.average(np.array([x1,x2])), np.average(np.array([y1,y2])))
            center = np.array([centerx, centery])
            maxWH = max([abs(x2-x1), abs(y2-y1)])
            scale = maxWH / 200.0

            joints = a['joints']
            joints_vis = np.ones(15, dtype=np.int)

            path_save = image_name.split('/')
            path_save.pop()
            path_save = '/'.join(path_save)

            if not os.path.exists(path_save):
                # check if path exist, if not create new folder
                os.makedirs(path_save)

            plot_boxes_cv2(img, boxes[0], savename=image_name, class_names=class_names)

            gt_db.append({
                'joints_vis': joints_vis,
                'joints': joints,
                'image': image_name,
                'center': center,
                'scale': scale
            })

        json_gt = json.dumps(gt_db)

        # Create a Json
        with open(file, 'w') as outfile:
            outfile.write(json.dumps(gt_db))