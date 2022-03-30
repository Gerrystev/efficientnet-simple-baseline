import numpy as np
import cv2
import pandas as pd
import os
import json_tricks

# Frame numbers for train/test split
# format: [start_frame, end_frame[ (inclusive, exclusive)
# these index from videopose3d and starting index is n frame manually
# format
DATASET_INDEX = {
    'S1': {
        'ThrowCatch 1': [(5, 473), (528, 984)],     # +55
        'Walking 1': [(80, 590), (665, 1203)],      # +75
        'Box 1': [(55, 435), (435, 789)],           # +50
        'Jog 1': [(50, 367), (412, 740)],           # +45
        'Gestures 1': [(45, 395), (435, 801)],      # +40
    },
    'S2': {
        'Jog 1': [(100, 398), (493, 795)],            # +92 // +88
        'Box 1': [(117, 382), (494, 734)],            # +112
        'Gestures 1': [(122, 500), (617, 901)],       # +617
        'ThrowCatch 1': [(130, 550), (675, 1128)],    # +125
        'Walking 1': [(115, 438), (548, 876)],        # +110
    },
    'S3': {
        'Box 1': [(1, 508), (508, 1017)],           # -4
        'Gestures 1': [(83, 533), (611, 1102)],     # +78
        'Jog 1': [(65, 401), (461, 842)],           # +60
    },
}

# these value for fixing imbalance dataset on test set
# Formula for these values are (30*(validate+train)/100)
TEST_N_FRAME = {
    'S1': {
        'Box 1': 88,
        'ThrowCatch 1': -1,
        'Walking 1': 204,
        'Jog 1': 73,
        'Gestures 1': 143,
    },
    'S2': {
        'Box 1': 70,
        'Gestures 1': 111,
        'ThrowCatch 1': 124,
        'Jog 1': 120,
        'Walking 1': 130,
    },
    'S3': {
        'Box 1': 186,
        'Gestures 1': 39,
        'Jog 1': 141,
    },
}

VALID_N_FRAME = {
    'S1': {
        'Box 1': 71,
        'ThrowCatch 1': 36,
        'Walking 1': 163,
        'Jog 1': 58,
        'Gestures 1': 115,
    },
    'S2': {
        'Box 1': 56,
        'Gestures 1': 89,
        'ThrowCatch 1': 99,
        'Jog 1': 96,
        'Walking 1': 104,
    },
    'S3': {
        'Box 1': 148,
        'Gestures 1': 31,
        'Jog 1': 113,
    },
}

# Frames to skip for each video (synchronization)
SYNC_DATA = {
    'S1': {
        'Walking 1': (82, 81, 82),
        'Jog 1': (51, 51, 50),
        'ThrowCatch 1': (61, 61, 60),
        'Gestures 1': (45, 45, 44),
        'Box 1': (57, 57, 56),
    },
    'S2': {
        'Walking 1': (115, 115, 114),
        'Jog 1': (100, 100, 99),
        'ThrowCatch 1': (127, 127, 127),
        'Gestures 1': (122, 122, 121),
        'Box 1': (119, 119, 117),
    },
    'S3': {
        'Walking 1': (80, 80, 80),
        'Jog 1': (65, 65, 65),
        'Gestures 1': (83, 83, 82),
        'Box 1': (1, 1, 1),
    }
}

def mouse_move(event, x, y, flags, param):
    global mouseX,mouseY
    if event == cv2.EVENT_MOUSEMOVE:
        mouseX,mouseY = x,y
        print(str(mouseX) + ", " + str(mouseY))

# configurate np.load allow_pickle to true
np_load_old = np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

# load dataset from .npz file and convert to dictionary
data = np.load("../data/humaneva/video/annot/data_2d_humaneva15_gt.npz")
metadata = data.f.metadata
pos = data.f.positions_2d
dataset = dict(enumerate(pos.flatten()))[0]

gt_db = []
valid_db = []
test_db = []

json_test = json_tricks.dumps(test_db)

for c in range(3):
    # loop through 3 camera
    # camera index to preview
    camera_index = c
    cam_num = camera_index + 1

    # valid frame value
    valid_frame_df = {'subject': [], 'data_type': [], 'movement': [], 'valid_frame': []}

    for subject, movement in DATASET_INDEX.items():
        # iterate train/ test subject dataset
        subject_name = subject
        for move, idx in movement.items():
            # iterate movement chunks
            video_index = -1
            start_index = -1
            end_index = -1

            train_index = False
            valid_index = False
            is_stop = False

            additional_frame = 0
            valid_frame = 0

            data_type = "Test"
            for i in range(len(idx)):
                # iterate train/ validate start/ end frame
                # start video from starting index
                if subject_name == "S1" and move == "ThrowCatch 1" and i == 1:
                    continue
                if video_index == -1 or train_index:
                    if train_index:
                        data_section = "Train/" + subject_name
                    else:
                        data_section = "Validate/" + subject_name

                    video_index = DATASET_INDEX[subject_name][move][i][0]
                    end_index = DATASET_INDEX[subject_name][move][i][1]

                    if subject_name == "S1" and move == "ThrowCatch 1":
                        data_section = "Train/" + subject_name
                        video_index = DATASET_INDEX[subject_name][move][1][0]
                        end_index = DATASET_INDEX[subject_name][move][1][1]

                    if start_index == -1:
                        # if switch movement/ start video
                        video_name = move.replace(" ", "_") + "_(C" + str(cam_num) + ")"
                        input_video_path = '../data/humaneva/video/' + subject_name + '/Image_Data/' + video_name + '.avi'
                        cap = cv2.VideoCapture(input_video_path)

                cap.set(cv2.CAP_PROP_POS_FRAMES, video_index)
                frame_index = 0
                chunk_index = 0
                current_chunk = None
                ret, frame = cap.read()
                while video_index < end_index:
                    # show current frame from video_index
                    if not is_stop:
                        ret, frame = cap.read()

                    key = cv2.waitKey(1)
                    if key == ord('q'):
                        break
                    if key == ord('p'):
                        is_stop = True
                    if key == ord('x'):
                        is_stop = False
                    if is_stop:
                        if key == ord('w'):
                            ret, frame = cap.read()
                            additional_frame += 1
                            video_index += 1
                            print(additional_frame)
                        if key == ord('s'):
                            video_index -= 1
                            cap.set(cv2.CAP_PROP_POS_FRAMES, video_index)
                            ret, frame = cap.read()
                            additional_frame -= 1
                            print(additional_frame)

                    if ret:
                        if frame_index == 0:
                            # switch to next chunk
                            movement_name = move + " chunk" + str(chunk_index)
                            current_chunk = dataset[data_section][movement_name][camera_index]

                        img = frame
                        if np.isfinite(current_chunk).all():
                            # if current chunk is valid draw the skeleton
                            # get current keypoint from current frame
                            current_keypoint = current_chunk[frame_index]

                            # get keypoint (15 keypoints)
                            pelvis_point = tuple(current_keypoint[0])
                            thorax_point = tuple(current_keypoint[1])
                            left_shoulder_point = tuple(current_keypoint[2])
                            left_elbow_point = tuple(current_keypoint[3])
                            left_wrist_point = tuple(current_keypoint[4])
                            right_shoulder_point = tuple(current_keypoint[5])
                            right_elbow_point = tuple(current_keypoint[6])
                            right_wrist_point = tuple(current_keypoint[7])
                            left_hip_point = tuple(current_keypoint[8])
                            left_knee_point = tuple(current_keypoint[9])
                            left_ankle_point = tuple(current_keypoint[10])
                            right_hip_point = tuple(current_keypoint[11])
                            right_knee_point = tuple(current_keypoint[12])
                            right_ankle_point = tuple(current_keypoint[13])
                            head_point = tuple(current_keypoint[14])

                            # draw line to make a skeleton
                            # color (argument 4 is BGR)
                            # thickness in px
                            thickness = 5

                            img = cv2.line(frame, pelvis_point, thorax_point, (203, 192, 255), thickness)
                            img = cv2.line(img, thorax_point, left_shoulder_point, (0, 165, 255), thickness)
                            img = cv2.line(img, left_shoulder_point, left_elbow_point, (128, 0, 128), thickness)
                            img = cv2.line(img, left_elbow_point, left_wrist_point, (0, 75, 150), thickness)
                            img = cv2.line(img, thorax_point, right_shoulder_point, (0, 255, 255), thickness)
                            img = cv2.line(img, right_shoulder_point, right_elbow_point, (0, 255, 0), thickness)
                            img = cv2.line(img, right_elbow_point, right_wrist_point, (0, 0, 255), thickness)
                            img = cv2.line(img, pelvis_point, left_hip_point, (33, 0, 133), thickness)
                            img = cv2.line(img, left_hip_point, left_knee_point, (0, 76, 255), thickness)
                            img = cv2.line(img, left_knee_point, left_ankle_point, (0, 255, 0), thickness)
                            img = cv2.line(img, pelvis_point, right_hip_point, (248, 0, 252), thickness)
                            img = cv2.line(img, right_hip_point, right_knee_point, (0, 196, 92), thickness)
                            img = cv2.line(img, right_knee_point, right_ankle_point, (0, 238, 255), thickness)
                            img = cv2.line(img, head_point, thorax_point, (255, 0, 0), thickness)

                            # save image from valid frame
                            no_test = False
                            if subject_name == "S1" and move == "ThrowCatch 1":
                                data_type = "Validate"
                                no_test = True
                                valid_index = True
                                if valid_index:
                                    data_type = "Validate"
                                if train_index:
                                    data_type = "Train"

                            if TEST_N_FRAME[subject_name][move] <= valid_frame and not valid_index and not no_test:
                                valid_frame = 0
                                valid_index = True
                                data_type = "Validate"

                            if VALID_N_FRAME[subject_name][move] <= valid_frame and not train_index and valid_index:
                                valid_frame = 0
                                train_index = True
                                data_type = "Train"

                            path = "../data/humaneva/images/C" + str(cam_num) + "/" + subject_name + "/" + move + "/" + data_type
                            if not os.path.exists(path):
                                # check if path exist, if not create new folder
                                os.makedirs(path)
                            image_name = str(valid_frame) + '.jpg'
                            isWrite = cv2.imwrite(os.path.join(path, image_name), img)

                            # write to json
                            gt_coord = dict()
                            gt_coord['joints'] = current_keypoint
                            gt_coord['image'] = os.path.join(path, image_name)

                            if data_type == 'Train':
                                gt_db.append(gt_coord)
                            elif data_type == "Validate":
                                valid_db.append(gt_coord)
                            else:
                                test_db.append(gt_coord)

                            # score valid frame
                            valid_frame += 1
                    else:
                        break

                    cv2.imshow(subject_name + " " + move + " C" + str(cam_num), img)

                    if not is_stop:
                        # continue to next video index & check sync_data index
                        video_index += 1
                        if video_index in SYNC_DATA[subject_name][move]:
                            # if next index is included in sync_data skip again
                            cap.set(cv2.CAP_PROP_POS_FRAMES, video_index)
                            video_index += 1

                        # continue to next frame in a chunk & check if frame index is more than len(chunk)
                        frame_index += 1
                        if frame_index >= len(current_chunk):
                            frame_index = 0
                            chunk_index += 1

                valid_frame_df['subject'].append(subject_name)
                valid_frame_df['data_type'].append(data_type)
                valid_frame_df['movement'].append(move)
                valid_frame_df['valid_frame'].append(valid_frame)

                # switch to train index
                # train_index = True

            # release capture
            cap.release()
            cv2.destroyAllWindows()

    # df = pd.DataFrame(valid_frame_df)
    # df.to_csv("valid_data_C" + str(cam_num) + ".csv", index_label=False)

json_gt = json_tricks.dumps(gt_db)
json_test = json_tricks.dumps(valid_db)
json_test = json_tricks.dumps(test_db)

# Create a Json
with open('../data/humaneva/annot/train.json', 'w') as outfile:
    outfile.write(json_tricks.dumps(gt_db))

with open('../data/humaneva/annot/valid.json', 'w') as outfile:
    outfile.write(json_tricks.dumps(valid_db))

with open('../data/humaneva/annot/test.json', 'w') as outfile:
    outfile.write(json_tricks.dumps(test_db))

# restore np.load for future normal usage
np.load = np_load_old