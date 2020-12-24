import os
import json
import argparse
import pickle

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont


def cv2ImgAddText(img, texts, middle, top, text_color, text_size):
    top_ = top
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    texts = set(texts)
    for text in list(texts):
        # STHeiti.ttf  msjh.ttf
<<<<<<< HEAD
        fontText = ImageFont.truetype("/DATA/home/yuxinzou/yangshi/vedaseg/tools/STHeiti.ttf", text_size, encoding="utf-8")
=======
        fontText = ImageFont.truetype("STHeiti.ttf", text_size,
                                      encoding="utf-8")
>>>>>>> 49c210114351b6583fc6c7d8b0d16585df30d2f8
        text_w, text_h = fontText.getsize(text)
        x1 = middle - int(text_w / 2)
        draw.text((x1, top_), text, text_color, font=fontText)
        top_ += text_h
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)


class Video:
    CLASSES = ('Embrace', 'ShakeHands', 'Meeting', 'ReleaseConference',
               'Conference', 'Photograph', 'TakeOffPlane', 'MilitaryParade',
               'MilitaryExercise', 'RocketLaunching', 'ConferenceSpeech',
               'Interview')

    MAP = {'Embrace': '拥抱',
           'ShakeHands': '握手',
           'Meeting': '会晤',
           'ReleaseConference': '发布会',
           'Conference': '会议',
           'Photograph': '合影',
           'TakeOffPlane': '下⻜机',
           'MilitaryParade': '阅兵',
           'MilitaryExercise': '军演',
           'RocketLaunching': '火箭发射',
           'ConferenceSpeech': '会议讲话',
           'Interview': '采访',
           'Ignore': '忽略',
           }

    def __init__(self, root, save_path, json_file, pickle_file):
        self.root = root
        self.save_path = save_path
        os.makedirs(self.save_path, exist_ok=True)
        self.gt = json.load(open(json_file, 'r'))
        self.data = pickle.load(open(pickle_file, 'rb'))

    def add_sound(self, ori, need_sound, result_path):
        command = f"ffmpeg -i {need_sound} -i {ori} -c copy {result_path}"
        print('command is:', command)
        os.system(command)

    def generate_single(self, id, score=0.2):
        video_path = os.path.join(self.root, id + '.mp4')
        save_path = os.path.join(self.save_path, id + '.mp4')
        gt = self.gt[id]
        dt = self.data[id]
        duration_second = gt['duration_second']
        gt = gt['annotations']
        for an in dt:
            an['segment'][0] *= duration_second
            an['segment'][1] *= duration_second
            an['label'] = self.CLASSES[an['label']]

        # anno = [x for x in anno if x['score'] > score]

        input_movie = cv2.VideoCapture(video_path)

        fps = input_movie.get(cv2.CAP_PROP_FPS)
        size = (int(input_movie.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(input_movie.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        w, h = size

        # define the type of the output movie
        output_movie = cv2.VideoWriter(save_path,
                                       cv2.VideoWriter_fourcc('M', 'J', 'P',
                                                              'G'), fps, size)

        word_x_gt = int(w / 2) - 100
        word_y_gt = 0
        word_x_dt = int(w / 2) + 100
        word_y_dt = 0
        while True:
            # Grab a single frame of video
            ret, frame = input_movie.read()
            if ret:
                current_time = input_movie.get(cv2.CAP_PROP_POS_MSEC) / 1000
                frame = self.write_text(current_time, frame, gt, word_x_gt,
                                        word_y_gt, color=(255, 0, 0))
                frame = self.write_text(current_time, frame, dt, word_x_dt,
                                        word_y_dt, color=(0, 0, 255))
                output_movie.write(frame)
            else:
                output_movie.release()
                break

    def write_text(self, current_time, frame, anno, word_x, word_y, color):
        indexs = [i for i in range(len(anno)) if
                  anno[i]['segment'][0] < current_time <
                  anno[i]['segment'][1]]
        if len(indexs):
<<<<<<< HEAD
            texts = [f"{anno[i]['label']}" for i in indexs]
=======
            texts = [f"{self.MAP[anno[i]['label']]}" for i in indexs]
>>>>>>> 49c210114351b6583fc6c7d8b0d16585df30d2f8
            frame = cv2ImgAddText(frame, texts,
                                  word_x, word_y,
                                  color, 50)
        return frame

    def generate_all(self, score=0.2):
<<<<<<< HEAD
        ids = os.listdir(self.root)
        for id in tqdm(ids):
            id, _ = os.path.splitext(id)
=======
        ids = list(self.data.keys())
        for id in tqdm(ids):
>>>>>>> 49c210114351b6583fc6c7d8b0d16585df30d2f8
            self.generate_single(id, score)

    def find_video(self):
        pass


def parse_args():
    parser = argparse.ArgumentParser(description='Video label')
    parser.add_argument('--root', type=str,
<<<<<<< HEAD
                        default='/DATA/data/public/TAD/cctv/11_27/videos')
    parser.add_argument('--json_file', type=str,
                        default='/DATA/data/public/TAD/cctv/11_27/cctv_action_detection_11_27.json')
    parser.add_argument('--pickle_file', type=str,
                        default='../train_result.pickle')
    parser.add_argument('--save_path', type=str,
                        default='../data/cctv/result/')
=======
                        default='../data/cctv/11-27/cctv_11_27/')
    parser.add_argument('--json_file', type=str,
                        default='../data/cctv/11-27/cctv_action_detection_11_27.json')
    parser.add_argument('--pickle_file', type=str,
                        default='../data/cctv/train_result.pickle')
    parser.add_argument('--save_path', type=str,
                        default='../data/cctv/11-27/result/')
>>>>>>> 49c210114351b6583fc6c7d8b0d16585df30d2f8

    args = parser.parse_args()
    return args


def get_video_shape(path):
    widths, heights = [], []
    for name in os.listdir(path):
        print(name)
        video_name = os.path.join(path, name)
        cap = cv2.VideoCapture(video_name)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        widths.append(width)
        heights.append(height)
    print(widths)
    print(heights)
    print(np.min(widths), np.min(heights))
    print(np.max(widths), np.max(heights))
    print(np.mean(widths), np.mean(heights))


if __name__ == '__main__':
    args = parse_args()
    video = Video(args.root, args.save_path, args.json_file, args.pickle_file)
    # print(video.data)
    #  007112e808f054701bbf5874c0dc250b
<<<<<<< HEAD
    video.generate_single('20201127_000004')
    # video.generate_all(score=0.2)

=======
    video.generate_single('20201127_000160')
    # video.generate_all(score=0.2)
>>>>>>> 49c210114351b6583fc6c7d8b0d16585df30d2f8
