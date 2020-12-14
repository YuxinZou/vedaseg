import os
import sys
import subprocess
from multiprocessing import Pool
from tqdm import tqdm
import json

num_worksers = 20


def flv2mp4(src, dst):
    cmd = f'ffmpeg -i {src} -strict -2 {dst}'
    subprocess.call(cmd, shell=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL)


def main(path, save_path):
    i = 0
    for subfolder in os.listdir(path):
        sub = os.path.join(path, subfolder)
        for item in os.listdir(sub):
            print(item)
            print(i)
            i += 1
            os.makedirs(os.path.join(save_path, subfolder), exist_ok=True)
            fname = os.path.join(sub, item)
            save_fname = os.path.join(save_path, subfolder,
                                      item.split('.')[0] + '.mp4')
            print(fname)
            print(save_fname)
            flv2mp4(fname, save_fname)


if __name__ == '__main__':
    path = '/home/admin123/PycharmProjects/DATA/央视视频时序场景检测/b站视频_2020_12_07'
    save_path = '/home/admin123/PycharmProjects/DATA/b站视频_2020_12_07'
    main(path, save_path)
