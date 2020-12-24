import os
import sys
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='combine json files')
    parser.add_argument('--path', type=str,
                        default='/home/admin123/PycharmProjects/DATA/央视视频时序场景检测/12_10/annotation/4_yuxinzou_CCTV_12_10_1_1607612041074')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    file_names = os.listdir(args.path)
    out_json = dict()
    for f in file_names:
        if f.endswith('.json'):
            tmp = json.load(open(os.path.join(args.path, f), 'r'))
            name = list(tmp.keys())[0]
            out_json[name[:-4]] = tmp[name]

    out = os.path.join(args.path, 'cctv_action_detection_12_10.json')

    with open(out, 'w') as outfile:
        json.dump(out_json, outfile, indent=4)


if __name__ == '__main__':
    main()
