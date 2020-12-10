import os
import sys
import argparse
import json


def parse_args():
    parser = argparse.ArgumentParser(description='combine json files')
    parser.add_argument('--path', type=str)
    parser.add_argument('--out', type=str)
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    file_names = os.listdir(args.path)
    out_json = dict()
    for f in file_names:
        if f.endswith('.json'):
            tmp = json.load(open(os.path.join(args.path, f),'r'))
            name = list(tmp.keys())[0]
            out_json[name[:-4]] = tmp[name]

    with open(args.out, 'w') as outfile:
        json.dump(out_json, outfile)

if __name__ == '__main__':
    main()
