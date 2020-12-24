# import requests, re, json, os
# from urllib.parse import urlencode
# from urllib.request import urlretrieve
# import pprint
# import re
#
#
# def get_pid(url):
#     # url为该页面的网址
#     return_text = requests.get(url).text
#     # searchObj = re.findall(r'<!--repaste.video.code.begin-->(.+?)<!--repaste.video.code.end-->', return_text)
#     searchObj = re.findall(r'videoCenterId: "(.+?)"', return_text)
#     return searchObj[0]
#
#
# def getVideInfo(pid):
#     # pid为该页面视频对应的pid
#     # pid 从页面中的源代码找到，在<!--repaste.video.code.begin-->和<!--repaste.video.code.end-->之间
#     url = "http://vdn.apps.cntv.cn/api/getHttpVideoInfo.do?pid=" + pid
#     res = requests.get(url).text
#     res = json.loads(res)
#     title = res['title']
#     video_url = res['video']['chapters4'][0]['url']
#     return title, video_url
#
#
# def save_video(url):
#     pid = get_pid(url)
#     title, video_url = getVideInfo(pid)
#     urlretrieve(video_url, title + ".mp4")
#     print("完成保存！")
#
#
# if __name__ == "__main__":
#     url = 'http://tv.cctv.com/2018/04/21/VIDEcTSu2OR2GLjhHniIHT9T180421.shtml'
#     save_video(url)
#     print("=" * 79)

import os
import re
import json
import pprint
import pandas as pd

import requests
from tqdm import tqdm
from urllib.parse import urlencode
from urllib.request import urlretrieve


def get_pid(url):
    return_text = requests.get(url).text
    videoCenterIdCompile = r'addVariable\("videoCenterId","(.*?)"\);'
    videoCenterIdpattern = re.compile(videoCenterIdCompile, re.DOTALL)
    searchObj = re.findall(videoCenterIdpattern, return_text)
    if len(searchObj) == 0:
        searchObj = re.findall(r'videoCenterId: "(.+?)"', return_text)
    print(searchObj)
    return searchObj


def getVideInfo(pid):
    # pid为该页面视频对应的pid
    # pid 从页面中的源代码找到，在<!--repaste.video.code.begin-->和<!--repaste.video.code.end-->之间
    url = "http://vdn.apps.cntv.cn/api/getHttpVideoInfo.do?pid=" + pid
    res = requests.get(url).text
    res = json.loads(res)
    title, video_url = None, None
    try:
        title = res['title']
        video_url = res['video']['chapters4'][0]['url']
    except:
        pass
    return title, video_url


def save_video(url, folder):
    pid = get_pid(url)
    if len(pid) > 0:
        pid = pid[0]
    else:
        return
    title, video_url = getVideInfo(pid)
    if title is None:
        return
    save_path = os.path.join(folder, title + ".mp4")
    if os.path.exists(save_path):
        return
    try:
        urlretrieve(video_url, os.path.join(folder, title + ".mp4"))
    except:
        pass
    print("完成保存！")


def analysis_xlsx(path):
    cctv = {}
    data = pd.read_excel(io=path)
    data = data.to_dict()
    for k, v in data.items():
        cctv[k] = []
        for k_, v_ in v.items():
            if not pd.isna(v_):
                cctv[k].append(v_)
    return cctv


if __name__ == "__main__":
    # url = 'https://tv.cctv.com/2020/06/16/VIDEcnBYoqvMwVpZ4adytVxB200616.shtml'
    # save_video(url)
    root = '/home/admin123/PycharmProjects/github/meizhi/vedaseg/data'
    path = '/home/admin123/PycharmProjects/DATA/央视视频时序场景检测/12-09/excel/视频链接(1).xlsx'
    cctv = analysis_xlsx(path)
    i = 0
    for k, v in cctv.items():
        folder = os.path.join(root, k)
        os.makedirs(folder, exist_ok=True)
        for v_ in v:
            i += 1
            print(v_)
            # save_video(v_, folder)
    print(i)