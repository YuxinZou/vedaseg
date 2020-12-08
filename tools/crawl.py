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

import requests
from tqdm import tqdm
from urllib.parse import urlencode
from urllib.request import urlretrieve


def get_pid(url):
    return_text = requests.get(url).text
    videoCenterIdCompile = r'addVariable\("videoCenterId","(.*?)"\);'
    videoCenterIdpattern = re.compile(videoCenterIdCompile, re.DOTALL)
    searchObj = re.findall(videoCenterIdpattern, return_text)
    return searchObj[0]


def getVideInfo(pid):
    # pid为该页面视频对应的pid
    # pid 从页面中的源代码找到，在<!--repaste.video.code.begin-->和<!--repaste.video.code.end-->之间
    url = "http://vdn.apps.cntv.cn/api/getHttpVideoInfo.do?pid=" + pid
    res = requests.get(url).text
    res = json.loads(res)
    title = res['title']
    video_url = res['video']['chapters4'][0]['url']
    return title, video_url


def save_video(url):
    pid = get_pid(url)
    title, video_url = getVideInfo(pid)
    urlretrieve(video_url, title + ".mp4")
    print("完成保存！")


if __name__ == "__main__":
    url = "http://tv.cctv.com/2018/04/23/VIDE6hYjqaZqGwZkCbUeU0ds180423.shtml"
    save_video(url)
