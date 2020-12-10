#!/usr/bin/env bash


root = /home/admin123/PycharmProjects/github/meizhi/vedaseg/data/thumos14/
videos_root=/home/admin123/PycharmProjects/github/meizhi/vedaseg/data/thumos14/test
save_root=/home/admin123/PycharmProjects/github/meizhi/vedaseg/data/thumos14/images/test/
for video in $videos_root/*;
do
  echo $video
  save_dir=$save_root$(basename $video .mp4)
  echo $save_dir
  if [ ! -d $save_dir ];then
  mkdir $save_dir
  fi

  ffmpeg -i $video -vf fps=10 $save_dir/%06d.png
done