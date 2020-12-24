import os
import json
import csv
import random


def write(path):
    global sum
    global list_sum
    print(path)
    data = json.load(open(path, 'r'))
    data = str(data)
    fnames = data.split("', '")
    print(len(fnames))
    fnames = list(set(fnames))
    print(len(fnames))
    fnames = list(set([f.split(']')[-1] for f in fnames]))

    random.shuffle(fnames)
    print(len(fnames))
    sum += len(fnames)
    list_sum.extend(fnames)


#     save_path = path[:-5] + '.txt'
#     file_write_obj = open(save_path, 'w')
#     for var in fnames:
#         if var != '':
#             file_write_obj.writelines(var)
#             file_write_obj.write('\n')
#     file_write_obj.close()

sum = 0
list_sum = []
path = '/home/admin123/Downloads/title/'
for fname in os.listdir(path):
    fname = os.path.join(path, fname)
    write(fname)
print(sum)
print(len(list(set(list_sum))))