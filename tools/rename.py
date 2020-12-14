import os
import shutil


def rename(src, dst):
    os.makedirs(dst, exist_ok=True)

    i = 1
    for item in os.listdir(src):
        fname = os.path.join(src, item)

        dfname = "20201210_%06d.mp4" % i
        dfname = os.path.join(dst, dfname)
        print(dfname)
        i += 1
        os.rename(fname, dfname)


CLASSES = (
'会议讲话', '军演', '阅兵', '采访', '会议', '下飞机', '合影', '火箭发射', '发布会', '握手', '拥抱', '会晤')


def move_img(src, dst):
    for folder in os.listdir(src):
        if folder in CLASSES:
            subfolder = os.path.join(src, folder)
            for item in os.listdir(subfolder):
                fname = os.path.join(subfolder, item)
                shutil.copy(fname, dst)


NAME_LIST = ('/home/admin123/PycharmProjects/DATA/cctv/11_27/11-27',
             '/home/admin123/PycharmProjects/DATA/cctv/12_10/央视')


def check_duplicate(name_list, src):
    set_list = set()
    for i in name_list:
        set_list = set_list | set(os.listdir(i))

    duplicate = set(os.listdir(src)) & set_list
    for i in duplicate:
        print(os.path.join(src, i))
        # os.remove(os.path.join(src, i))


if __name__ == '__main__':
    pass
    # src = "/home/admin123/PycharmProjects/DATA/cctv/12_11/12_11_ori"
    # dst = "/home/admin123/PycharmProjects/DATA/cctv/12_11/cctv_12_11"
    # rename(src, dst)

    # src = "/home/admin123/PycharmProjects/DATA/cctv/12_11/haung"
    # dst = "/home/admin123/PycharmProjects/DATA/cctv/12_11/12_11"
    # move_img(src, dst)

    # src = "/home/admin123/PycharmProjects/DATA/cctv/12_11/12_11_ori"
    # check_duplicate(NAME_LIST, src)
