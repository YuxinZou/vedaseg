from volkscv.utils.parser import parse_data
from volkscv.analyzer.visualization import visualization
import numpy as np
import random

np.random.seed(0)
categories = ('cat', 'dog')


def test_cls():
    gt_anno = parse_data(format='txt',
                         txt_file='/home/admin123/PycharmProjects/DATA/cls/txt_file_gt.txt',
                         imgs_folder='/home/admin123/PycharmProjects/DATA/cls/imagenet',
                         categories=categories,
                         )
    dt_anno = parse_data(format='txt',
                         txt_file='/home/admin123/PycharmProjects/DATA/cls/txt_file_dt.txt',
                         imgs_folder='/home/admin123/PycharmProjects/DATA/cls/imagenet',
                         categories=categories,
                         )
    print(gt_anno)
    print(dt_anno)
    dt_anno.update({'scores': np.array([0.8] * 10000)})
    vis = visualization(task='cls', gt=gt_anno, pred=dt_anno)
    params = dict(save_folder='./result',
                  # specified_imgs='/home/admin123/PycharmProjects/DATA/cls/cloth_quality/data_split/test/test.txt',
                  show_ori=True,
                  category_to_show=None,
                  show_fpfn=False,
                  show_score=True, )
    # vis.show(**params)
    # vis.save(**params)
    vis.show_single_image(
        fname='/home/admin123/PycharmProjects/DATA/cls/imagenet/cat/ILSVRC2012_val_00049998.JPEG',
        **params)


def test_coco_det():
    gt_anno = parse_data(format='coco',
                         ignore=False,
                         anno_path='instances_val2014_.json',
                         imgs_folder='/home/admin123/PycharmProjects/YuxinZou/data/coco2014/val2014'
                         )
    dt_anno = parse_data(format='coco',
                         ignore=False,
                         anno_path='instances_val2014_.json',
                         imgs_folder='/home/admin123/PycharmProjects/YuxinZou/data/coco2014/val2014'
                         )

    for q, item in enumerate(dt_anno['bboxes']):
        print(len(item))
        for qq, ii in enumerate(item):
            for qqq, iii in enumerate(ii):
                dt_anno['bboxes'][q][qq][qqq] += random.randint(-15, 15)

    dt_anno.update({'scores': [np.array([0.8] * 2), np.array([0.8] * 2),
                               np.array([0.8] * 20), np.array([0.8] * 18)]})
    print(dt_anno)
    print(dt_anno['scores'])
    vis = visualization(task='det',
                        gt=gt_anno,
                        pred=dt_anno,
                        # colors=dict(face=(0, 255, 0)),
                        )

    # set ignore box color, default is reds
    # set matched with ignore box, default is red
    vis.colors = dict(ignore=(0, 0, 255),
                      matched_with_ignore=(0, 0, 255))
    print(vis.colors)
    params = dict(save_folder='./result',
                  # specified_imgs='/home/admin123/PycharmProjects/DATA/face/subfolder',
                  score_thr=0.3,
                  show_fpfn=True,
                  show_fpfn_format='mask',
                  show_ignore=False,
                  iou_thr=0.3,
                  )
    vis.show(**params)
    # vis.save(**params)
    # vis.show_single_image(fname='0_Parade_marchingband_1_20.jpg', **params)


def test_det():
    categories = ['face', ]
    gt_anno = parse_data(format='xml',
                         need_shape=False,
                         ignore=True,
                         txt_file='/home/admin123/PycharmProjects/DATA/face/val.txt',
                         imgs_folder='/home/admin123/PycharmProjects/DATA/face/face',
                         xmls_folder='/home/admin123/PycharmProjects/DATA/face/Annotations',
                         categories=categories,
                         )
    # dt_anno = parse_data(format='mmdet',
    #                      need_shape=False,
    #                      # txt_file='/home/admin123/PycharmProjects/DATA/face/val.txt',
    #                      anno_path='/home/admin123/PycharmProjects/DATA/face/retina_full_photo_bfp_biupsample_ssh_dcn_sgdr_gn.pkl.bbox.json',
    #                      imgs_folder='/home/admin123/PycharmProjects/DATA/face/face',
    #                      categories=categories,
    #                      )

    vis = visualization(task='det',
                        # gt=gt_anno,
                        pred=gt_anno,
                        colors=dict(face=(0, 255, 0)),
                        )

    # set ignore box color, default is reds
    # set matched with ignore box, default is red
    vis.colors = dict(ignore=(0, 0, 255),
                      matched_with_ignore=(0, 0, 255))

    params = dict(save_folder='./result',
                  # specified_imgs='/home/admin123/PycharmProjects/DATA/face/subfolder',
                  show_ori=False,
                  category_to_show=None,
                  show_score=False,
                  score_thr=0.3,
                  show_fpfn=False,
                  show_fpfn_format='mask',
                  show_ignore=True,
                  iou_thr=0.3,
                  )
    vis.show(**params)
    # vis.save(**params)
    # vis.show_single_image(fname='0_Parade_marchingband_1_20.jpg', **params)


def test_seg():
    gt_anno = parse_data(format='coco',
                         ignore=True,
                         anno_path='/home/admin123/PycharmProjects/YuxinZou/data/coco2014/annotations/instances_val2014.json',
                         imgs_folder='/home/admin123/PycharmProjects/YuxinZou/data/coco2014/val2014'
                         )

    dt_anno = parse_data(format='coco',
                         ignore=True,
                         anno_path='/home/admin123/PycharmProjects/YuxinZou/data/coco2014/annotations/instances_val2014.json',
                         imgs_folder='/home/admin123/PycharmProjects/YuxinZou/data/coco2014/val2014'
                         )

    # dt_anno.update({'scores': np.array([[0.98, 0.97]])})
    for q, item in enumerate(dt_anno['segs']):
        for qq, ii in enumerate(item):
            for qqq, i in enumerate(ii):
                for qqqq, k in enumerate(i):
                    dt_anno['segs'][q][qq][qqq][qqqq] += random.randint(-5, 5)

    vis = visualization(task='seg',
                        gt=gt_anno,
                        pred=dt_anno)

    vis.colors.update(dict(text=(0, 0, 255)))

    params = dict(save_folder='./result',
                  category_to_show=None,
                  show_score=False,
                  show_fpfn=True,
                  )
    vis.show(**params)
    # vis.save(**params)
    # vis.show_single_image(fname='COCO_val2014_000000000632.jpg', **params)


def test_xray():
    categories = ['knife', 'scissors', 'lighter', 'zippooil', 'pressure',
                  'slingshot', 'handcuffs', 'nailpolish', 'powerbank',
                  'firecrackers']
    gt_anno = parse_data(format='coco',
                         need_shape=False,
                         ignore=True,
                         anno_path='/home/admin123/PycharmProjects/DATA/xray/chenghao/xray_coco_val.json',
                         imgs_folder='/home/admin123/PycharmProjects/DATA/xray/chenghao/val/image',
                         )

    dt_anno = parse_data(format='mmdet',
                         need_shape=False,
                         anno_path='/home/admin123/PycharmProjects/DATA/xray/chenghao/result_val.bbox.json',
                         imgs_folder='/home/admin123/PycharmProjects/DATA/xray/chenghao/val/image/',
                         categories=categories,
                         )

    vis = visualization(task='det',
                        gt=gt_anno,
                        pred=dt_anno
                        )

    vis.colors.update(dict(ignore=(0, 0, 255),
                           matched_with_ignore=(0, 0, 255),
                           text=(0, 0, 255)))

    params = dict(save_folder='./result',
                  show_ori=False,
                  category_to_show=('knife', 'scissors', 'lighter'),
                  show_score=True,
                  score_thr=0.3,
                  show_fpfn=True,
                  show_fpfn_format='line',
                  show_ignore=True,
                  iou_thr=0.3,
                  )
    # vis.show(**params)
    vis.save(**params)


if __name__ == '__main__':
    # test_coco_det()
    test_det()
    # test_seg()
    # test_xray()
    # test_cls()
