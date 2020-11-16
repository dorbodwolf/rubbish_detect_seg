import os
import os.path as osp
# import geopandas as gpd
from collections import defaultdict
import json
from json import dumps
import argparse
import time
import glob
import multiprocessing
import imgaug as ia
from imgaug import augmenters as iaa
from imgaug.augmentables.polys import Polygon,PolygonsOnImage
# from tools.pascal_voc_io import *
import numpy as np
from skimage import io as skio
from tqdm import tqdm
from pathlib import Path,PurePosixPath
# import shapefile
import collections
from itertools import chain
from PIL import Image
import base64
import io
import datetime
# from sklearn.model_selection import train_test_split
import shutil
from osgeo import gdal,ogr,osr


def list_images(image_folder,suffix):
    '''
    列出指定后缀的图像
    '''

    tiff_image_list = [name for name in glob.glob(image_folder+'/*.'+suffix)]
    return tiff_image_list

def check_image_list(image_folder,suffix):
    '''
    检查tiff文件列表 剔除不在json文件中tiff文件
    :param tiff_image_list: 待处理tiff文件夹
    :param json_path: 记录矩形框坐标信息的json文件
    :return: 待处理的tiff文件
    '''
    # tiff_image_list = glob.glob(osp.join(config.img_folder, '*' + config.suffix))
    #tiff_image_list = search(tiff_image_folder, config.suffix)

    path_list_=Path(image_folder)
    tiff_image_list = [str(i) for i in path_list_.rglob(suffix)]
    # print(tiff_image_list)
    return tiff_image_list

def lonlat2imagexy(dataset,lon,lat):
    '''
    根据地理坐标(经纬度)转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param lon: 经度坐标
    :param lat: 纬度坐标
    :return: 地理坐标(lon,lat)对应的影像图上行列号(row, col)
    '''

    transform = dataset.GetGeoTransform()

    x_origin = transform[0]
    # print(x_origin)
    y_origin = transform[3]
    # print(y_origin)
    pixel_width = transform[1]
    # print(pixel_width)
    pixel_height = transform[5]
    # print(pixel_height)

    x_pix = int((lon - x_origin) / pixel_width + 0.5)
    y_pix = int((lat - y_origin) / pixel_height + 0.5)

    return [x_pix, y_pix]

def create_image_info(image_id, file_name, image_size,
                      date_captured=datetime.datetime.utcnow().isoformat(' '),
                      license_id=1, coco_url="", flickr_url=""):
    image_info = {
            "id": image_id,
            "file_name": file_name,
            "width": image_size[0],
            "height": image_size[1],
            "date_captured": date_captured,
            "license": license_id,
            "coco_url": coco_url,
            "flickr_url": flickr_url
    }

    return image_info