'''
Author: deyu
Date: 2020-12-11 16:51:43
LastEditTime: 2020-12-16 15:44:13
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /darknet/darknet_tiles.py
'''

import argparse
import darknet
import os
import glob
import random
import time
import cv2
import numpy as np
from multiprocessing import Pool
import time
from osgeo import gdal, ogr, osr
import torch
import torchvision

parser = argparse.ArgumentParser(description="YOLO Object Detection of GDAL TILES")
parser.add_argument("--input", type=str, default="/home/asd/Mission/Rubbish/tiles/tile_K52E012004.TIF",
                    help="GDAL tiles or folder of GDAL tiles, now support .tif .tiff .TIF .TIFF")
parser.add_argument("--batch_size", default=1, type=int,
                    help="number of images to be processed at the same time")
parser.add_argument("--weights", default="data/backup/yolo-obj_best.weights",
                    help="yolo weights path")
parser.add_argument("--config_file", default="cfg/yolo-obj.cfg",
                    help="path to config file")
parser.add_argument("--data_file", default="/home/asd/Project/darknet/data/obj.data",
                    help="path to data file")
parser.add_argument("--thresh", type=float, default=0.5,
                    help="remove detections with lower confidence")
parser.add_argument("--sliding_size", type=int, default=608,
                    help="滑窗尺寸大小")
parser.add_argument("--sliding_step", type=int, default=256,
                    help="滑窗步长")
parser.add_argument("--nms_thresh", type=float, default=0.4,
                    help="滑窗重叠区域预测后处理的nms阈值")
args = parser.parse_args()

# 1 load network and weights
network, class_names, _ = darknet.load_network(
    args.config_file,
    args.data_file,
    args.weights,
    batch_size=args.batch_size
)

def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    # if args.input and not os.path.exists(args.input):
    #     raise(ValueError("Invalid image path {}".format(os.path.abspath(args.input))))
    # invalid sliding size and step inputs
    if args.sliding_step < 1 or args.sliding_step > args.sliding_size:
        raise(ValueError("滑窗步长不能大于窗口尺寸也不能小于1，请重新输入！"))


def load_tiles(tiles_path):
    """
    加载分幅数据。可以接受：
        单独的tile文件 
        包含tile路径的txt文件
        包含tile的文件夹路径
    """
    input_path_extension = tiles_path.split('.')[-1]
    if input_path_extension in ['tif', 'tiff', 'TIF', 'TIFF']:
        return [tiles_path]
    elif input_path_extension == "txt":
        with open(tiles_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(tiles_path, "*.tif")) + \
            glob.glob(os.path.join(tiles_path, "*.tiff")) + \
            glob.glob(os.path.join(tiles_path, "*.TIF")) + \
               glob.glob(os.path.join(tiles_path, "*.TIFF"))

def image_detection(image_data):
    
    # 2 set threshold
    thresh = args.thresh
    
    # 3 convert numpy array to darknet image
    
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image_rgb = image_data
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    
    # 4 do detection
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    return detections, class_names
    # image = darknet.draw_boxes(detections, image_resized, class_colors)
    # return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def save_shapefile(boxes, keep, shapefile_output, data_header):
    dataset = EsriPolygonVectorIO(shapefile_output)
    dataset.createTextField('class')
    dataset.createTextField('score')
    for i, bbox in enumerate(boxes):
        if i in keep:    
            dataset.createFeature()
            dataset.featureSetField('class', bbox[0])
            dataset.featureSetField('score', float(bbox[5]))
            tl = (bbox[1], bbox[2])
            br = (bbox[3], bbox[4])
            tr = (bbox[3], bbox[2])
            bl = (bbox[1], bbox[4])
            wkt_bbox = bbox2wkt(tl, tr, br, bl, data_header)
            dataset.featureSetGeometry(wkt_bbox)
            dataset.featureWriteGeometry()


def post_nms_process(bbox_txt_output):
    # xyxy_list = []
    # score_list = []
    bbox_list = []
    with open(bbox_txt_output, 'r') as f:
        lines = f.read().splitlines()
        # print(len(lines))
        for line in lines:
            bbox_list.append(line.split(' ')[:])
        #     xyxy_list.append(line.split(' ')[1:5])
        #     score_list.append(line.split(' ')[5])
        bboxs = np.asarray(bbox_list, dtype=np.float)
        xyxy_arr = bboxs[:, 1:5]
        score_arr = bboxs[:, 5]
        xyxy_tensor = torch.tensor(xyxy_arr)
        score_tensor = torch.tensor(score_arr)
        keep = torchvision.ops.nms(xyxy_tensor, score_tensor, iou_threshold=args.nms_thresh)
        keep_arr = torch.Tensor.numpy(keep)
    return bboxs, keep_arr


class EsriPolygonVectorIO:
    def __init__(self, shapefile_output):
        shp_driver = ogr.GetDriverByName('Esri Shapefile')
        self.dataset = shp_driver.CreateDataSource(shapefile_output)
        self.layer = self.dataset.CreateLayer('', None, ogr.wkbPolygon)
        self.layer_defn = self.layer.GetLayerDefn()
        self.feat = None
    def createTextField(self, field_name):
        self.layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTString))
    def createIntergerField(self, field_name):
        self.layer.CreateField(ogr.FieldDefn(field_name, ogr.OFTInteger))
    def createFloatField(self, field_name):
        self.layer.CreateField(ogr.FieldDefn(field_name, ogr.OFSTFloat32))
    def createFeature(self):
        self.feat = ogr.Feature(self.layer_defn)
    def featureSetField(self, field_name, value):
        self.feat.SetField(field_name, value)
    def featureSetGeometry(self, wkt_bbox):
        self.feat.SetGeometry(ogr.CreateGeometryFromWkt(wkt_bbox))
    def featureWriteGeometry(self):
        self.layer.CreateFeature(self.feat)

def bbox2wkt(tl, tr, br, bl, data_header):
    """
    bbox转成wkt描述的多边形
    """
    # posX = px_w * x + rot1 * y + xoffset
    # posY = rot2 * x + px_h * y + yoffset
    tl_x = (data_header['px_w'] * tl[0] + data_header['rot1'] * tl[1] + data_header['xoffset'])
    tl_y = (data_header['rot2'] * tl[0] + data_header['px_h'] * tl[1] + data_header['yoffset'])
    tl = (tl_x, tl_y)
    tr_x = (data_header['px_w'] * tr[0] + data_header['rot1'] * tr[1] + data_header['xoffset'])
    tr_y = (data_header['rot2'] * tr[0] + data_header['px_h'] * tr[1] + data_header['yoffset'])
    tr = (tr_x, tr_y)
    bl_x = (data_header['px_w'] * bl[0] + data_header['rot1'] * bl[1] + data_header['xoffset'])
    bl_y = (data_header['rot2'] * bl[0] + data_header['px_h'] * bl[1] + data_header['yoffset'])
    bl = (bl_x, bl_y)
    br_x = (data_header['px_w'] * br[0] + data_header['rot1'] * br[1] + data_header['xoffset'])
    br_y = (data_header['rot2'] * br[0] + data_header['px_h'] * br[1] + data_header['yoffset'])
    br = (br_x, br_y)
    wkt_bbox = "POLYGON ((%s %s, %s %s, %s %s, %s %s, %s %s))" % (tl_x, tl_y, tr_x, tr_y, br_x, br_y, bl_x, bl_y, tl_x, tl_y)
    return wkt_bbox

def process_tile(tile):
    """
    对输入的切片进行处理
    """
    # 1 gdal read image data
    data_obj = gdal.Open(tile)
    if data_obj == None:
        print("无法加载%s" % (tile))
        return
    
    data_header = {'width': data_obj.RasterXSize, 
                   'height': data_obj.RasterYSize, 
                   'band': data_obj.RasterCount, 
                   'dtype': gdal.GetDataTypeName(data_obj.GetRasterBand(1).DataType),
                   'xoffset': data_obj.GetGeoTransform()[0],
                   'px_w': data_obj.GetGeoTransform()[1],
                   'rot1': data_obj.GetGeoTransform()[2],
                   'yoffset': data_obj.GetGeoTransform()[3], 
                   'rot2': data_obj.GetGeoTransform()[4],
                   'px_h': data_obj.GetGeoTransform()[5]
                   }
    
    # 2 sliding window, do detection and save 'xyxy' format detection result
    window_size = args.sliding_size
    step = args.sliding_step    
    bbox_txt_output = tile.split(".")[0] + '.txt'
    with open(bbox_txt_output, 'w') as f:
        for ycurse in range(0, data_header['height'], step):
            # y方向游标越界，结束外层循环
            if ycurse + window_size > data_header['height']:
                break
            for xcurse in range(0, data_header['width'], step):
                # 数据读取游标越界，就结束内层循环
                if xcurse + window_size > data_header['width']:
                    break
                # 读取一个块，做检测，提取检测结果
                else:
                    chunk = data_obj.ReadAsArray(xcurse, ycurse, window_size, window_size).transpose(1, 2, 0)
                    # print(xcurse, ycurse)
                    detections, class_names = image_detection(chunk)
                    for label, score, bbox in detections:
                        label = class_names.index(label)
                        score = float(score)
                        # chunk的bbox center坐标
                        x, y, w, h = bbox
                        # bbox center坐标对应到tile
                        xtile = x + xcurse
                        ytile = y + ycurse
                        
                        # 计算bbox的四角点坐标
                        tl = (xtile - w / 2, ytile - h / 2)
                        tr = (xtile + w / 2, ytile - h / 2)
                        bl = (xtile - w / 2, ytile + h / 2)
                        br = (xtile + w / 2, ytile + h / 2)
                    
                        # # 转成相对坐标
                        # xtile /= data_header['width']
                        # w /= data_header['width']
                        # ytile /= data_header['height']
                        # h /= data_header['height']
                        # 写入 txt, 按照'xyxy'格式
                        f.write("%i %.4f %.4f %.4f %.4f %.4f\n" % (label, tl[0], tl[1], br[0], br[1], score))

    # 3 对滑窗预测重叠区域的重复预测结果进行nms处理
    boxes, keep = post_nms_process(bbox_txt_output)
    # 4 输出shapefile
    shapefile_output = tile.split(".")[0] + '.shp'
    save_shapefile(boxes, keep, shapefile_output, data_header)
    
def main():
    check_arguments_errors(args)
    
    # 1 获取tiles为列表
    tiles = load_tiles(args.input)

    # 2 利用多进程方式同时对多个tile处理
    start_time = time.time()
    print("开启多个进程来检测分幅数据。。。。。")
    # 获取os进程数
    process_count = 24 # 192.168.20.122机器包含32个核心
    # 创建进程池
    with Pool(processes=process_count) as pool:
        pool.imap_unordered(process_tile, tiles)
    # 关闭进程池
    pool.close()
    print("所有分幅在多进程模式下处理完毕！")
    # process_tile(tiles[0])
    end_time = time.time()
    print("共耗时%s ms" % ((end_time-start_time) / 1000))

if __name__ == "__main__":
    main()