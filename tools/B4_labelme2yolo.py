# encoding: utf-8
"""
It will create .txt-file for each .jpg-image-file - in the same directory and with the same name, but with .txt-extension, and put to file: object number and object coordinates on this image, for each object in new line:

<object-class> <x_center> <y_center> <width> <height>

Where:

<object-class> - integer object number from 0 to (classes-1)
<x_center> <y_center> <width> <height> - float values relative to width and height of image, it can be equal from (0.0 to 1.0]
for example: <x> = <absolute_x> / <image_width> or <height> = <absolute_height> / <image_height>
atention: <x_center> <y_center> - are center of rectangle (are not top-left corner)
For example for img1.jpg you will be created img1.txt containing:

1 0.716797 0.395833 0.216406 0.147222
0 0.687109 0.379167 0.255469 0.158333
1 0.420312 0.395833 0.140625 0.166667

@version: 3.6
@author: mas
@file: 3_json2coco.py
@time: 2020/6/1 19:24
"""
from utils import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
# 路径相关设置
parser.add_argument('--img_path', type=str,help='影像文件路径', default=r"F:\2021\3——算法研究\车辆检测\1—阿布扎比车辆\labelme\img")
parser.add_argument('--img_prefix', type=str,help='图片后缀', default=r".png")
# parser.add_argument('--json_path', type=str,help='labelme标注json文件路径', default=r"F:\田德宇5st\2020——三室自研\固体废弃物\采样点\bboxs\labelme")
parser.add_argument('--output_dir', type=str,help='文件输出路径', default=r"F:\2021\3——算法研究\车辆检测\1—阿布扎比车辆\labelme\img")

config = parser.parse_args()

CLASSES = {
    "car": 0,
    "track": 1
}

def json2yolotxt(img_name):
    '''

    :param img_name:  
    :return:
    '''
    # 判断文件夹是否存在
    if not osp.exists(config.output_dir):
        os.mkdir(config.output_dir)

    json_path = img_name.replace(config.img_prefix, ".json")
    if not os.path.exists(json_path):
        return
    json_data = json.load(open(json_path))
    objects = json_data["shapes"]
    image_weight = json_data["imageWidth"]
    image_height = json_data["imageHeight"]
    yolo_txt_path = json_path[:-4] + 'txt'
    with open(yolo_txt_path, 'w+') as f:
        for obj in objects:
            cls_id = CLASSES[obj["label"]]
            absolute_x_center = obj["bbox"][0] + obj["bbox"][2] / 2
            absolute_y_center = obj["bbox"][1] + obj["bbox"][3] / 2
            x_center = absolute_x_center / image_weight
            y_center = absolute_y_center / image_height
            width = obj["bbox"][2] / image_weight
            height = obj["bbox"][3] / image_height
            obj_line = str(cls_id) + ' ' + str(x_center) + ' ' + str(y_center) + ' ' + str(width) + ' ' + str(height) + '\n' #注意写入换行转义字符
            f.write(obj_line)


if __name__ == '__main__':
    image_file_list = check_image_list(config.img_path, "*" + config.img_prefix)
    print(f"开始制作yolo数据集。。。")
    for image in image_file_list:
        json2yolotxt(image)