"""
draw yolo bboxes on image
"""
from cv2 import cv2
import glob
import os
import gdal

class Image(object):
    def __init__(self, image_path):
        self.image_path = image_path
        image = cv2.imread(image_path)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.width = image.shape[0]
        self.height = image.shape[1]
    def show_draw_box(self, box):
        """
        opencv显示绘制框
        """
        for bbox in box.bboxs: # 逐个读取bbox，将其绘制在图像上
            _ = bbox['cls']
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']
            absolute_x_center = self.width * x
            absolute_y_center = self.height * y
            absolute_width = self.width * width
            absolute_height = self.height * height
            ul = (int(absolute_x_center - absolute_width / 2), int(absolute_y_center - absolute_height / 2)) # 转成int坐标，cv2.rectangle不接受浮点坐标
            dr = (int(absolute_x_center + absolute_width / 2), int(absolute_y_center + absolute_height / 2))
            image = cv2.putText(self.image, u'Attacked Tree', (ul[0], ul[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 255)) # 绘制文字
            image = cv2.rectangle(image, ul, dr, (0, 0, 255), 2) # 绘制矩形
        cv2.imshow('yolo_bbox', image) # 在image上绘制多个bbox
        cv2.waitKey(0) # 等待用户按键
    def save_draw_box(self, box):
        """
        opencv保存绘制框的图片，用于做样本筛选等工作
        """
        for bbox in box.bboxs: # 逐个读取bbox，将其绘制在图像上
            cls_id = bbox['cls']
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']
            absolute_x_center = self.width * x
            absolute_y_center = self.height * y
            absolute_width = self.width * width
            absolute_height = self.height * height
            ul = (int(absolute_x_center - absolute_width / 2), int(absolute_y_center - absolute_height / 2)) # 转成int坐标，cv2.rectangle不接受浮点坐标
            dr = (int(absolute_x_center + absolute_width / 2), int(absolute_y_center + absolute_height / 2))
            image = cv2.putText(self.image, cls_id, (ul[0], ul[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 255)) # 绘制文字
            image = cv2.rectangle(image, ul, dr, (0, 0, 255), 2) # 绘制矩形
        # cv2.imshow('yolo_bbox', image) # 在image上绘制多个bbox
        # cv2.waitKey(0) # 等待用户按键
        cv2.imwrite(self.image_path[:-4]+'_draw.jpg', image)

class TIFFImage(object):
    def __init__(self, image_path):
        image = gdal.Open(image_path).ReadAsArray().transpose(1,2,0)
        image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        self.image = image
        self.width = image.shape[0]
        self.height = image.shape[1]
    def draw_box(self, box):
        for bbox in box.bboxs: # 逐个读取bbox，将其绘制在图像上
            _ = bbox['cls']
            x = bbox['x']
            y = bbox['y']
            width = bbox['width']
            height = bbox['height']
            absolute_x_center = self.width * x
            absolute_y_center = self.height * y
            absolute_width = self.width * width
            absolute_height = self.height * height
            ul = (int(absolute_x_center - absolute_width / 2), int(absolute_y_center - absolute_height / 2)) # 转成int坐标，cv2.rectangle不接受浮点坐标
            dr = (int(absolute_x_center + absolute_width / 2), int(absolute_y_center + absolute_height / 2))
            print("bbox: %s %s" % (ul, dr))
            # image = cv2.putText(self.image, u'Rubbish', (ul[0], ul[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color=(0, 0, 255)) # 绘制文字
            image = cv2.rectangle(self.image, ul, dr, (0, 0, 255), 2) # 绘制矩形
        # cv2.imshow('yolo_bbox', image) # 在image上绘制多个bbox
        # cv2.imwrite(r'F:\田德宇5st\2020——三室自研\固体废弃物\采样点\影像\白山市\draw.png', image)
        # cv2.waitKey(0) # 等待用户按键

class Box(object):
    def __init__(self, box_path):
        self.bboxs = []
        with open(box_path) as f:
            while True:  # 逐行读取bbox坐标信息
                data = f.readline()
                if data:
                    class_id = data.split(' ')[0]
                    relative_x_center = data.split(' ')[1]
                    relative_y_center = data.split(' ')[2]
                    relative_width = data.split(' ')[3]
                    relative_height = data.split(' ')[4]
                    bbox = {'cls': class_id, 'x': float(relative_x_center), 'y': float(relative_y_center), 'width': float(relative_width), 'height': float(relative_height)}
                    self.bboxs.append(bbox)
                else:
                    break

class FilePath(object):
    def __init__(self, path):
        self.path = path
    def png_file_list(self):
        pngs = glob.glob(os.path.join(self.path,'*.png'))
        return pngs
    def jpeg_file_list(self):
        jpegs = glob.glob(os.path.join(self.path, '*.jpg'))
        return jpegs
    def yolo_file_list(self):
        yolos = glob.glob(os.path.join(self.path, '*.txt'))
        return yolos
    def tiff_image_list(self):
        tifs = glob.glob(os.path.join(self.path, '*.TIF'))
        return tifs


if __name__ == "__main__":
    # fpath = FilePath(r"C:\Users\jadymayor\Desktop\obj\obj")
    fpath = FilePath(r"F:\田德宇5st\2020——三室自研\变色立木\样本\6——剔除nodata和满屏都是变色立木的样本\img")
    type = 'jpg'
    if type == 'jpg':
        jpg_files = fpath.jpeg_file_list()
        for jpg_file in jpg_files:
            image = Image(jpg_file)
            box = Box(jpg_file[:-4]+'.txt')
            image.save_draw_box(box)
    if type == 'png':
        png_files = fpath.png_file_list()
        # yolo_files = fpath.yolo_file_list()
        for png_file in png_files:
            image = Image(png_file)
            box = Box(png_file[:-4]+'.txt')
            image.show_draw_box(box)
    if type == 'tiff':
        tiff_files = fpath.tiff_image_list()
        for tiff_file in tiff_files:
            yolo_file_path = tiff_file[:-4]+'.txt'
            if os.path.exists(yolo_file_path):
                image = TIFFImage(tiff_file)
                box = Box(yolo_file_path)    
                image.draw_box(box)

