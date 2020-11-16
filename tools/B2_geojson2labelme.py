# encoding: utf-8
"""
@version: 3.6
@author: mas
@file: 2_json2xml.py
@time: 2020/6/1 9:34
给定一个TIF文件夹，一个geojson标注文件，对每个TIF切割成patch，存储包含标注的patch为labelme格式
"""
from utils import *

parser = argparse.ArgumentParser()
# 路径相关设置
parser.add_argument('--img_path', type=str,help='影像文件路径', default=r"F:\田德宇5st\2020——三室自研\固体废弃物\采样点\影像")
parser.add_argument('--geojson_path', type=str,help='样本geojson文件路径', default=r"F:\田德宇5st\2020——三室自研\固体废弃物\采样点\bboxs\bboxes.geojson")
parser.add_argument('--output_dir', type=str,help='文件输出路径', default=r"F:\labelme")
# 裁剪相关参数设置
parser.add_argument('--overlap_ratio', type=float,help='裁剪重叠率',default=0.4)
parser.add_argument('--min_box_area', type=float,help='截断坐标框最小面积(不包含)',default=16.0)
parser.add_argument('--max_size_radio', type=float,help='截断坐标长宽比',default=20)
parser.add_argument('--clip_width', type=float,help='裁剪影像宽',default=512)
parser.add_argument('--clip_height', type=float,help='裁剪影像高',default=512)

config = parser.parse_args()

def is_out_of_image(polygon):
    """
    判断一个imgaug polygon是否在图像上,只有有一部分不在图像上就返回True
    参数：
    polygon————坐标已经shift到当前图像xy
    """
    if np.any(polygon.xx > 1024) or np.any(polygon.xx < 0):
        return True
    elif np.any(polygon.yy > 1024) or np.any(polygon.yy < 0):
        return True
    return False

def IMG_CLIP(image_path, geojson_path):
    '''
    输入一个TIF及对应的geojson，对其进行分块处理，并将包含标注信息的分块存储为labelme格式
    :param image_path:
    :return:
    '''
    # 1 读取数据
    dataset = gdal.Open(image_path)
    if None == dataset:
        print(f"file {image_path} can't open")
        return
    img_width = dataset.RasterXSize  # 栅格矩阵的列数
    img_height = dataset.RasterYSize  # 栅格矩阵的行数
    img_bands = dataset.RasterCount  # 波段数
    img_data_type = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)  # 原始数据类型
    print(image_path, img_bands, img_data_type)

    # 2 计算分块截图的坐标区域
    ################分块大小预处理#####################
    # min_image_size = min(img_height, img_width)  # 最小图像分块判断
    block_size = max(config.clip_height, config.clip_width)
    actual_size = round(block_size * (1 - config.overlap_ratio))
    # g_r = math.ceil(img_height / actual_size)  # 向上取整
    clip_box_list = []
    ######################分块计算#######################
    for r in range(0, img_height - block_size, actual_size):
        for c in range(0, img_width - block_size, actual_size):
            coords = [r, r + block_size, c, c + block_size]
            clip_box_list.append(coords)
        #######右侧边界
        coords = [r, r + block_size, img_width - block_size, img_width]
        clip_box_list.append(coords)
    for c in range(0, img_width - block_size, actual_size):  #########下侧边界
        coords = [img_height - block_size, img_height, c, c + block_size]
        clip_box_list.append(coords)
    ########右下角
    coords = [img_height - block_size, img_height, img_width - block_size, img_width]
    clip_box_list.append(coords)
    print(f"影像{image_path}对应{len(clip_box_list)}个分块")

    # 3 对影像对应的实例分割坐标集进行坐标变换
    with open(geojson_path, 'r') as f:
        instance = json.load(f)
        features_points_list = []
        for feature in instance['features']:
            if feature["geometry"]["type"] != "Polygon" and feature["geometry"]["type"] != "MultiPolygon":
                id_num = feature["properties"]["FID"]
                print(f"标记文件中FID为 {id_num} 的图斑不是Polygon或MultiPolygon格式，请检查！")
            else:
                features_points_list.append(feature["geometry"]['coordinates'][0][0])
    # features_xy_points_list = []
    polygon_list = []
    for feature_points in features_points_list:
        xy_points_list = []
        for lonlat in feature_points:
            xy = lonlat2imagexy(dataset,float(lonlat[0]),float(lonlat[1]))
            xy_points_list.append(xy)
        # features_xy_points_list.append(xy_points_list)
        polygon_list.append(Polygon(xy_points_list))

    # # 检查多边形合法性
    # polygon_list_ = []
    # for polygon in polygon_list:
    #     if not polygon.is_valid:
    #         print(f"{image_path} - 图斑存在问题，请检查")
    #         #return
    #     else:
    #         polygon_list_.append(polygon)
    # polygon_list = polygon_list_

    # 4 根据裁剪框对相应的目标框进行处理
    clip_box_index = 0
    for clip_box in tqdm(clip_box_list):
        xmin = clip_box[2]
        ymin = clip_box[0]
        width = block_size
        height = block_size
        #

        # 存储图片
        out_img_name = osp.basename(image_path).split(".")[0] + "_clip_" + str(clip_box_index) + ".png"
        output_dir = osp.join(config.output_dir, "img")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        out_img_path = osp.join(output_dir, out_img_name)
        # 存储json
        out_json_name = osp.basename(image_path).split(".")[0] + "_clip_" + str(clip_box_index) + ".json"
        output_dir = osp.join(config.output_dir, "img")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        out_json_path = osp.join(output_dir, out_json_name)
        if os.path.exists(out_json_path):
            print(f"文件{out_json_path}已存在，跳过存储")
            clip_box_index += 1
            continue

        # 多边形的坐标平移到当前分块patch
        polygon_list_shift = list(map(lambda x: x.shift(top=-ymin, left=-xmin), polygon_list))

        # 剔除及截断坐标框
        polygon_list_shift_ = []
        for polygon in polygon_list_shift:
            if not is_out_of_image(polygon):
                polygon_list_shift_.append(polygon)
        polygon_list_shift = polygon_list_shift_
        # psoi = ia.PolygonsOnImage(polygon_list_shift,
                                #   shape=(height, width))
        # try:
        #     psoi_aug = psoi.remove_out_of_image_(fully=True, partly=True)
        # except ValueError:
        #     print("移除分块边界外polygon时出现问题，继续下一个分块")
        #     continue
        
        aug_polygon_list = polygon_list_shift

        # # 剔除面积小于阈值的坐标框
        # aug_polygon_large_list = []
        # for polygon in aug_polygon_list:
        #     area = polygon.area
        #     w = max(polygon.xx) - min(polygon.xx)
        #     h = max(polygon.yy) - min(polygon.yy)
        #     long_ = max(w, h)
        #     short_ = min(w, h) + 0.001
        #     # print(w,h,area)
        #     # if (area > config.min_box_area) & ((long_ / short_) < config.max_size_radio):
        #     if (area > config.min_box_area) :
        #         aug_polygon_large_list.append(polygon)
        # aug_polygon_list = aug_polygon_large_list

        polygons_num = len(aug_polygon_list)
        # print(polygons_num)
        if polygons_num == 0:
            # 裁剪后没有多边形 直接跳出 不做文件存储
            print(f"clip{clip_box_index} 裁剪后没有多边形 不做文件存储")
            clip_box_index += 1
            continue
        else:
            print(f"当前分块有{polygons_num}个多边形")
            if not os.path.exists(out_img_path):
                img_data_int8 = dataset.ReadAsArray(xmin, ymin, width, height).astype(np.uint8)  # 获取分块数据
                img_data = np.transpose(img_data_int8, (1, 2, 0))[:, :, [2, 1, 0]]
                skio.imsave(out_img_path, img_data)
            else:
                print(f"文件{out_img_path}已存在，跳过存储")

            if not os.path.exists(out_json_path):
                json_dict = {
                    "version": "4.5.5",
                    "flags": {},
                    "shapes": [],
                    "lineColor": [0, 255, 0, 128],
                    "fillColor": [255, 0, 0, 128],
                    "imagePath": out_img_path,
                    "imageData": "",
                    "imageHeight": height,
                    "imageWidth": width
                }
                # 点位数据
                for aug_polygon in aug_polygon_list:
                    json_shape = {
                            "label": "dapeng",
                            "line_color": None,
                            "fill_color": None,
                            "points": [],
                            "shape_type": "polygon",
                            "flags": {},
                            "area": 0.0,
                            "bbox":[]
                        }
                    xx_list = aug_polygon.xx.tolist()
                    yy_list = aug_polygon.yy.tolist()
                    seg_list = list(chain.from_iterable(zip(xx_list, yy_list)))
                    json_shape["segmentation"] = [seg_list]

                    x_min = min(xx_list)
                    x_max = max(xx_list)
                    y_min = min(yy_list)
                    y_max = max(yy_list)

                    width = x_max - x_min
                    height = y_max - y_min


                    seg_list = []
                    for xx, yy in zip(xx_list, yy_list):
                        seg_list.append([xx,yy])
                    json_shape["points"] = seg_list
                    json_shape["area"] = -1
                    json_shape["bbox"] = [x_min,y_min,width,height]

                    json_dict["shapes"].append(json_shape)
                # # 影像数据
                # img_data_pil = Image.fromarray(img_data)
                # with io.BytesIO() as f:
                #     img_data_pil.save(f, format='PNG')
                #     f.seek(0)
                #     imageData = base64.b64encode(f.read()).decode('utf-8')
                #     json_dict["imageData"] = imageData
                with open(out_json_path, 'w') as output_json_file:
                    json.dump(json_dict, output_json_file, indent=4)
            else:
                print(f"文件{out_json_path}已存在，跳过存储")
            clip_box_index += 1

if __name__ == '__main__':
    tif_file_list = check_image_list(config.img_path,"*.TIF")
    # 0 数据准备
    geojson_path = config.geojson_path
    if not osp.exists(geojson_path):
        print(f"文件 {geojson_path} 不存在，请检查后再运行！")
        exit()
    # 生成tif-json对的list，作为starmap函数的迭代变量，an iterable of [(1,2), (3, 4)] results in [func(1,2), func(3,4)].
    tif_file_list_with_json = [(tif, geojson_path) for tif in tif_file_list]
    
    start_time = time.time()
    print("开始处理。。。")
    pool = multiprocessing.Pool(processes=10)
    with tqdm(total=len(tif_file_list_with_json)) as t:
        for _ in pool.starmap(IMG_CLIP, tif_file_list_with_json):
            t.update(1)
    # IMG_CLIP(tif_file_list[0], geojson_path)
    print("处理完毕~ ~ ~")
    end_time = time.time()
    print(f"共耗时： {(end_time - start_time)} s")






