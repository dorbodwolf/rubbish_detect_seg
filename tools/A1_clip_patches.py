# encoding: utf-8
"""
@version: 3.6
@author: mas
@file: 2_json2xml.py
@time: 2020/6/1 9:34
"""
from utils import *

parser = argparse.ArgumentParser()
# 路径相关设置
parser.add_argument('--img_path', type=str,help='图像数据路径', default=r"F:\田德宇5st\2020——三室自研\固体废弃物\相关工作\gago\BDCI\BDCI\0_data\CCF-training\data")
parser.add_argument('--mask_path', type=str,help='标签掩膜路径', default=r"F:\田德宇5st\2020——三室自研\固体废弃物\相关工作\gago\BDCI\BDCI\0_data\CCF-training\mask_bmp")
parser.add_argument('--img_ext', type=str,help='图像数据后缀名称', default="png")
parser.add_argument('--mask_ext', type=str,help='标签掩膜后缀名称', default="bmp")

parser.add_argument('--output_suffix', type=str,help='文件输出前缀', default="ccf")
parser.add_argument('--output_ext', type=str,help='文件输出后缀名称', default="png")
parser.add_argument('--output_dir', type=str,help='文件输出路径', default=r"F:\田德宇5st\2020——三室自研\固体废弃物\数据集")

# 裁剪相关参数设置
parser.add_argument('--overlap_ratio', type=float,help='裁剪重叠率',default=0.4)
parser.add_argument('--min_box_area', type=float,help='截断坐标框最小面积(不包含)',default=16.0)
parser.add_argument('--max_size_radio', type=float,help='截断坐标长宽比',default=20)
parser.add_argument('--clip_width', type=float,help='裁剪影像宽',default=1024)
parser.add_argument('--clip_height', type=float,help='裁剪影像高',default=1024)

config = parser.parse_args()

def IMG_CLIP(path_pair):
    '''
    对影像和掩膜切块
    :param image_path:
    :return:
    '''
    # 1 读取数据
    image_path = list(path_pair.keys())[0]
    dataset = gdal.Open(image_path)
    if None == dataset:
        print(f"file {image_path} can't open")
        return
    img_width = dataset.RasterXSize  # 栅格矩阵的列数
    img_height = dataset.RasterYSize  # 栅格矩阵的行数
    img_bands = dataset.RasterCount  # 波段数
    img_data_type = gdal.GetDataTypeName(dataset.GetRasterBand(1).DataType)  # 原始数据类型

    mask_path = path_pair[image_path]
    mask = gdal.Open(mask_path)
    if None == mask:
        print(f"file {mask_path} can't open")
        return
    mask_width = mask.RasterXSize  # 栅格矩阵的列数
    mask_height = mask.RasterYSize  # 栅格矩阵的行数
    mask_bands = mask.RasterCount  # 波段数
    mask_data_type = gdal.GetDataTypeName(mask.GetRasterBand(1).DataType)  # 原始数据类型    
    print(image_path, img_bands, img_data_type)
    print(mask_path, mask_bands, mask_data_type)

    if img_width == mask_width and img_height == mask_height:
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

        # 3 保存切块
        for clip_box_index, clip_box in tqdm(enumerate(clip_box_list)):
            xmin = clip_box[2]
            ymin = clip_box[0]
            width = block_size
            height = block_size

            if not os.path.exists(config.output_dir):
                os.mkdir(config.output_dir)

            # 存储图像   
            out_img_name = config.output_suffix + "_" + osp.basename(image_path).split(".")[0] + "_patch_" + str(clip_box_index) + "_im." + config.output_ext
            out_img_path = osp.join(config.output_dir, out_img_name)         
            if not os.path.exists(out_img_path):
                img_data_int8 = dataset.ReadAsArray(xmin, ymin, width, height).astype(np.uint8)  # 获取分块数据
                img_data = np.transpose(img_data_int8, (1, 2, 0))[:, :, [2, 1, 0]]
                skio.imsave(out_img_path, img_data)
            else:
                print(f"文件{out_img_path}已存在，跳过存储")
            # 存储mask
            out_mask_name = config.output_suffix + "_" + osp.basename(mask_path).split(".")[0] + "_patch_" + str(clip_box_index) + "_mask." + config.output_ext
            out_mask_path = osp.join(config.output_dir, out_mask_name)     
            if not os.path.exists(out_mask_path):
                mask_data_int8 = mask.ReadAsArray(xmin, ymin, width, height).astype(np.uint8)  # 获取分块数据
                # mask_data = np.transpose(mask_data_int8, (1, 2, 0))[:, :, [2, 1, 0]]
                skio.imsave(out_mask_path, mask_data_int8)
            else:
                print(f"文件{out_img_path}已存在，跳过存储")
            # break
    else:
        print(f"图像{image_path}和掩膜{mask_path}尺寸不一样，请检查！")
        return
    

if __name__ == '__main__':
    image_lists = list_images(config.img_path, config.img_ext)
    mask_lists = list_images(config.mask_path, config.mask_ext)
    if len(image_lists) == len(mask_lists):
        image_mask_pair = [{image_lists[i]:mask_lists[i]} for i in range(len(image_lists))]

        start_time = time.time()
        print("开始处理。。。")

        pool = multiprocessing.Pool(processes=5)
        with tqdm(total=len(image_lists)) as t:
            for _ in pool.imap_unordered(IMG_CLIP,image_mask_pair):
                t.update(1)
        print("处理完毕~ ~ ~")
        end_time = time.time()
        print(f"共耗时： {(end_time - start_time)} s")






