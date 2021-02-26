# encoding: utf-8
"""
@version: 3.6
@author: mas
@file: 3_json2coco.py
@time: 2020/6/1 19:24
"""
from utils import *
from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
# 路径相关设置
parser.add_argument('--img_path', type=str,help='影像文件路径', default=r"F:\labelme\img")
parser.add_argument('--json_path', type=str,help='labelme标注json文件路径', default="/home/sjss/Mession/HZDP/DP_data/new_output/img")
parser.add_argument('--output_dir', type=str,help='文件输出路径', default=r"F:\labelme\coco")
# 裁剪相关参数设置
parser.add_argument('--min_greenhouse_area', type=float,default=0.0)
parser.add_argument('--val_ratio', type=float, help='验证样本比例', default=0.1)
parser.add_argument('--clip_width', type=float,help='裁剪影像宽',default=512)
parser.add_argument('--clip_height', type=float,help='裁剪影像高',default=512)

config = parser.parse_args()

INFO = {
    "description": "JL01 CCGreenhouse Dataset",
    "url": "",
    "version": "0.1.0",
    "year": 20200713,
    "contributor": "mjy",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}
LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://www.mama.com"
    }
]
CATEGORIES = [
    {
        'id': 1,
        'name': 'greenhouse',
        'supercategory': 'greenhouse',
    },
]

def json2cocodataset(imgs,type_):
    '''

    :param imgs:
    :param type:
    :return:
    '''
    print(f"开始制作 {type_} 对应coco格式数据集。。。")
    # 判断文件夹是否存在
    if not osp.exists(config.output_dir):
        os.mkdir(config.output_dir)

    # 最终放进json文件里的字典
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],  # 放一个空列表占位置，后面再append
        "annotations": []
    }
    segmentation_id = 0
    for i in tqdm(range(len(imgs))):
        image_id = i
        img_name = imgs[i]
        # 生成 segmentation 相关信息
        json_path = img_name.replace(".png", ".json")
        if not os.path.exists(json_path):
            continue
        json_data = json.load(open(json_path))
        image_info = create_image_info(image_id, img_name, [config.clip_width, config.clip_height])
        coco_output["images"].append(image_info)

        seg_list = json_data["shapes"]
        seg_num = len(seg_list)


        for seg_index in range(seg_num):
            annotation = {
                "id": 0,
                "image_id": 0,
                "category_id": 1,
                "iscrowd": 0,
                "area": 0.0,
                "bbox": [],
                "segmentation": [],
                "width": config.clip_width,
                "height": config.clip_height
            }
            # seg_area = seg_list[seg_index]["area"]
            # if seg_area <= config.min_greenhouse_area:
            #     continue
            # else:   
            seg_bbox = seg_list[seg_index]["bbox"]
            seg_segm = seg_list[seg_index]["segmentation"]

            annotation["id"] = segmentation_id
            annotation["image_id"] = image_id
            # annotation["area"] = seg_area
            annotation["bbox"] = seg_bbox
            annotation["iscrowd"] = 0
            annotation["segmentation"] = seg_segm
            coco_output["annotations"].append(annotation)
            segmentation_id += 1

    coco_json_path = osp.join(config.output_dir,("CCGreenhouse_" + type_ + "_2020.json"))
    with open(coco_json_path, 'w') as output_json_file:
        json.dump(coco_output, output_json_file, indent=4)



if __name__ == '__main__':
    png_file_list = check_image_list(config.img_path, "*.png")
    # json_file_list = [i.replace(".png", ".json") for i in png_file_list]
    train_x, valid_x = train_test_split(png_file_list, test_size=config.val_ratio, random_state=0)  # 分训练集和验证集
    type_= ["train", "val"]
    data_split = [train_x, valid_x]

    for i in range(len(type_)):
        json2cocodataset(data_split[i],type_[i])