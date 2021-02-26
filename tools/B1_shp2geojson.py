# encoding: utf-8
"""
@version: 3.6
@author: mas
@file: 1_shp2json.py
@time: 2020/5/30 10:30
"""
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--shp_file_path', type=str,help='矢量文件路径', default=r"F:\2021\3——算法研究\车辆检测\1—阿布扎比车辆\样本\merge.shp")

parser.add_argument('--class_field', type=str,help='表征类别的字段名', default="category")

parser.add_argument('--json_file_folder', type=str,help='json文件输出路径', default=r"F:\2021\3——算法研究\车辆检测\1—阿布扎比车辆\样本")

config = parser.parse_args()

def shp2geojson(shp,outDir):
    '''

    :param dataDic: tif shp 字典
    :param outDir: 输出文件存放文件夹
    :return:
    '''
    reader = shapefile.Reader(shp)
    fields = reader.fields[1:]
    field_names = [field[0] for field in fields]
    buffer = []

    # 获取类别字段的索引
    class_field_index = field_names.index(config.class_field)
    if class_field_index < 0:
        print("shapefile中找不到制定的类别字段！")
        exit(1)

    index = 0
    for sr in reader.shapeRecords():
        # 属性字段赋值，包括id字段和类别字段
        atr = {field_names[0]: index, field_names[class_field_index]: sr.record[class_field_index]}
        geom = sr.shape.__geo_interface__
        buffer.append(dict(type="Feature", geometry=geom, properties=atr))
        index += 1
    output_dir = osp.join(outDir,"geojson")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    json_path = os.path.join(outDir, 'merge.json')
    geojson = open(json_path, "w", encoding='utf-8')
    geojson.write(dumps({"type": "变色立木",
                         "shp_path": shp,
                         "features_num": len(buffer),
                         "features": buffer}, indent=4) + '\n')
    geojson.close()
    print(f"geojson文件 {json_path} 已输出~~")

if __name__ == '__main__':
    shp2geojson(config.shp_file_path, config.json_file_folder)