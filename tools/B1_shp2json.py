# encoding: utf-8
"""
@version: 3.6
@author: mas
@file: 1_shp2json.py
@time: 2020/5/30 10:30
"""
from utils import *

parser = argparse.ArgumentParser()

parser.add_argument('--file_path', type=str,help='影像及图斑矢量文件路径', default="/home/sjss/Mession/HZDP")
parser.add_argument('--file_path', type=str,help='影像及图斑矢量文件路径', default="/home/sjss/Mession/HZDP")
parser.add_argument('--json_file_folder', type=str,help='json文件输出路径', default="/home/sjss/Mession/HZDP/DP_data")

config = parser.parse_args()

def shp2geojson(dataDic,outDir):
    '''

    :param dataDic: tif shp 字典
    :param outDir: 输出文件存放文件夹
    :return:
    '''
    for tif,shp in dataDic.items():
        if not osp.exists(shp):
            print(f"{shp} 文件不存在 对应影像为：{tif} 跳过处理~~")
            continue
        reader = shapefile.Reader(shp)
        fields = reader.fields[1:]
        field_names = [field[0] for field in fields]
        buffer = []
        index = 0
        for sr in reader.shapeRecords():
            atr = {field_names[0]: index}
            geom = sr.shape.__geo_interface__
            buffer.append(dict(type="Feature", geometry=geom, properties=atr))
            index += 1
        output_dir = osp.join(outDir,"geojson")
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        json_path = osp.join(output_dir,osp.basename(PurePosixPath(tif).stem)+".json")
        geojson = open(json_path, "w", encoding='utf-8')
        geojson.write(dumps({"type": "FeatureCollection",
                             "tif_path": tif,
                             "shp_path": shp,
                             "features_num": len(buffer),
                             "features": buffer}, indent=4) + '\n')
        geojson.close()
        print(f"geojson文件 {json_path} 已输出~~")

if __name__ == '__main__':
    shp2geojson(data_dic, config.json_file_folder)