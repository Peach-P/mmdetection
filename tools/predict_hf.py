from mmdet.apis import init_detector, inference_detector
import mmcv
import cv2
import tqdm
from PIL import Image
import glob
import os
import numpy as np
from xml.dom.minidom import Document
import xml.etree.ElementTree as ET

def Crop_hf(root_dir, checkpoint_file_hanfeng, config_file_hangfeng, crop_flag=True):
    # 焊缝检测
    model_hanfeng = init_detector(config_file_hangfeng, checkpoint_file_hanfeng, device='cuda:0')
    images_path = glob.glob(os.path.join(root_dir + '/*.tif'))  # 所有图片路径
    for image_path in tqdm.tqdm(images_path):
        imgCrop_all = []
        result_hanfeng = inference_detector(model_hanfeng, image_path)
        print('result_hf',result_hanfeng)
        if result_hanfeng[0].size == 0: ## 需要根据个人的修改*************************
            print("未检测到焊缝！")
        else:
            print("焊缝存在！")
            tmp_path = os.path.join(root_dir, "tmp")
            isExists=os.path.exists(tmp_path)
            if not isExists:
                os.makedirs(tmp_path) 
                print(tmp_path+' 创建成功')
            print('hf_geshu',result_hanfeng[0].shape[0])
            for j in range(result_hanfeng[0].shape[0]):
              print('j_wei', j)
            for j in range(result_hanfeng[0].shape[0]):
              bbox_ = [int(i) for i in result_hanfeng[0][j][0:-1]]# 获得焊缝的4个坐标
              print('bbox_ coordinat', bbox_)
              if crop_flag:
                imgCrop_all.append(bbox_) 
              print('img_cro_all',imgCrop_all)
              for i,img in enumerate(imgCrop_all):
                  image = cv2.imread(image_path, -1)
                  new_img = image[img[1]:img[3], img[0]:img[2]]
                  new_img = Image.fromarray(np.uint16(new_img))
                  name = image_path.split('\\')[-1][:-4]
                  save_path = os.path.join(root_dir, "tmp/"+name+'_'+str(i)+'.tif')
                  new_img.save(save_path)


if __name__ == '__main__':
    root_dir = r"D:\PHD_research\Model_train\object_detection\mmdetection_crop_hf\mmdetection-2.7.0\test_crop_hf" #存放要检测的图片
    crop_flag = True
    res_path = os.path.join(root_dir,"res")
    isExists=os.path.exists(res_path)
    if not isExists:
        os.makedirs(res_path) 
        print(res_path+' 创建成功')
    checkpoint_file_hanfeng = "D:/PHD_research/Model_train/object_detection/mmdetection_crop_hf/mmdetection-2.7.0/hf_cfg_dg/epoch_12.pth"
    config_file_hangfeng = 'D:/PHD_research/Model_train/object_detection/mmdetection_crop_hf/mmdetection-2.7.0/hf_cfg_dg/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
    Crop_hf(root_dir, checkpoint_file_hanfeng, config_file_hangfeng, crop_flag)
