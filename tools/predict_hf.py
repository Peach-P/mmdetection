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

def hist(root_dir):
    pass
def tif_to_jpg(root_dir):
    for filename in os.listdir(root_dir):
        portion = os.path.splitext(filename)#分离文件名字和后缀
        if portion[1] ==".tif":#根据后缀来修改,如无后缀则空
            newname = portion[0]+".jpg"#要改的新后缀
            os.rename(os.path.join(root_dir,filename),os.path.join(root_dir,newname))
    
def myTest(root_dir, checkpoint_file_hanfeng, checkpoint_file_others,config_file_hangfeng, config_file_others, crop_flag=True):
    # 焊缝检测
    model_hanfeng = init_detector(config_file_hangfeng, checkpoint_file_hanfeng, device='cuda:0')
    model_others = init_detector(config_file_others, checkpoint_file_others, device='cuda:0')
    # images_path = glob.glob(os.path.join(root_dir + '/*.jpg'))  # 所有图片路径
    images_path = glob.glob(os.path.join(root_dir + '/*.tif'))  # 所有图片路径
    for image_path in tqdm.tqdm(images_path):
        doc = Document()
        annotation = doc.createElement("annotation")
        doc.appendChild(annotation)
        filename = doc.createElement("filename")
        annotation.appendChild(filename)
        filename.appendChild(doc.createTextNode(image_path.split('/')[-1]))
        imgCrop_all = []
        result_hanfeng = inference_detector(model_hanfeng, image_path)
        print('result_hf',result_hanfeng)
        # if result_hanfeng[0].size == 0: ## 需要根据个人的修改*************************
        if result_hanfeng[8].size == 0: ## 需要根据个人的修改*************************
            print("未检测到焊缝！")
        else:
            print("焊缝存在！")
            tmp_path = os.path.join(root_dir, "tmp")
            isExists=os.path.exists(tmp_path)
            # 判断结果
            if not isExists:
                # 如果不存在则创建目录
                 # 创建目录操作函数
                os.makedirs(tmp_path) 
                print(tmp_path+' 创建成功')
            
            # bbox_ = [int(i) for i in result_hanfeng[0][0][0:-1]]# 获得焊缝的4个坐标
            bbox_ = [int(i) for i in result_hanfeng[8][0][0:-1]]# 获得焊缝的4个坐标
            print('bbox_ coordinat', bbox_)
            if crop_flag:
              imgCrop_all.append(bbox_) 
            else:
                '''
                检测气孔，需要裁剪，一共裁剪为4份
                xmin,ymin,xmax,ymax
                '''
                imgCrop1 = [bbox_[0],bbox_[1],int((bbox_[0]+bbox_[2])/2),int((bbox_[1]+bbox_[3])/2)]
                imgCrop_all.append(imgCrop1)
                imgCrop2 = [int((bbox_[0]+bbox_[2])/2),bbox_[1],bbox_[2],int((bbox_[1]+bbox_[3])/2)]
                imgCrop_all.append(imgCrop2)
                imgCrop3 = [bbox_[0],int((bbox_[1]+bbox_[3])/2),int((bbox_[0]+bbox_[2])/2),bbox_[3]]
                imgCrop_all.append(imgCrop3)
                imgCrop4 = [int((bbox_[0]+bbox_[2])/2),int((bbox_[1]+bbox_[3])/2),bbox_[2],bbox_[3]]
                imgCrop_all.append(imgCrop4)
            print('img_cro_all',imgCrop_all)
            for i,img in enumerate(imgCrop_all):
                # image = cv2.imread(image_path, 2)
                image = cv2.imread(image_path, -1)
                new_img = image[img[1]:img[3], img[0]:img[2]]
                new_img = Image.fromarray(np.uint16(new_img))
                #save_path = os.path.join(root_dir, "tmp/"+ os.path.basename(image_path).split('.')[0] + '_'+str(i)+'.tif')
                save_path = os.path.join(root_dir, "tmp/"+str(i)+'.tif')
                new_img.save(save_path)
                # tif_to_jpg(os.path.join(root_dir, "tmp/"))
            # 其他缺陷检测
            # others_path = glob.glob(os.path.join(root_dir+'/tmp' + '/*.jpg')) # 所有图片路径
            # others_path.sort()
            # for index,other_path in enumerate(others_path):
            #     others_result = inference_detector(model_others, other_path)
            #     for j,category in enumerate(others_result): # 遍历每一类
            #         if j == 0:
            #             object_type = '气孔'
            #         elif j == 1:
            #             object_type = '未熔合'
            #         elif j == 2:
            #             object_type = '未焊透'
            #         for res in category:
            #             [xmin,ymin,xmax,ymax,p] = res
            #             [xmin,ymin,xmax,ymax] = [xmin+imgCrop_all[index][0],ymin+imgCrop_all[index][1],xmax+imgCrop_all[index][0],ymax+imgCrop_all[index][1]]
            #             object = doc.createElement("object")
            #             annotation.appendChild(object)
            #             name = doc.createElement("name")
            #             object.appendChild(name)
            #             name.appendChild(doc.createTextNode(object_type))
            #             bndbox = doc.createElement("bndbox")
            #             object.appendChild(bndbox)
            #             xmin_ = doc.createElement("xmin")
            #             bndbox.appendChild(xmin_)
            #             xmin_.appendChild(doc.createTextNode(str(int(xmin))))
            #             ymin_ = doc.createElement("ymin")
            #             bndbox.appendChild(ymin_)
            #             ymin_.appendChild(doc.createTextNode(str(int(ymin))))
            #             xmax_ = doc.createElement("xmax")
            #             bndbox.appendChild(xmax_)
            #             xmax_.appendChild(doc.createTextNode(str(int(xmax))))
            #             ymax_ = doc.createElement("ymax")
            #             bndbox.appendChild(ymax_)
            #             ymax_.appendChild(doc.createTextNode(str(int(ymax))))
                    
        filename_save = image_path.split('.')[0]+".xml"
        f = open(filename_save, "w+",encoding="utf8")
        f.write(doc.toprettyxml(indent="  "))
        f.close()
        

#标注文件
def xml_jpg2labelled(root_dir):
    xmls_list = []
    imgs_list = []
    res_path = os.path.join(root_dir,"res")
    all_list = os.listdir(root_dir)
    for list_name in all_list:
        if list_name[-3:] == "xml":
            xmls_list.append(list_name)
        elif list_name[-3:] == "jpg":
            imgs_list.append(list_name)
    nums = len(imgs_list)
    xmls_list.sort()
    imgs_list.sort()

    for i in tqdm.tqdm(range(nums)):
        img_path = os.path.join(root_dir, imgs_list[i])
        xml_path = os.path.join(root_dir, xmls_list[i])
        img = cv2.imread(img_path)
        labelled = img
        root = ET.parse(xml_path).getroot()
        objects = root.findall('object')
        for obj in objects:
            bbox = obj.find('bndbox')
            xmin = int(float(bbox.find('xmin').text.strip()))
            ymin = int(float(bbox.find('ymin').text.strip()))
            xmax = int(float(bbox.find('xmax').text.strip()))
            ymax = int(float(bbox.find('ymax').text.strip()))
            labelled = cv2.rectangle(labelled, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        #labelled.save('%s/%s.jpg' % (res_path, imgs_list[i][:-4]))
        cv2.imwrite('%s/%s.jpg' % (res_path, imgs_list[i][:-4]), labelled)


if __name__ == '__main__':
    root_dir = "/xiaopeng/backup/object_detection/cascade_13class" #存放要检测的图片
    crop_flag = True
    res_path = os.path.join(root_dir,"res")
    isExists=os.path.exists(res_path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录
        # 创建目录操作函数
        os.makedirs(res_path) 
        print(res_path+' 创建成功')
	#配置文件和日志文件
    checkpoint_file_hanfeng = "/xiaopeng/backup/object_detection/cascade_13class/epoch_140.pth"
    checkpoint_file_others = "/xiaopeng/backup/object_detection/cascade_13class/epoch_140.pth"
    config_file_hangfeng = '/xiaopeng/backup/object_detection/cascade_13class/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
    config_file_others = '/xiaopeng/backup/object_detection/cascade_13class/cascade_rcnn_hrnetv2p_w32_20e_coco.py'
    myTest(root_dir, checkpoint_file_hanfeng, checkpoint_file_others, config_file_hangfeng, config_file_others, crop_flag)
    xml_jpg2labelled(root_dir)
