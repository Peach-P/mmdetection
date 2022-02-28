# coding=utf-8
import os
from mmdet.apis import init_detector
from mmdet.apis import inference_detector
from mmdet.apis import init_detector, inference_detector, show_result

def get_file(root_path, all_files=[]):
  files = os.listdir(root_path)
  for file in files:
    if not os.path.isdir(root_path + '/' + file):
      all_files.append(root_path + '/' + file)
    else:
      get_file((root_path + '/' + file), all_files)
  return all_files

# 模型配置文件
config_file = 'models/cascade_rcnn_hrnetv2p_w32_20e_coco.py'

# 预训练模型文件
checkpoint_file = 'models/hrnet_cascade_rcnn.pth'

# 通过模型配置文件与预训练文件构建模型
model = init_detector(config_file, checkpoint_file, device='cuda:0')


# 测试图片
path = '/content/drive/MyDrive/Cui/dg_dataset/VOC2007/JPEGImages'
img = get_file(path)
result = inference_detector(model, img)


# category_list = ['porosity','lack_of_fusion','lack_of_penetration','overlap','crack','undercut','hollow','faulty_formation','other']
# output=[] # 输出
# for single_img in result:
#   single_output=[]
#   for n,item in enumerate(single_img): # n是标签
#     for box in item:
#       score = box[4]
#       xmin = int(box[0])
#       xmax = int(box[2])
#       ymin = int(box[1])
#       ymax = int(box[3])
#       w = xmax-xmin
#       h = ymax-ymin
#       di = {'category':category_list[n],'score':score,'bbox':[xmin,ymin,w,h]}
#       single_output.append(di)
#   output.append(single_output)
# print(output)

show_result(img, result, model.CLASSES, out_file='testOut.jpg')