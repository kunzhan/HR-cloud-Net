import os
import time
import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import sys
from model import HRcloudNet
import cv2
import re
from util.saliency_metric import cal_mae, cal_fm, cal_sm, cal_em, cal_wfm

def main():
    parser = argparse.ArgumentParser(description='High_Resolution_cloud_net')
    parser.add_argument('--d', default="38", type=str)
    parser.add_argument('--m', default="best", type=str)
    parser.add_argument('--gpu', default=0, type=int)
    args = parser.parse_args()
    DS = args.d
    mdl = 'HRcloud_' + args.m
    os.makedirs('testdata', exist_ok=True)
    predict_multi(DS, mdl, args.gpu)
    joint(DS, mdl)
    test_metric_score(DS,mdl)

def subtest(DS, gpu):
    mdl = "HRcloud_" + "best"
    os.makedirs('testdata', exist_ok=True)
    predict_multi(DS, mdl, gpu)
    joint(DS, mdl)
    test_metric_score(DS, mdl)

class test_dataset:
    def __init__(self, image_root, gt_root):
        self.img_list = [os.path.splitext(f)[0] for f in os.listdir(gt_root) if f.endswith('.png')]
        self.image_root = image_root
        self.gt_root = gt_root
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.img_list)
        self.index = 0

    def load_data(self):
        image = self.binary_loader(os.path.join(self.image_root,self.img_list[self.index]+ '.png'))
        gt = self.binary_loader(os.path.join(self.gt_root,self.img_list[self.index] + '.png'))

        self.index += 1
        return image, gt

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)      
            return img.convert('L')

def test_metric_score(DS, mdl):
    p_dataset = DS + '_LandSat_8'
    dataset_path = '/idata/cloud/' + DS + '_large_gt/'
    if DS == 'CH':
        dataset = DS + '_Test_img'
    else:
        dataset = DS + 'img'
    dataset_path_pre = './testdata/' + dataset + mdl + '_large/'
    sal_root = dataset_path_pre
    gt_root = dataset_path 
    test_loader = test_dataset(sal_root, gt_root)
    mae, fm, sm, em, wfm= cal_mae(), cal_fm(test_loader.size), cal_sm(), cal_em(), cal_wfm()
    for i in range(test_loader.size):
        sal, gt = test_loader.load_data()        
        
        if sal.size != gt.size:
            x, y = gt.size
            sal = sal.resize((x, y))
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-20)
        gt[gt > 0.5] = 1
        gt[gt != 1] = 0
        res = sal
        res = np.array(res)
        
  
        if res.max() == res.min():
            res = res/255
        else:
            res = (res - res.min()) / (res.max() - res.min())
        
        mae.update(res, gt)
        sm.update(res,gt)
        fm.update(res, gt)
        em.update(res,gt)
        wfm.update(res,gt)

    MAE = mae.show()
    maxf, meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(p_dataset, MAE, maxf,meanf,wfm,sm,em))

def joint(DS, mdl):
    if DS == 'CH':
        dataset = DS + '_Test_img'
    else:
        dataset = DS + 'img'

    pic_path = './testdata/' + dataset + mdl +'/'
    pic_target = './testdata/' + dataset + mdl +'_large/'
    if not os.path.exists(pic_target):
        os.mkdir(pic_target)


    large_picture_names = os.listdir(pic_path)
    if DS =='spars':
        large_picture_names = [large_picture_names[15:] for large_picture_names in large_picture_names] 
    else:
        large_picture_names = [large_picture_names[-44:] for large_picture_names in large_picture_names] 

    large_picture_names = set(large_picture_names) 
    # print(large_picture_names)
    picture_names = os.listdir(pic_path)                 
     # 随便一张图的的地址，用来查询小图长和宽
    width, length, depth = 352, 352, 3
    if len(picture_names)==0:
        print("none")
    else:
        num = 0
        for pic in large_picture_names:
            # 获取文件夹中的所有文件名  
            filenames = os.listdir(pic_path)  
            # print(pic)
            # 遍历文件名列表，并查找包含目标序列的文件名
            if DS == 'spars':
                num_width = 3
                num_length= 3
            else:            
                matching_filenames = [filename for filename in filenames if pic in filename]
                a = 0 
                b = 0
                for filename in matching_filenames:
                    w = int(filename.split("_")[2])
                    l = int(filename.split("_")[4])
                    if w > a:
                        a = w 
                    if l > b:
                        b = l  
                num_width = a
                num_length= b
            num += 1
            
            splicing_pic = np.zeros((num_width*width, num_length*length, depth))

            for idx in range(0, 1):
                k = 0
                splicing_pic = np.zeros((num_width*width, num_length*length, depth))
                for i in range(1, num_width+1):
                    for j in range(1, num_length+1):
                            k += 1
                            img_part = cv2.imread(pic_path + 'patch_{}_{}_by_{}_'.format(k, i, j)+pic,1)         
                            splicing_pic[ width*(i-1) : width*i, length*(j-1) : length*j, :] = img_part
                cv2.imwrite(pic_target + pic, splicing_pic)
        print(num)
        print("done!!!")



def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def predict_multi(DS, mdl, gpu):
    if DS == 'CH':
        dataset = DS + '_Test_img'
    else:
        dataset = DS + 'img'
    test_data = '/data/cloud/' + dataset# 
    to_test = {'test':test_data}
    classes = 1  # exclude background
    weights_path = './result/gpu_' + str(gpu) +'/' + mdl + ".pth"
    ckpt_path = './testdata/' + dataset + mdl
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)

    assert os.path.exists(weights_path), f"weights {weights_path} not found."

    mean = (0.342, 0.413, 0.359)    # Trian
    std = (0.085, 0.094, 0.091)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    model = HRcloudNet()

    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.to(device)
    model.eval()
 
    with torch.no_grad():
        for name, root in to_test.items():
            # 获取图片名称list
            img_list = [os.path.splitext(f)[0] for f in os.listdir(root) if f.endswith('.png')]
            # 开始计时
            t_start = time_synchronized()
            # 图像处理
            data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
            for idx, img_name in enumerate(img_list):
                # print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                original_img = Image.open(os.path.join(root, img_name +'.png')).convert('RGB')
                img = data_transform(original_img)
                
                img = torch.unsqueeze(img, dim=0)   
                img_height, img_width = img.shape[-2:] 
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)

                output = model(img.to(device))
                
                prediction = output['out'].argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                # 将前景对应的像素值改成255(白色)
                prediction[prediction == 1] = 255
                # 将不敢兴趣的区域像素设置成0(黑色)
                # prediction[roi_img == 0] = 0
                mask = Image.fromarray(prediction)
                # mask.save(os.path.join(ckpt_path, exp_name, img_name + '.png'))
                mask.save(os.path.join(ckpt_path, img_name + '.png'))
            # 预测用时
            t_end = time_synchronized()
            # print("inference time: {}".format(t_end - t_start))

if __name__ == '__main__':    
    main()