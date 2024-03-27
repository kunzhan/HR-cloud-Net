import numpy as np
import os
from PIL import Image
from saliency_metric import cal_mae,cal_fm,cal_sm,cal_em,cal_wfm
import torchvision.transforms as transforms
import argparse

# dataset_path = '/idata/cloud/38_large_gt/'
# dataset_path_pre = './testdata/38imgHRcloud_86.528_large/'
# dataset = ['38_LandSat_8']
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
        #image = self.rgb_loader(self.images[self.index])
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
            '''
            angle = 12.85
            img_rotated = img.rotate(angle, resample=Image.BICUBIC, center=(img.width / 2, img.height / 2))
            angle_crop = angle % 180
            if angle > 90:
                angle_crop = 180 - angle_crop

            theta = angle_crop * np.pi / 180
            hw_ratio = float(img.height) / float(img.width)
            tan_theta = np.tan(theta)
            numerator = np.cos(theta) + np.sin(theta) * np.tan(theta)
            r = hw_ratio if img.height > img.width else 1 / hw_ratio
            denominator = r * tan_theta + 1
            crop_mult = numerator / denominator

            w_crop = int(crop_mult * img.width) - 450
            h_crop = int(crop_mult * img.height) - 450

            x0 = int((img.width - w_crop) / 2)
            y0 = int((img.height - h_crop) / 2)
            img_rotated = img_rotated.crop((x0, y0, x0 + w_crop, y0 + h_crop))    
            '''        
            return img.convert('L')


def main():
    parser = argparse.ArgumentParser(description='High_Resolution_cloud_net')
    parser.add_argument('--DS', default="38", type=str)
    parser.add_argument('--model', default="528", type=str)
    args = parser.parse_args()
    DS = args.DS
    p_dataset = DS + '_LandSat_8'
    mdl = 'HRcloud_86.' + args.model
    dataset_path = '/idata/cloud/' + DS + '_large_gt/'
    if DS == 'CH':
        dataset = DS + '_Test_img'
    else:
        dataset = DS + 'img'
    dataset_path_pre = './testdata/' + dataset + mdl + '_large/'
# for dataset in test_datasets:
    sal_root = dataset_path_pre
    gt_root = dataset_path 
    test_loader = test_dataset(sal_root, gt_root)
    mae,fm,sm,em,wfm= cal_mae(),cal_fm(test_loader.size),cal_sm(),cal_em(),cal_wfm()
    for i in range(test_loader.size):
        print ('predicting for %d / %d' % ( i + 1, test_loader.size))
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
    maxf,meanf,_,_ = fm.show()
    sm = sm.show()
    em = em.show()
    wfm = wfm.show()
    print('dataset: {} MAE: {:.4f} maxF: {:.4f} avgF: {:.4f} wfm: {:.4f} Sm: {:.4f} Em: {:.4f}'.format(p_dataset, MAE, maxf,meanf,wfm,sm,em))


if __name__ == '__main__':
    
    main()