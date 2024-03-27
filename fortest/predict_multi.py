import os
import time
import argparse
import torch
from torchvision import transforms
import numpy as np
from PIL import Image

import sys
from model import HighResolutionNet
parser = argparse.ArgumentParser(description='High_Resolution_cloud_net')
parser.add_argument('--DS', default="38", type=str)
parser.add_argument('--model', default="528", type=str)

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()



def main():

    args = parser.parse_args()
    DS = args.DS
    mdl = 'HRcloud_86.' + args.model
    if DS == 'CH':
        dataset = DS + '_Test_img'
    else:
        dataset = DS + 'img'
    # dataset = 'CH_Test_img'
    # dataset = '38img'# 
    test_data = '/data/cloud/' + dataset# 
    to_test = {'test':test_data}
    classes = 1  # exclude background
    weights_path = "./result/0217/" + mdl +".pth"
    ckpt_path = './testdata/' + dataset + mdl
    if not os.path.exists(ckpt_path):
        os.mkdir(ckpt_path)
    

    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    # assert os.path.exists(img_path), f"image {img_path} not found."
    # assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."
    # [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    # mean = (0.709, 0.381, 0.224)  # Ori
    # std = (0.127, 0.079, 0.043)
    # mean = (0.485, 0.456, 0.406) # Cor
    # std = (0.229, 0.224, 0.225)
    mean = (0.342, 0.413, 0.359)    # Trian
    std = (0.085, 0.094, 0.091)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = UNet(in_channels=3, num_classes=2, base_c=32)
    model = HighResolutionNet()
    # num_classes = 2
    # model = CDnetV1_MODEL(Bottleneck,[3, 4, 6, 3], num_classes)
    # model = GateNet(Bottleneck,[3,4,6,3])
    # model = CDnetV2_MODEL(Bottleneck,[3, 4, 6, 3], num_classes)

    # load weights
    # model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'])
    model.load_state_dict(torch.load(weights_path, map_location='cpu'))
    model.to(device)
    model.eval() 

    # load roi mask
    # roi_img = Image.open(roi_mask_path).convert('L')
    # roi_img = np.array(roi_img)
    '''
    original_img = Image.open(img_path).convert('RGB')

    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    '''
    
     # load image 
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
                print ('predicting for %s: %d / %d' % (name, idx + 1, len(img_list)))
                original_img = Image.open(os.path.join(root, img_name +'.png')).convert('RGB')
                img = data_transform(original_img)
                
                img = torch.unsqueeze(img, dim=0)
                # init model？   
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
            print("inference time: {}".format(t_end - t_start))


if __name__ == '__main__':   
    main()