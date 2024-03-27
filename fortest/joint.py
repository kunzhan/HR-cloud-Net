import cv2
import numpy as np
import os
import re
import argparse

def main():
    parser = argparse.ArgumentParser(description='High_Resolution_cloud_net')
    parser.add_argument('--DS', default="38", type=str)
    parser.add_argument('--model', default="528", type=str)
    args = parser.parse_args()
    DS = args.DS
    mdl = 'HRcloud_86.' + args.model
    width, length, depth = 352, 352, 3
    if DS == 'CH':
        dataset = DS + '_Test_img'
        # img_1_1 = cv2.imread('/data/cloud/CH_Test_mask/patch_9_1_by_9_LC08_L1TP_148035_20210109_20210307_01_T1.png')
    else:
        dataset = DS + 'img'
        # if DS =='38':
            # img_1_1 = cv2.imread('/data/cloud/38mask/patch_1_1_by_1_LC08_L1TP_003052_20160120_20170405_01_T1.png')
        # else:
            # img_1_1 = cv2.imread('/data/cloud/sparsmask/patch_1_1_by_1_LC80010812013365LGN00_18.png')

    pic_path = './testdata/'+dataset + mdl +'/'
    pic_target = './testdata/'+dataset + mdl +'_large/'
    # pic_path = './testdata/CH_Test_imgHRcloud_86.528/'
    # pic_target = './testdata/CH_Test_imgHRcloud_86.528_large/'
    if not os.path.exists(pic_target):
        os.mkdir(pic_target)


    large_picture_names = os.listdir(pic_path)
    if DS =='spars':
        large_picture_names = [large_picture_names[15:] for large_picture_names in large_picture_names] 
    else:
        large_picture_names = [large_picture_names[-44:] for large_picture_names in large_picture_names] 
    # large_picture_names.sort(key=lambda x: int(x.split("_")[8]))
    large_picture_names = set(large_picture_names) 
    # print(large_picture_names)
    picture_names = os.listdir(pic_path)                 
     # 随便一张图的的地址，用来查询小图长和宽
    # (width, length, depth) = img_1_1.shape
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

if __name__ == '__main__':
    
    main()