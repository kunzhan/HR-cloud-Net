import argparse
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml
# from dataset.data import SemiDataset,my_SemiDataset, my_SemiDataset_HR
from util.utils import AverageMeter, intersectionAndUnion
import numpy as np
import os
from PIL import Image
from util.saliency_metric import cal_mae,cal_fm,cal_fm_,cal_sm,cal_em,cal_wfm
import torchvision.transforms as transforms
def evaluate_add(model, loader, mode, cfg):
    return_dict = {}
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    
    with torch.no_grad():
        mae,fm,sm,em,wfm= cal_mae(),cal_fm_(39),cal_sm(),cal_em(),cal_wfm()
        for img, mask, ids in loader:
            img = img.cuda()
            b, _, h, w = img.shape
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        res = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        pred = res['out']
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                res = model(img)
                pred = res['out'].argmax(dim=1)
                ped = np.array(pred.cpu()).squeeze(axis=0)  #ARRAY
                gt = np.asarray(mask, np.float32).squeeze(axis=0)   #ARRAY
            mae.update(ped, gt)
            sm.update(ped,gt)
            fm.cal(ped, gt)
            # em.update(res,mask)
            wfm.update(ped,gt)
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255) 
                # intersectionAndUnion(pred, mask, cfg['nclass'], 255)
            intersection_meter.update(torch.tensor(intersection))
            union_meter.update(torch.tensor(union))
    MAE = mae.show()
    _,meanf,precision,recall = fm.show()
    sm = sm.show()
    # em = em.show()
    wfm = wfm.show()
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(np.array(iou_class)) * 100.0
    return_dict['iou_class'] = iou_class
    return_dict['mIOU'] = mIOU
    return_dict['MAE'] = MAE
    return_dict['meanf'] = meanf
    return_dict['wfm'] = wfm
    return_dict['sm'] = sm
    return_dict['precision'] = precision
    return_dict['recall'] = recall

    return return_dict
def evaluate(model, loader, mode, cfg):
    return_dict = {}
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    
    with torch.no_grad():
        mae,fm,sm,em,wfm= cal_mae(),cal_fm(10080),cal_sm(),cal_em(),cal_wfm()
        for img, mask, ids in loader:
            img = img.cuda()
            b, _, h, w = img.shape
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                final = torch.zeros(b, 19, h, w).cuda()
                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        res = model(img[:, :, row: min(h, row + grid), col: min(w, col + grid)])
                        pred = res['out']
                        final[:, :, row: min(h, row + grid), col: min(w, col + grid)] += pred.softmax(dim=1)
                        col += int(grid * 2 / 3)
                    row += int(grid * 2 / 3)

                pred = final.argmax(dim=1)

            else:
                if mode == 'center_crop':
                    h, w = img.shape[-2:]
                    start_h, start_w = (h - cfg['crop_size']) // 2, (w - cfg['crop_size']) // 2
                    img = img[:, :, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]
                    mask = mask[:, start_h:start_h + cfg['crop_size'], start_w:start_w + cfg['crop_size']]

                res = model(img)
                pred = res['out'].argmax(dim=1)
                ped = np.array(pred.cpu()).squeeze(axis=0)  #ARRAY
                gt = np.asarray(mask, np.float32).squeeze(axis=0)   #ARRAY
            mae.update(ped, gt)
            sm.update(ped,gt)
            fm.update(ped, gt)
            # em.update(res,mask)
            wfm.update(ped,gt)
            intersection, union, target = \
                intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255) 
                # intersectionAndUnion(pred, mask, cfg['nclass'], 255)
            intersection_meter.update(torch.tensor(intersection))
            union_meter.update(torch.tensor(union))
    MAE = mae.show()
    _,meanf,_,_ = fm.show()
    sm = sm.show()
    # em = em.show()
    wfm = wfm.show()
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIOU = np.mean(np.array(iou_class)) * 100.0
    return_dict['iou_class'] = iou_class
    return_dict['mIOU'] = mIOU
    return_dict['MAE'] = MAE
    return_dict['meanf'] = meanf
    return_dict['wfm'] = wfm
    return_dict['sm'] = sm

    return return_dict

def main():
    parser = argparse.ArgumentParser(description='Semi-Supervised Semantic Segmentation')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    model = DeepLabV3Plus(cfg)
    model.load_state_dict(torch.load(args.checkpoint_path))
    model.eval()
    model.cuda()

    valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4)

    model.eval()
    res_val = evaluate(model, valloader, 'original', cfg)
    mIOU = res_val['mIOU']
    iou_class = res_val['iou_class']
    print(mIOU)
    print(iou_class)

if __name__ == '__main__':
    main()
