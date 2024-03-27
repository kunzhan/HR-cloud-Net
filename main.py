import argparse
import logging
import os
import pprint
from tqdm import tqdm
import numpy as np
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.optim import SGD
from torch.utils.data import DataLoader
import yaml
# import sys
import datetime
from dataset.data import cn_landsat
from model import HRcloudNet
# from baseline.CDnetv1 import CDnetV1
# from baseline.CDnetv2 import CDnetV2
# from baseline.hrnet import HRNet
# from baseline.pspnet import PSPNet
# from baseline.SegNet import SegNet
# from baseline.unet import UNet
from util.evaluate import evaluate_add
from util.utils import count_params, init_log
from util.dist_helper import setup_distributed
import random


parser = argparse.ArgumentParser(description='High_Resolution_cloud_net')
parser.add_argument('--gpu', default='2', type=int, help='id(s) for CUDA_VISIBLE_DEVICES')
parser.add_argument('--config', default="./configs/LandSat.yaml", type=str)
parser.add_argument('--save-path', default="./result/gpu_", type=str)
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)

def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.enabled = True
    if cuda_deterministic:
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:
        cudnn.deterministic = False
        cudnn.benchmark = True


def main():
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)
    model_name = 'HRcloud'

    results_file = args.save_path + str(args.gpu) + "/results_{}.txt".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0


    rank = 0
    
    if rank == 0:
        logger.info('{}\n'.format(pprint.pformat(cfg)))

    if rank == 0:
        os.makedirs(args.save_path + str(args.gpu), exist_ok=True)
    init_seeds(0, False)

    # model = CDnetV1(num_classes = 2)
    # model = CDnetV2(num_classes = 2)
    # model = HRNet(num_classes = 2)
    # model = PSPNet()
    # model = SegNet(n_classes = 2)
    # model = UNet(in_channels = 3)

    model = HRcloudNet(num_classes=2)

    params_to_optimize = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(  
        params_to_optimize, 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.0005
    )

    model.cuda()
    ce_sup = nn.CrossEntropyLoss().cuda()
    x_sup = nn.CrossEntropyLoss(reduction='none').cuda()

    China_landaset = cn_landsat(cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'])
    
    valset = cn_landsat(cfg['dataset'], cfg['data_root'], 'val')

    cn_data = DataLoader(China_landaset, batch_size=cfg['batch_size'],
                               pin_memory=False, num_workers=0, drop_last=True, sampler=None)

    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4,
                           drop_last=True, sampler=None)

    total_iters = len(cn_data) * cfg['epochs']
    previous_best = 0.0

    for epoch in range(cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.6f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss, total_loss_x_sup, total_loss_s, total_loss_w_fp = 0.0, 0.0, 0.0, 0.0
        total_loss_kl = 0.0
        total_mask_ratio = 0.0

        if rank == 0:
            tbar = tqdm(total=len(cn_data))
        for i, (img_w, img_s, mask) in enumerate(cn_data):
            mask = mask.cuda()
            img_w = img_w.cuda()
            img_s = img_s.cuda()
            
            model.train()
            num_lb, num_ulb = img_s.shape[0], img_w.shape[0]
            res_w = model(torch.cat((img_s, img_w)), need_fp=False, use_corr=False)
            preds = res_w['out']
            pred_s, pred_w = preds.split([num_lb, num_ulb])
            pred_w_ = pred_w.detach()
            conf_w = pred_w_.detach().softmax(dim=1).max(dim=1)[0]
            conf_fliter_w = conf_w >= 0.8

            loss_x_sup = ce_sup(pred_w, mask) 
            loss_x_w2s = torch.sum(x_sup(pred_s, mask)*conf_fliter_w)/8/352/352
            
            loss = 0.75 * loss_x_sup + loss_x_w2s * 0.5
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_loss_x_sup += loss_x_sup.item()
            total_loss_s += loss_x_w2s.item()
            total_loss_kl += loss_x_w2s.item()

            iters = epoch * len(cn_data) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

        if rank == 0:
            tbar.set_description(' Total: {:.3f}, x: {:.3f} '
                                 's: {:.3f}, w_fp: {:.3f}, w_kl: {:.3f}'.format(
                total_loss / (i + 1), total_loss_x_sup / (i + 1), total_loss_s / (i + 1),
                total_loss_w_fp / (i + 1),total_loss_kl / (i + 1)))
            tbar.update(1)

        if rank == 0:
            tbar.close()

        if cfg['dataset'] == 'cityscapes':
            eval_mode = 'center_crop' if epoch < cfg['epochs'] - 20 else 'sliding_window'
        else:
            eval_mode = 'original'

        res_val = evaluate_add(model, valloader, eval_mode, cfg)

        mIOU = res_val['mIOU']
        class_IOU = res_val['iou_class']
        MAE = res_val['MAE']
        meanf = res_val['meanf']
        wfm = res_val['wfm']
        sm = res_val['sm']
        pre = res_val['precision']
        rec = res_val['recall']
        if rank == 0:
            logger.info('***** Evaluation {} ***** >>>> meanIOU: {:.6f} \n'.format(eval_mode, mIOU))
            logger.info('***** ClassIOU ***** >>>> \n{}\n'.format(class_IOU))
        # torch.distributed.barrier()
        with open(results_file, "a") as f:
            train_info = f"[epoch: {epoch}]\n" \
                         f"train_loss: {total_loss / (i + 1):.4f}\n" \
                         f"lr: {lr:.6f}\n" \
                         f"val_mIOU:{mIOU} \n"\
                         f"val_class_IOU:{class_IOU}\n"\
                         f"MAE:{MAE}\n"\
                         f"AvgFm:{meanf}\n"\
                         f"presicion:{pre}\n"\
                         f"recall:{rec}\n"\
                         f"Wfm:{wfm}\n"\
                         f"Sm:{sm}\n"
            # f.write(train_info + val_info + "\n\n")
            f.write(train_info+ "\n\n")
        

        if mIOU > previous_best and rank == 0:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path + str(args.gpu), '%s_%s.pth' % (model_name, "best")))
            previous_best = mIOU
            # torch.save(model.module.state_dict(), os.path.join(args.save_path, '%s_%.3f.pth' % (cfg['backbone'], mIOU)))
            torch.save(model.state_dict(), os.path.join(args.save_path + str(args.gpu), '%s_%s.pth' % (model_name, "best")))
        # torch.distributed.barrier()
    from landsat_test import subtest
    subtest('38', str(args.gpu))
    subtest('spars', str(args.gpu))
    subtest('CH', str(args.gpu))


if __name__ == '__main__':
    main()
