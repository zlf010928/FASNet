import os
import random
import logging
import argparse
import torch
import torch.nn as nn
import torch.utils.data as data
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from math import ceil
import numpy as np
from distutils.version import LooseVersion
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
import warnings
import sys

sys.path.append(os.path.abspath('.'))
from utils.eval import Eval
from utils.train_helper import get_model

from datasets.cityscapes_Dataset import City_Dataset, City_DataLoader, inv_preprocess, decode_labels





def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Unsupported value encountered.')

class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, device=0, ignore = -1):
        super().__init__()
        self.device = device
        self.ignore = ignore
    def forward(self, output, target):
        lis = []
        for i in range(19):
            # non_zero_num = torch.nonzero(target).shape[0]
            # print(type(non_zero_num))
            gt = (target == i).float()  # B
            inter = torch.sum(gt, dim=(0, 1, 2)).cpu().numpy()  # B


            total_num = torch.prod(torch.tensor(target.shape)).float()

            k = inter.item() / total_num.item()

            lis.append(1-k)
        # print(lis)

        scaled_weight = torch.tensor(lis).cuda(self.device)
        # scaled_weight = torch.tensor([]).cuda(self.device)

        # non_zero_num = torch.nonzero(target).shape[0]
        # total_num = torch.prod(torch.tensor(target.shape)).float()
        # k = non_zero_num / total_num
        # scaled_weight = torch.tensor([k, 1-k]).cuda(self.device)

        if type(output) == list:
            loss = F.cross_entropy(output[0], target, weight=scaled_weight, ignore_index= self.ignore)
            for i in range(1, len(output)):
                loss += F.cross_entropy(output[i], target, weight=scaled_weight, ignore_index= self.ignore)
        else:
            loss = F.cross_entropy(output, target, weight=scaled_weight, ignore_index= self.ignore)

        return loss



device_ids = [0]



class Trainer():
    def __init__(self, args, cuda=None, train_id="None", logger=None):
        self.args = args
        ITER_MAX = args.each_epoch_iters
        self.device = torch.device('cuda:{}'.format(device_ids[0]))
        # os.environ["CUDA_VISIBLE_DEVICES"] = self.args.gpu
        os.environ["CUDA_VISIBLE_DEVICES"] = "2"
        self.cuda = cuda and torch.cuda.is_available()
        # self.device = torch.device('cuda' if self.cuda else 'cpu')
        self.train_id = train_id
        self.logger = logger

        self.current_MIoU = 0
        self.best_MIou = 0
        self.best_source_MIou = 0
        self.current_epoch = 0
        self.current_iter = 0
        self.second_best_MIou = 0

        # set TensorboardX
        self.writer = SummaryWriter(self.args.checkpoint_dir)

        # Metric definition
        self.Eval = Eval(self.args.num_classes)

        # loss definition
        if args.weight_loss:
            self.loss = WeightedCrossEntropyLoss().to(self.device)
        else:
            self.loss = nn.CrossEntropyLoss(weight=None, ignore_index=-1)

        # self.loss.to(self.device)
        self.loss = self.loss.cuda(device_ids[0])

        # model
        self.model, params = get_model(self.args)
        
       


        # self.model.to(self.device)
        self.model = self.model.cuda()
        self.model = torch.nn.DataParallel(self.model, device_ids=device_ids)
        self.model=torch.load('80.pth')

        self.optimizer = torch.optim.SGD(
                params=params,
                momentum=self.args.momentum,
                weight_decay=self.args.weight_decay)

        if len(device_ids) > 1:
                self.optimizer = nn.DataParallel(self.optimizer, device_ids=device_ids)

      
        self.dataloader = City_DataLoader("train")
       
        self.dataloader.num_iterations = min(self.dataloader.num_iterations, ITER_MAX)
        print(self.args.iter_max, self.dataloader.num_iterations)
        self.epoch_num = ceil(self.args.iter_max / self.dataloader.num_iterations) if self.args.iter_stop is None else \
            ceil(self.args.iter_stop / self.dataloader.num_iterations)

    def main(self):
       

        

        self.current_epoch = 0
        # train
        self.train()




        self.writer.close()


    def train(self):
        # self.validate() # check image summary
        pixel_num = []
        for epoch in range(201):

            self.train_one_epoch(pixel_num = pixel_num)
            # validate
            if epoch%20==0:
              torch.save(self.model,
                       '{}.pth'.format(epoch))
            self.val()




            

            self.current_epoch += 1

    def val(self):
        from tools.metrics import get_all_metrics
        result=np.zeros(20)
        num=0
        loss=0
        for i,data in enumerate(self.dataloader.data_loader):
            
            num +=1
            x,y,_=data


              
            x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
            print(y.shape)
            y = torch.squeeze(y, 1)
            self.optimizer.zero_grad()
            print(y.shape)
            

            # model
            pred = self.model(x)            



          
            if isinstance(pred, tuple):

                pred_2 = pred[1]
                pred_lay2 = pred[2]
                pred_lay1 = pred[3]
                mid_lay2_ori = pred[4]
                mid_lay2_ined = pred[5]
                mid_lay1_ori = pred[6]
                mid_lay1_ined = pred[7]
                mid_lay2_iwed = pred[8]
                mid_lay1_iwed = pred[9]
                pred = pred[0]

            a_ = torch.ones(y.size()[0],y.size()[1],y.size()[2],dtype=torch.long)*args.num_classes
            a_ = a_.to(self.device)
            y_ = torch.where(y==-1, a_, y).to(self.device)
            gt_one_hot = self.get_one_hot(y_, args.num_classes+1).to(self.device)

            outs_lay2 = []
            for i in args.selected_classes:
                mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
                mask = F.interpolate(mask, size=mid_lay2_ori.size()[2:],mode='nearest')
                out = mid_lay2_ori * mask
                out = self.model.module.SAN_stage_2.IN(out)
                # out = nn.InstanceNorm2d(512, affine=True)(out)
                outs_lay2.append(out)
            mid_lay2_label = sum(outs_lay2)
            mid_lay2_label = self.model.module.SAN_stage_2.relu(mid_lay2_label)

            outs_lay1 = []
            for i in args.selected_classes:
                mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
                mask = F.interpolate(mask, size=mid_lay1_ori.size()[2:], mode='nearest')
                out = mid_lay1_ori * mask
                out = self.model.module.SAN_stage_1.IN(out)
                # out = nn.InstanceNorm2d(512, affine=True)(out)
                outs_lay1.append(out)
            mid_lay1_label = sum(outs_lay1)
            mid_lay1_label = self.model.module.SAN_stage_1.relu(mid_lay1_label)

            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            # loss
            loss_main = criterion(pred, y)
            loss_lay2 = 0.1 * criterion(pred_lay2, y)
            loss_lay1 = 0.1 * criterion(pred_lay1, y)
            loss_in_lay2 = 0.1 * F.smooth_l1_loss(mid_lay2_ined, mid_lay2_label)
            loss_in_lay1 = 0.1 * F.smooth_l1_loss(mid_lay1_ined, mid_lay1_label)
            loss_iw_lay2 = 0.1 * mid_lay2_iwed
            loss_iw_lay1 = 0.1 * mid_lay1_iwed
            cur_loss = loss_main + loss_lay2 + loss_lay1 + loss_in_lay2 + loss_in_lay1 + loss_iw_lay2 + loss_iw_lay1


           


            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
                cur_loss += loss_2
                

            

           

           

          
          
            
            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            temp=get_all_metrics(argpred,label)
            loss+=cur_loss.item()
            result[0]+=loss
            for i in range(len(temp)):
                result[i+1]+=temp[i]

          
            
            
            label = label[0].astype(np.uint8) * 255
            Image.fromarray(label).save('%d_label.png' % 2)
            preds = argpred
            predict = preds
            pred = predict[0].astype(np.uint8) * 255


            Image.fromarray(pred).save('%d_pred.png' % 1)
        for i in range(8):
            result[i]/=num
        num=0

        dateloader=City_DataLoader("val").data_loader
        loss=0
        for i,data in enumerate(dateloader):
            num +=1
            x,y,_=data


              
            x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
            y = torch.squeeze(y, 1)
            self.optimizer.zero_grad()
            print(y.shape)

            # model
            pred = self.model(x)            



          
            if isinstance(pred, tuple):

                pred_2 = pred[1]
                pred_lay2 = pred[2]
                pred_lay1 = pred[3]
                mid_lay2_ori = pred[4]
                mid_lay2_ined = pred[5]
                mid_lay1_ori = pred[6]
                mid_lay1_ined = pred[7]
                mid_lay2_iwed = pred[8]
                mid_lay1_iwed = pred[9]
                pred = pred[0]

            a_ = torch.ones(y.size()[0],y.size()[1],y.size()[2],dtype=torch.long)*args.num_classes
            a_ = a_.to(self.device)
            y_ = torch.where(y==-1, a_, y).to(self.device)
            gt_one_hot = self.get_one_hot(y_, args.num_classes+1).to(self.device)

            outs_lay2 = []
            for i in args.selected_classes:
                mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
                mask = F.interpolate(mask, size=mid_lay2_ori.size()[2:],mode='nearest')
                out = mid_lay2_ori * mask
                out = self.model.module.SAN_stage_2.IN(out)
                # out = nn.InstanceNorm2d(512, affine=True)(out)
                outs_lay2.append(out)
            mid_lay2_label = sum(outs_lay2)
            mid_lay2_label = self.model.module.SAN_stage_2.relu(mid_lay2_label)

            outs_lay1 = []
            for i in args.selected_classes:
                mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
                mask = F.interpolate(mask, size=mid_lay1_ori.size()[2:], mode='nearest')
                out = mid_lay1_ori * mask
                out = self.model.module.SAN_stage_1.IN(out)
                # out = nn.InstanceNorm2d(512, affine=True)(out)
                outs_lay1.append(out)
            mid_lay1_label = sum(outs_lay1)
            mid_lay1_label = self.model.module.SAN_stage_1.relu(mid_lay1_label)

            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            # loss
            loss_main = criterion(pred, y)
            loss_lay2 = 0.1 * criterion(pred_lay2, y)
            loss_lay1 = 0.1 * criterion(pred_lay1, y)
            loss_in_lay2 = 0.1 * F.smooth_l1_loss(mid_lay2_ined, mid_lay2_label)
            loss_in_lay1 = 0.1 * F.smooth_l1_loss(mid_lay1_ined, mid_lay1_label)
            loss_iw_lay2 = 0.1 * mid_lay2_iwed
            loss_iw_lay1 = 0.1 * mid_lay1_iwed
            cur_loss = loss_main + loss_lay2 + loss_lay1 + loss_in_lay2 + loss_in_lay1 + loss_iw_lay2 + loss_iw_lay1


           


            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
                cur_loss += loss_2
                

            

           

           

          
          
            
            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            temp=get_all_metrics(argpred,label)
            loss+=cur_loss.item()
            result[8]+=loss
            for i in range(len(temp)):
                result[i+9]+=temp[i]

          
            
            
            label = label[0].astype(np.uint8) * 255
            Image.fromarray(label).save('vv{}_label.png'.format(1))
            preds = argpred
            predict = preds
            pred = predict[0].astype(np.uint8) * 255


            Image.fromarray(pred).save('vv{}_pred.png'.format(2))
        for i in range(8):
            result[8+i]/=num
        
        import csv
        path = './result2.csv'
        with open(path, 'a+', newline='') as f:
            csv_write = csv.writer(f)
            csv_row = result
            csv_write.writerow(csv_row)

       

       

       



    def get_one_hot(self, label, N):
        size = list(label.size())
        label = label.view(-1)
        ones = torch.sparse.torch.eye(N).to(self.device)
        ones = ones.index_select(0, label)
        size.append(N)
        ones = ones.view(*size)
        ones = ones.transpose(2, 3)
        ones = ones.transpose(1, 2)
        return ones

    def train_one_epoch(self,pixel_num):
        tqdm_epoch = tqdm(self.dataloader.data_loader,
                          total=self.dataloader.num_iterations,
                          desc="Train Epoch-{}-total-{}".format(self.current_epoch + 1, self.epoch_num),file=sys.stdout)
        
        self.Eval.reset()

        train_loss = []
        loss_seg_value_2 = 0
        iter_num = self.dataloader.num_iterations

        self.model.train()
        # Initialize your average meters

        batch_idx = 0

        for x, y, _ in tqdm_epoch:
            self.poly_lr_scheduler(
                optimizer=self.optimizer,
                init_lr=self.args.lr,
                iter=self.current_iter,
                max_iter=self.args.iter_max,
                power=self.args.poly_power,
            )
            
            if self.cuda:
                x, y = x.to(self.device), y.to(device=self.device, dtype=torch.long)
            y = torch.squeeze(y, 1)
            self.optimizer.zero_grad()
            print(y.shape)

            # model
            pred = self.model(x)            



            if isinstance(pred, tuple):

                pred_2 = pred[1]
                pred_lay2 = pred[2]
                pred_lay1 = pred[3]
                mid_lay2_ori = pred[4]
                mid_lay2_ined = pred[5]
                mid_lay1_ori = pred[6]
                mid_lay1_ined = pred[7]
                mid_lay2_iwed = pred[8]
                mid_lay1_iwed = pred[9]
                pred = pred[0]

            a_ = torch.ones(y.size()[0],y.size()[1],y.size()[2],dtype=torch.long)*args.num_classes
            a_ = a_.to(self.device)
            y_ = torch.where(y==-1, a_, y).to(self.device)
            gt_one_hot = self.get_one_hot(y_, args.num_classes+1).to(self.device)

            outs_lay2 = []
            for i in args.selected_classes:
                mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
                mask = F.interpolate(mask, size=mid_lay2_ori.size()[2:],mode='nearest')
                out = mid_lay2_ori * mask
                out = self.model.module.SAN_stage_2.IN(out)
                # out = nn.InstanceNorm2d(512, affine=True)(out)
                outs_lay2.append(out)
            mid_lay2_label = sum(outs_lay2)
            mid_lay2_label = self.model.module.SAN_stage_2.relu(mid_lay2_label)

            outs_lay1 = []
            for i in args.selected_classes:
                mask = torch.unsqueeze(gt_one_hot[:, i, :, :], 1)
                mask = F.interpolate(mask, size=mid_lay1_ori.size()[2:], mode='nearest')
                out = mid_lay1_ori * mask
                out = self.model.module.SAN_stage_1.IN(out)
                # out = nn.InstanceNorm2d(512, affine=True)(out)
                outs_lay1.append(out)
            mid_lay1_label = sum(outs_lay1)
            mid_lay1_label = self.model.module.SAN_stage_1.relu(mid_lay1_label)

            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            # loss
            loss_main = criterion(pred, y)
            loss_lay2 = 0.1 * criterion(pred_lay2, y)
            loss_lay1 = 0.1 * criterion(pred_lay1, y)
            loss_in_lay2 = 0.1 * F.smooth_l1_loss(mid_lay2_ined, mid_lay2_label)
            loss_in_lay1 = 0.1 * F.smooth_l1_loss(mid_lay1_ined, mid_lay1_label)
            loss_iw_lay2 = 0.1 * mid_lay2_iwed
            loss_iw_lay1 = 0.1 * mid_lay1_iwed
            cur_loss = loss_main + loss_lay2 + loss_lay1 + loss_in_lay2 + loss_in_lay1 + loss_iw_lay2 + loss_iw_lay1


            #########################
            lis = []
            for i in range(19):
                # non_zero_num = torch.nonzero(target).shape[0]
                # print(type(non_zero_num))
                gt = (y == i).float()  # B
                inter = torch.sum(gt, dim=(0, 1, 2)).cpu().numpy()  # B

                total_num = torch.prod(torch.tensor(y.shape)).float()

                k = inter.item() / total_num.item()

                lis.append(k)
            pixel_num.append(lis)
            #########################


            if self.args.multi:
                loss_2 = self.args.lambda_seg * self.loss(pred_2, y)
                cur_loss += loss_2
                loss_seg_value_2 += loss_2.cpu().item() / iter_num

            tqdm_epoch.set_postfix(loss_total=cur_loss.item(), loss_main=loss_main.item())

            # optimizer
            cur_loss.backward()
            if len(device_ids) > 1:
                self.optimizer.module.step()
            else:
                self.optimizer.step()

            train_loss.append(cur_loss.item())

          
            batch_idx += 1

            self.current_iter += 1

            if np.isnan(float(cur_loss.item())):
                raise ValueError('Loss is nan during training...')

            pred = pred.data.cpu().numpy()
            label = y.cpu().numpy()
            argpred = np.argmax(pred, axis=1)
            label = label[0].astype(np.uint8) * 255
            Image.fromarray(label).save('11{}_label.png'.format(1))
            preds = argpred
            predict = preds
            pred = predict[0].astype(np.uint8) * 255


            Image.fromarray(pred).save('22{}_pred.png'.format(2))
            

            if batch_idx == self.dataloader.num_iterations:
                break
            

        #######

        tqdm_epoch.close()





    def poly_lr_scheduler(self, optimizer, init_lr=None, iter=None,
                          max_iter=None, power=None):
        init_lr = self.args.lr if init_lr is None else init_lr
        iter = self.current_iter if iter is None else iter
        max_iter = self.args.iter_max if max_iter is None else max_iter
        power = self.args.poly_power if power is None else power
        new_lr = init_lr * (1 - float(iter) / max_iter) ** power
        if len(device_ids) > 1:
            optimizer.module.param_groups[0]["lr"] = new_lr
            if len(optimizer.module.param_groups) == 2:
                optimizer.module.param_groups[1]["lr"] = 10 * new_lr
        else:
            optimizer.param_groups[0]["lr"] = new_lr
            if len(optimizer.param_groups) == 2:
                optimizer.param_groups[1]["lr"] = 10 * new_lr




def add_train_args(arg_parser):
    # Path related arguments
    arg_parser.add_argument('--data_root_path', type=str, default='./GTA5',
                            help="the root path of dataset")
    arg_parser.add_argument('--list_path', type=str, default='./GTA5',
                            help="the root path of dataset")
    arg_parser.add_argument('--checkpoint_dir', default="./log/gta5_pretrain_2",
                            help="the path of ckpt file")
    arg_parser.add_argument('--xuanran_path', default=None,
                            help="the path of ckpt file")

    # Model related arguments
    arg_parser.add_argument('--weight_loss', default=True,
                            help="if use weight loss")
    
    arg_parser.add_argument('--backbone', default='Deeplab50_CLASS_INW',
                            help="backbone of encoder")
    arg_parser.add_argument('--bn_momentum', type=float, default=0.1,
                            help="batch normalization momentum")
    arg_parser.add_argument('--imagenet_pretrained', type=str2bool, default=True,
                            help="whether apply imagenet pretrained weights")
    arg_parser.add_argument('--pretrained_ckpt_file', type=str, default=None,
                            help="whether apply pretrained checkpoint")
    arg_parser.add_argument('--continue_training', type=str2bool, default=False,
                            help="whether to continue training ")
    arg_parser.add_argument('--show_num_images', type=int, default=2,
                            help="show how many images during validate")

    # train related arguments
    arg_parser.add_argument('--seed', default=12345, type=int,
                            help='random seed')
    arg_parser.add_argument('--gpu', type=str, default="0",
                            help=" the num of gpu")
    arg_parser.add_argument('--batch_size_per_gpu', default=32, type=int,
                            help='input batch size')
    arg_parser.add_argument('--alpha', default=0.3, type=int,
                            help='input mix alpha')

    # dataset related arguments
    arg_parser.add_argument('--dataset', default='gta5', type=str,
                            help='dataset choice')
    arg_parser.add_argument('--val_dataset', type=str, default='cityscapes',
                        help='a list consists of cityscapes, mapillary, gtav, bdd100k, synthia')
    arg_parser.add_argument('--base_size', default="512,512", type=str,
                            help='crop size of image')
    arg_parser.add_argument('--crop_size', default="512,512", type=str,
                            help='base size of image')
    arg_parser.add_argument('--target_base_size', default="512,512", type=str,
                            help='crop size of target image')
    arg_parser.add_argument('--target_crop_size', default="512,512", type=str,
                            help='base size of target image')
    arg_parser.add_argument('--num_classes', default=2, type=int,
                            help='num class of mask')
    arg_parser.add_argument('--data_loader_workers', default=8, type=int,
                            help='num_workers of Dataloader')
    arg_parser.add_argument('--pin_memory', default=2, type=int,
                            help='pin_memory of Dataloader')
    arg_parser.add_argument('--split', type=str, default='train',
                            help="choose from train/val/test/trainval/all")
    arg_parser.add_argument('--random_mirror', default=True, type=str2bool,
                            help='add random_mirror')
    arg_parser.add_argument('--random_crop', default=False, type=str2bool,
                            help='add random_crop')
    arg_parser.add_argument('--resize', default=True, type=str2bool,
                            help='resize')
    arg_parser.add_argument('--gaussian_blur', default=True, type=str2bool,
                            help='add gaussian_blur')
    arg_parser.add_argument('--numpy_transform', default=True, type=str2bool,
                            help='image transform with numpy style')
    arg_parser.add_argument('--color_jitter', default=True, type=str2bool,
                            help='image transform with numpy style')

    # optimization related arguments

    arg_parser.add_argument('--freeze_bn', type=str2bool, default=False,
                            help="whether freeze BatchNormalization")
    
    arg_parser.add_argument('--momentum', type=float, default=0.9)
    arg_parser.add_argument('--weight_decay', type=float, default=5e-4)

    arg_parser.add_argument('--lr', type=float, default=5e-4,
                            help="init learning rate ")
    arg_parser.add_argument('--iter_max', type=int, default=200000,
                            help="the maxinum of iteration")
    arg_parser.add_argument('--iter_stop', type=int, default=200000,
                            help="the early stop step")
    arg_parser.add_argument('--each_epoch_iters', default=1000,
                            help="the path of ckpt file")
    arg_parser.add_argument('--poly_power', type=float, default=0.9,
                            help="poly_power")
    arg_parser.add_argument('--selected_classes', default=[0,1],
                            help="poly_power")

    # multi-level output

    arg_parser.add_argument('--multi', default=False, type=str2bool,
                            help='output model middle feature')
    arg_parser.add_argument('--lambda_seg', type=float, default=0.1,
                            help="lambda_seg of middle output")
    return arg_parser


def init_args(args):
    # args.batch_size = args.batch_size_per_gpu * ceil(len(args.gpu) / 2)
    args.batch_size = args.batch_size_per_gpu
    print("batch size: ", args.batch_size)

    train_id = str(args.dataset)

    crop_size = args.crop_size.split(',')
    base_size = args.base_size.split(',')
    if len(crop_size) == 1:
        args.crop_size = int(crop_size[0])
        args.base_size = int(base_size[0])
    else:
        args.crop_size = (int(crop_size[0]), int(crop_size[1]))
        args.base_size = (int(base_size[0]), int(base_size[1]))


    target_crop_size = args.target_crop_size.split(',')
    target_base_size = args.target_base_size.split(',')
    if len(target_crop_size) == 1:
        args.target_crop_size = int(target_crop_size[0])
        args.target_base_size = int(target_base_size[0])
    else:
        args.target_crop_size = (int(target_crop_size[0]), int(target_crop_size[1]))
        args.target_base_size = (int(target_base_size[0]), int(target_base_size[1]))

  



    args.class_16 =  False
    args.class_13 =  False

   

    # set seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    return args, train_id


if __name__ == '__main__':
    print(torch.cuda.is_available())
    assert LooseVersion(torch.__version__) >= LooseVersion('1.0.0'), 'PyTorch>=1.0.0 is required'
    warnings.filterwarnings('ignore')
    arg_parser = argparse.ArgumentParser()
    arg_parser = add_train_args(arg_parser)

    args = arg_parser.parse_args()
    args, train_id = init_args(args)

    agent = Trainer(args=args, cuda=True, train_id=train_id)
    agent.main()
