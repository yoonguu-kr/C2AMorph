import numpy as np
import torch.utils.data as Data
import nibabel as nib
import torch
import itertools
import os
import re
from datetime import datetime
import logging
from pynvml.smi import nvidia_smi
from config import _C as cfg
import random
import warnings
import sys
import pynvml
from torch.optim import lr_scheduler


def get_scheduler(optimizer, lr_policy, alpha):
    if lr_policy == 'lambda':
        #lr = lr * lambda(epoch)
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch:alpha ** epoch)
    elif lr_policy == 'step':
        #each step, lr = gamma * lr
        scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_decay_iters, gamma=0.1)
    elif lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    else:
        return NotImplementedError('learning rate policy is not implemented')
    return scheduler

def dice_orignal(im1, atlas):
    '''
    이거 LapIRN에서 사용한 dice score임
    '''
    unique_class = np.unique(atlas)
    dice = 0
    num_count = 0
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / num_count


def dice(im1, atlas, labels=None):
    '''
    im1 = val_warpped_label.data.int().cpu().numpy()[0,0,:,:,:]
    atlas = val_Y_label.cpu().int().numpy()[0,0,:,:,:]
    labels = list(num_classes)
    '''

    if labels is None:
        unique_class = np.unique(atlas)
    else:
        unique_class = labels.copy()
    dice = 0
    num_count = 0
    eps = 1e-7
    for i in unique_class:
        if (i == 0) or ((im1 == i).sum() == 0) or ((atlas == i).sum() == 0):
            continue
        # print(f'i : {i}')

        sub_dice = np.sum(atlas[im1 == i] == i) * 2.0 / (np.sum(im1 == i) + np.sum(atlas == i))
        dice += sub_dice
        num_count += 1
        # print(sub_dice)
    # print(num_count, len(unique_class)-1)
    return dice / (num_count + eps)

def dice_multi(vol1, vol2, num_classes=None, nargout=1):
    '''
    Dice [1] volume overlap metric

    The default is to *not* return a measure for the background layer (label = 0)

    [1] Dice, Lee R. "Measures of the amount of ecologic association between species."
    Ecology 26.3 (1945): 297-302.

    Parameters
    ----------
    vol1 : nd array. The first volume (e.g. predicted volume)
    vol2 : nd array. The second volume (e.g. "true" volume)
    num_classes : optional vector of labels on which to compute Dice.
        If this is not provided, Dice is computed on all non-background (non-0) labels
    nargout : optional control of output arguments. if 1, output Dice measure(s).
        if 2, output tuple of (Dice, labels)

    Output
    ------
    if nargout == 1 : dice : vector of dice measures for each labels
    if nargout == 2 : (dice, labels) : where labels is a vector of the labels on which
        dice was computed

    '''


    if num_classes is None:
        num_classes = np.unique(np.concatenate((vol1, vol2)))
        num_classes = np.delete(num_classes, np.where(num_classes == 0))  # remove background

    dicem = np.zeros(len(num_classes))
    dicem2 = np.zeros(len(num_classes))
    for idx, lab in enumerate(num_classes):
        # print(f'idx : {idx} lab : {lab}')
        vol1l = vol1 == lab
        vol2l = vol2 == lab
        top = 2. * np.sum(np.logical_and(vol1l, vol2l))
        bottom = np.sum(vol1l) + np.sum(vol2l)
        bottom = np.maximum(bottom, np.finfo(float).eps)  # add epsilon.
        dicem[idx] = 1.0 * top / bottom
        dicem2[idx] = 1.0 * int(np.sum(vol1l)) / (vol1l.shape[2] * vol1l.shape[3] * vol1l.shape[4])

    if nargout == 1:
        return dicem, dicem2
    else:
        return (dicem, dicem2, num_classes)


class best_dc:
    '''
    save the best_dc value class
    '''
    def __init__(self):
        self.best_dice = 0
        self.best_dice_epoch=0
    def best_memory(self, dice_score, epoch):
        '''
        if input parameter dice_score is best dice score, it turns True
        :param dice_score:
        :return:
        '''
        if self.best_dice <= dice_score:
            self.best_dice = dice_score
            self.best_dice_epoch = epoch
            print(f"self.best_dice : {self.best_dice} at epoch :{self.best_dice_epoch}")
            return True
        else:
            return False

    def get_best_dc(self):
        return self.best_dice, self.best_dice_epoch

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n

        self.avg = self.sum / self.count
# a= np.array([3,2,1])
# a.astype(float)
# type(a[0])
# a1 = AverageMeter()
# a1.update(np.array([1,2,3]))
# print(f'top_1_meter.avg : {a1.avg}')
# a1.update(np.array([3,2,1]))
# print(f'top_1_meter.avg : {a1.avg}')
# a1.update(np.array([6,6,6]))
# print(f'top_1_meter.avg : {a1.avg}')
# A = best_dc()
# print("A.get_best_dc() : ", A.get_best_dc())
# A.best_memory(0.5, 1)
# result_A1 = A.get_best_dc()
# A.get_best_dc()[0]
# for i in range(1,5):
#     print(f'i : {i}')
#     best_TF = A.best_memory(A.get_best_dc()[0] * 1.1, i)
#
#     print("A.get_best_dc() : ", A.get_best_dc())
#     print(f"best_TF : {best_TF}\n ")




def setup_logger(name, save_dir, filename="log.txt"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(message)s", "%Y-%m-%d %H:%M")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger

def cuda_check(logger_config):
    if not torch.cuda.is_available():
        assert torch.cuda.is_available(), "torch cuda is not available, Please configure the gpu first"
    else:
        #   TO USE pynvml.nvmlSystemGetDriverVersion() function
        pynvml.nvmlInit()
        logger_config.info("=" * 50)
        logger_config.info("This is running in GPU")
        logger_config.info("torch.cuda.is_available() : {}".format(torch.cuda.is_available()))
        logger_config.info("torch.cuda.is_available() : {}".format(torch.cuda.device_count()))
        logger_config.info("Driver Version : {}".format(pynvml.nvmlSystemGetDriverVersion()))

        deviceCount = pynvml.nvmlDeviceGetCount()
        for i in range(deviceCount):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            logger_config.info("Device {} : {}".format(i, pynvml.nvmlDeviceGetName(handle)))
        nvsmi = nvidia_smi.getInstance()
        logger_config.info("=" * 50+'\n')
        # GPU 할당 변경하기
        # 원하는 GPU 번호 입력

        device = torch.device(f'cuda:{cfg.MODEL.GPU_NUM}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)  # change allocation of current GPU
        logger_config.info('Current cuda device '.format(torch.cuda.current_device()))
        logger_config.info('Using device : {} >>{}'.format(device, torch.cuda.get_device_name(device)))
        logger_config.info(device.type)

        if device.type == 'cuda':
            logger_config.info(torch.cuda.get_device_name(cfg.MODEL.GPU_NUM))
            logger_config.info('Memory Usage:')
            logger_config.info('Allocated: {} GB'.format(round(torch.cuda.memory_allocated(cfg.MODEL.GPU_NUM) / 1024 ** 3, 1)))
            logger_config.info('Reserved:  {} GB'.format(round(torch.cuda.memory_reserved(device) / 1024 ** 3, 1)))
            logger_config.info(torch.cuda.get_device_properties('cuda:{}'.format(cfg.MODEL.GPU_NUM)))

        elif device.type == 'cpu':
            assert device.type != 'cpu', "device is running in cpu, please configure it first"


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Creating directory : ", directory)

def init_env(gpu_id='0', seed=42):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    warnings.filterwarnings('ignore')


def tr_val_test(names, tr_val_ratio, test=False):
    '''
    if test is False
     --> training data(ratio:8), validation data(ratio:2)
    else test is True
     --> training data(ratio:8), validation data(ratio:2 - 2), test_check_data1, test_check_data2
    :param names: list of names
    :param tr_val_ratio:
    :return:
    '''
    if test:
        if int(len(names) * (1-tr_val_ratio)) >=2:
            num_tr = int(len(names) * tr_val_ratio)
            names_tr = names[:num_tr]
            names_test1 = names[num_tr]
            names_test2 = names[num_tr+1]
            names_val= names[num_tr+2:]
            return names_tr, names_val, names_test1, names_test2
        else:
            raise Exception("Test data number is less than 2")
    else:
        # raise Exception("This only results Tr, val and 2 test to run this code")
        num_tr = int(len(names) * tr_val_ratio)
        names_tr = names[:num_tr]
        names_val = names[num_tr:]
        return names_tr, names_val



def fileNameUpdate(directory, *argv, dateupdate=False, num_version=None):
    '''
    This is making a file name
    " name_prefix +"_YYMMDD" +"fold"+"_version"."ext"
    - ex)
        " name_prefix +"_YYMMDD" + "_v1"
        " name_prefix +"_YYMMDD" + "_v2"
        " name_prefix +"_YYMMDD" + "_v3"

    :param directory: directory
    :param dateupdate: whether YYMMDD is added or not
    :param num_version: version number
    :return: filename_YYMMDD_version.ext
    '''

    filename_check =''
    for arg in argv:
        filename_check+= str(arg)
    # print("filename_check : ", filename_check)

    # files = [file for file in os.listdir(directory) if filename_check in file]
    files = [file for file in os.listdir(directory) if filename_check in file and file[:file.rindex("_")] == filename_check]

    #To check the version name
    if num_version:
        file_version = num_version
    else:
        if files:
            file_version = max([int(re.findall(r'\d+', file)[-1]) for file in files if filename_check in file]) + 1
        else:
            file_version = 1

    now = datetime.now()
    current_time = now.strftime("%Y%m%d")[2:]

    if dateupdate:
        filename = filename_check+"_"+current_time+"_v"+str(file_version)
    else:
        filename= filename_check+"_v"+str(file_version)

    return filename, file_version


def generate_grid(imgshape):
    x = np.arange(imgshape[0])
    y = np.arange(imgshape[1])
    z = np.arange(imgshape[2])
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def generate_grid_unit(imgshape):
    x = (np.arange(imgshape[0]) - ((imgshape[0] - 1) / 2)) / (imgshape[0] - 1) * 2
    y = (np.arange(imgshape[1]) - ((imgshape[1] - 1) / 2)) / (imgshape[1] - 1) * 2
    z = (np.arange(imgshape[2]) - ((imgshape[2] - 1) / 2)) / (imgshape[2] - 1) * 2
    grid = np.rollaxis(np.array(np.meshgrid(z, y, x)), 0, 4)
    grid = np.swapaxes(grid, 0, 2)
    grid = np.swapaxes(grid, 1, 2)
    return grid


def transform_unit_flow_to_flow(flow):
    x, y, z, _ = flow.shape
    flow[:, :, :, 0] = flow[:, :, :, 0] * (z-1)/2
    flow[:, :, :, 1] = flow[:, :, :, 1] * (y-1)/2
    flow[:, :, :, 2] = flow[:, :, :, 2] * (x-1)/2

    return flow


def flow_unit(flow):
    b, x, y, z, c = flow.shape
    flow[:, :, :, :, 0] = flow[:, :, :, :, 0] * (z-1)/2
    flow[:, :, :, :, 1] = flow[:, :, :, :, 1] * (y-1)/2
    flow[:, :, :, :, 2] = flow[:, :, :, :, 2] * (x-1)/2

    return flow

def load_3D(name):
    X = nib.load(name)
    X = X.get_fdata()
    return X


def load_4D(name):
    X = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + X.shape)
    return X


def load_5D(name):
    X = fixed_nii = nib.load(name)
    X = X.get_fdata()
    X = np.reshape(X, (1,) + (1,) + X.shape)
    return X


def imgnorm(img):
    max_v = np.max(img)
    min_v = np.min(img)

    norm_img = (img - min_v) / (max_v - min_v)
    return norm_img


def save_img(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)


def save_img_nii(I_img, savename):
    affine = np.diag([1, 1, 1, 1])
    new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    nib.save(new_img, savename)


def save_flow(I_img,savename,header=None,affine=None):
    if header is None or affine is None:
        affine = np.diag([1, 1, 1, 1])
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=None)
    else:
        new_img = nib.nifti1.Nifti1Image(I_img, affine, header=header)

    nib.save(new_img, savename)

