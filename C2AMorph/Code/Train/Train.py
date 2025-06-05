'''
model에 Attention을 넣어서 해결한 코드
'''

import glob
import os
import sys
from argparse import ArgumentParser
from datetime import datetime
import re
import numpy as np
import torch
import torch.utils.data as Data
import torch.nn.functional as F
import matplotlib.pyplot as plt
import nibabel as nib
import medpy
import medpy.metric
from medpy.metric.binary import dc, jc, hd, hd95, asd, assd, precision, recall, sensitivity, specificity
from sklearn.model_selection import KFold

IsPythonConsole = False

if IsPythonConsole:
    print("This is running on python console")
    print('current loc : ', os.getcwd())
    dir_C2AMorph = os.getcwd()

    dir_Code = os.path.join(dir_C2AMorph, 'Code')
    dir_models = os.path.join(dir_C2AMorph, 'models')
    dir_Train = os.path.join(dir_C2AMorph, 'Train')
    dir_Result = os.path.join(dir_C2AMorph, 'Result')
    num_worker = 0
    basefile = 'Pycharm_Console'

    assert os.path.isdir(dir_Code), "{} is not exist".format(dir_Code)
    assert os.path.isdir(dir_models), "{} is not exist".format(dir_models)

else:
    dir_Train = os.getcwd()
    dir_Code = os.path.dirname(dir_Train)
    dir_C2AMorph = os.path.dirname(dir_Code)
    dir_models = os.path.join(dir_Code, 'models')
    dir_Result = os.path.join(dir_C2AMorph, 'Result')
    num_worker = 4
    basefile = os.path.basename(__file__)[:os.path.basename(__file__).find(".")]

    assert os.path.isdir(dir_Code), "{} is not exist".format(dir_Code)
    assert os.path.isdir(dir_models), "{} is not exist".format(dir_models)

print(f'dir_ConLapIRN_Code : {dir_Code}')
print(f'dir_Train : {dir_Train}')
print(f'basefile : {basefile}')

sys.path.append(dir_Code)
sys.path.append(dir_models)
sys.path.append(dir_C2AMorph)

from config import _C as cfg
from Functions import generate_grid, flow_unit, \
    generate_grid_unit, createFolder, fileNameUpdate, setup_logger, cuda_check, init_env,save_img, save_flow, transform_unit_flow_to_flow, load_4D, imgnorm, best_dc, tr_val_test, dice, AverageMeter, dice_multi
from Data_augmentation import CustomCompose,RandomFlip, RandomRotate, AddGaussianNoise,RandomCrop, CenterCrop, ToFloatTensor
from loader import Dataset_LPBA40,Dataset_LPBA40_ValTest,  Dataset_epoch, Dataset_epoch_validation
# from model_Unet_conditional1 import model_Unet_conditional1, SpatialTransform_unit, SpatialTransformNearest_unit
from model import C2AMorph, SpatialTransformer
from losses import smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC, contrastive_loss, kl_loss, lamda_mse_loss


lr = cfg.SOLVER.LEARNING_RATE


checkpoint = cfg.SOLVER.CHECKPOINT
datapath = cfg.DATASET.DATA_PATH

gpu_num = cfg.MODEL.GPU_NUM
num_epoch = cfg.DATASET.NUM_EPOCHS
datatype = cfg.DATASET.DATATYPE
load_model = cfg.SOLVER.LOAD_MODEL
cfg.PLATFORM.basefile = basefile
doubleF = cfg.MODEL.FDoubleConv
doubleB = cfg.MODEL.BDoubleConv
data_norm = cfg.DATALOADER.NORM

start_channel = cfg.MODEL.Start_Channel
ch_magnitude = cfg.MODEL.Ch_magnitude

loss1_name = cfg.SOLVER.LOSS1
loss2_name = cfg.SOLVER.LOSS2
loss3_name = cfg.SOLVER.LOSS3
hyp_KL = cfg.SOLVER.HYP_KL
max_KL = cfg.SOLVER.MAX_KL

hyp_smooth = cfg.SOLVER.HYP_SMOOTH
max_smooth = cfg.SOLVER.MAX_SMOOTH

hyp_CL = cfg.SOLVER.HYP_CL
max_CL =cfg.SOLVER.MAX_CL

hyp_antifold = cfg.SOLVER.HYP_ANTIFOLD
max_antifold= cfg.SOLVER.MAX_ANTIFOLD

batch_size = cfg.DATALOADER.BATCH_SIZE

if cfg.DATASET.DATATYPE == 'LPBA40_small':

    '''
    LPBA40 is appied to 40(fix)*40(moving) = 1600
    '''

    names = sorted(glob.glob(cfg.DATASET.DATA_PATH_IMGS))
    labels = sorted(glob.glob(cfg.DATASET.DATA_PATH_LABELS))
    fixed_test_path = sorted(glob.glob(cfg.DATASET.DATA_PATH_IMGS))[0]

    atlas_exist = False
    label_list = None
    imgshape = (72, 96, 72)


torch.autograd.set_detect_anomaly(True)

# Create folders & Check the new file version that this file make
createFolder(dir_Result)
dir_datatype = os.path.join(dir_Result, datatype)
createFolder(dir_datatype)

loss_names = {}
for num, loss_name in enumerate([loss1_name, loss2_name, loss3_name]):
    print(f'num : {num}, loss_name: {loss_name}')
    var_name = "loss%d_name" %(num+2)
    loss_names[num] = [loss_name]

for k in loss_names:
    print(f'k : {k} : {loss_names[k]}')
    if k<=2:
        if loss_names[k][0] == 'REG':
            loss_names[k].append(hyp_smooth)
            loss_names[k].append(max_smooth)

        elif loss_names[k][0] == 'KL':
            loss_names[k].append(hyp_KL)
            loss_names[k].append(max_KL)

        elif loss_names[k][0] == 'CL':
            loss_names[k].append(hyp_CL)
            loss_names[k].append(max_CL)
        elif loss_names[k][0] == 'Jacobian':
            loss_names[k].append(hyp_antifold)
            loss_names[k].append(max_antifold)
    else:
        loss_names[k].append(0)
        loss_names[k].append(0)


file_name_loss = f'{loss_names[0][0]}_con1_{loss_names[1][2]}{loss_names[1][0]}_hyp_{loss_names[2][1]}{loss_names[2][0]}'
dir_losses = os.path.join(dir_datatype, file_name_loss)
createFolder(dir_losses)



# logger setup
Log_name_config = "Log_config.log"
Log_name_eval = 'Log_eval.log'
Log_best_test = 'Log_best_test.log'

logger_config = setup_logger('config', dir_losses, filename=Log_name_config)
logger_config.info(cfg)

cuda_check(logger_config)
logger_config.handlers.clear()

device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU
logger_eval = setup_logger('eval', dir_losses, filename=Log_name_eval)

def train_con1Unet(data_name, dir_save, doubleF=True, doubleB=True, batchsize =1):
    '''
    dir_save = dir_epoch
    '''
    print("Training Unet with contrastive learning and KL loss...")
    # dir_save = dir_rangeflow

    device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
    model =C2AMorph(1, start_channel, ch_magnitude=ch_magnitude, use_gpu=True, img_size=imgshape, batchsize= batchsize)
    for k in loss_names:
        print(f'k : {k} : {loss_names[k]}')
        if loss_names[k][0] == 'MSE':
            loss_names[k].append(lamda_mse_loss())
        elif loss_names[k][0] == 'NCC':
            loss_names[k].append(multi_resolution_NCC(win=7, scale=3))

        elif loss_names[k][0] == 'REG':
            loss_names[k].append(smoothloss)
        elif loss_names[k][0] == 'KL':
            loss_names[k].append(kl_loss())
        elif loss_names[k][0] == 'CL':
            loss_names[k].append(contrastive_loss(batch_size=batchsize))
        elif loss_names[k][0] == 'Jacobian':
            loss_names[k].append(neg_Jdet_loss)




    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    dataLPBA_tr = Dataset_LPBA40(names, norm=data_norm, img_size=imgshape)
    dataLPBA_tr.initialize()

    training_generator = Data.DataLoader(dataLPBA_tr, batch_size=batchsize,
                                         shuffle=True, num_workers=num_worker)

    dataLPBA_val = Dataset_LPBA40_ValTest(names, norm=data_norm, img_size=imgshape)
    dataLPBA_val.initialize('val')
    best_dice = best_dc()

    iteration = len(training_generator)
    lossall = np.zeros((4, iteration + 1, num_epoch))
    spatial_transform = SpatialTransformer(imgshape)
    spatial_transform.to(device)

    for epoch in range(num_epoch):
        for i, data in enumerate(training_generator):
            if i> 5:
                break
            X, Y, _, _ = data
            X = X.to(device).float()
            Y = Y.to(device).float()
            reg_code = torch.rand(1, dtype=X.dtype, device=X.device).unsqueeze(dim=0)  # torch.Size([1, 1])
            warped_image, deform, mean, sigma, f_x, f_y = model(X, Y, reg_code)
            loss_sim = loss_names[0][-1](warped_image, Y)

            _, _, x, y, z = deform.shape
            deform[:, 0, :, :, :] = deform[:, 0, :, :, :] * (z - 1)
            deform[:, 1, :, :, :] = deform[:, 1, :, :, :] * (y - 1)
            deform[:, 2, :, :, :] = deform[:, 2, :, :, :] * (x - 1)

            f_x = F.normalize(f_x, dim=1)
            f_y = F.normalize(f_y, dim=1)


            if loss_names[1][0] == 'REG':
                loss1 = loss_names[1][-1](deform)
                max1 = loss_names[1][2]
            elif loss_names[1][0] == 'KL':
                # CLMorph에서는 이걸 Smooth loss로 쓰긴함
                loss1 = loss_names[1][-1](mean, sigma)
                max1 = loss_names[1][2]
            elif loss_names[1][0] == 'CL':
                loss1 = loss_names[1][-1](f_x, f_y)
                max1 = loss_names[1][2]

            if loss_names[2][0] == 'REG':
                loss2 = loss_names[2][-1](deform)
                hyp2 = loss_names[2][1]
            elif loss_names[2][0] == 'KL':
                # CLMorph에서는 이걸 Smooth loss로 쓰긴함
                loss2 = loss_names[2][-1](mean, sigma)
                hyp2 = loss_names[2][1]
            elif loss_names[2][0] == 'CL':
                loss2 = loss_names[2][-1](f_x, f_y)
                hyp2 = loss_names[2][1]

            #original hyp_antifold =0 --> LapIRN
            # in CLMorph, they use KL loss as smooth loss
            weight1 = reg_code * max1
            loss = loss_sim + weight1 * loss1 +hyp2 * loss2

            optimizer.zero_grad()  # clear gradients for this training step
            loss.backward()  # backpropagation, compute gradients
            optimizer.step()  # apply gradients

            sys.stdout.write(
                "\r" + 'epoch "{0}" iter : {1}-> training loss "{2:.4f}" =  "{3}" "{4:4f}" + con1 "{5}" * {6}_loss "{7:.4f}" +hyp2 "{8}"*{9}_loss "{10:.4f}" '.format(
                    epoch, i, loss.item(), loss_names[0][0], loss_sim.item(), loss_names[1][2], loss_names[1][0],
                    loss1.item(), loss_names[2][1], loss_names[2][0], loss2.item()))
            sys.stdout.flush()

            lossall[:, i, epoch] = np.array(
                [loss.item(), loss_sim.item(), loss1.item(), loss2.item()])
        print("one epoch pass")
        if (epoch % checkpoint == 0):
            dir_ckpoint = os.path.join(dir_save,  str(epoch) + '.pth')
            torch.save(model.state_dict(), dir_ckpoint)
            dir_npy = os.path.join(dir_save, str(epoch) + '.npy')
            np.save(dir_npy, lossall)

    dir_modelname = os.path.join(dir_save, str(epoch) + '.pth')
    torch.save(model.state_dict(), dir_modelname)

    logger_eval.info("Best Dice : {} Epoch : {} \n".format(best_dice.get_best_dc()[0], best_dice.get_best_dc()[1]))
    dir_npy1 = os.path.join(dir_save, 'loss_' +  str(epoch) + '.npy')
    np.save(dir_npy1, lossall)
    logger_eval.handlers.clear()



def test(dir_save,max_only=False, brief = True, reg_input1s = [0.1,1,10]):
    '''
    BEWARE!!!!!
    those below is string!!!!

    :param model: model name
    :param multi_dataset:
    :param epoch:
    :return:
    '''

    Log_test_total = f'Log_test_max_only_{max_only}_brief_{brief}_total.log'
    logger_test_total = setup_logger('test_total', dir_save, filename=Log_test_total)
    files_model = [pth_file for pth_file in os.listdir(dir_save) if 'pth' in pth_file]
    if max_only == True:
        files_npy = [pth_file for pth_file in os.listdir(dir_save) if 'pth' in pth_file]
        num_max = max([int(re.findall(r'\d+', number)[-1]) for number in files_npy])
        files_model = [file for file in files_npy if str(num_max) in file]

    logger_test_total.info(
        f'\tepoch \tfile_ep \treg_input1'
        f'\t\tstr_mean \t\tstr_std \t'
        f'\tdice_nanmean_list.mean() \tdice_nanmean_list.std() \t\tdice_sum_list.mean() \tdice_sum_list.std()'
        f'\t\tavg_dc.mean() \tavg_dc.std() \tavg_hd.mean() \tavg_hd.std() \tavg_hd95.mean()  \tavg_hd95.std()'
        f'\tavg_prc.mean() \tavg_prc.std() \tavg_rcl.mean() \tavg_rcl.std() \tavg_sensi.mean() \tavg_sensi.std()'
        f'\tavg_speci.mean() \tavg_speci.std() \tavg_asd.mean() \tavg_asd.std() \tavg_assd.mean() \tavg_assd.std() \tavg_jc.mean() \tavg_jc.std()')

    for epoch_file in files_model:
        print(f'file_ep : {epoch_file}')
        epoch = re.findall(r'\d+', epoch_file)[-1]
        folder_epoch = os.path.join(dir_save, 'brief_' + str(brief), 'epoch' + epoch)
        createFolder(folder_epoch)

        model_path = os.path.join(dir_save, epoch_file)
        print(f"loading best model : {model_path}")

        init_env(str(gpu_num))
        device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)  # change allocation of current GPU

        model =C2AMorph(1, start_channel, ch_magnitude=ch_magnitude, use_gpu=True, img_size=imgshape, is_training= False)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        dataLPBA_test = Dataset_LPBA40_ValTest(names, norm=data_norm)
        dataLPBA_test.initialize('test')
        test_generator = Data.DataLoader(dataLPBA_test, batch_size=1, shuffle=False, num_workers=num_worker)

        spatial_transform = SpatialTransformer(imgshape)
        spatial_transform.to(device)




        for reg_input1 in reg_input1s:
            dir_reg_input1 = os.path.join(folder_epoch, 'reg1_' + str(reg_input1))
            createFolder(dir_reg_input1)

            dsc_list = []
            dice_nanmean_list = []
            dice_sum_list = []

            avg_dc = []
            avg_hd = []
            avg_hd95 = []
            avg_prc = []
            avg_rcl = []
            avg_sensi = []
            avg_speci = []
            avg_asd = []
            avg_assd = []
            avg_jc = []

            model.eval()
            Log_test = f'Log_test_max_only_{max_only}_brief_{brief}.log'
            logger_test = setup_logger('test', dir_reg_input1, filename=Log_test)
            logger_test.info(
                f'\tcur_iter \tfile_ep \treg_input1 '
                f'\t\tstr_mean \t\tstr_std \t'
                f'\tdice_nanmean_list.mean() \tdice_nanmean_list.std() \t\tdice_sum_list.mean() \tdice_sum_list.std()'
                f'\t\tavg_dc.mean() \tavg_dc.std() \tavg_hd.mean() \tavg_hd.std() \tavg_hd95.mean()  \tavg_hd95.std()'
                f'\tavg_prc.mean() \tavg_prc.std() \tavg_rcl.mean() \tavg_rcl.std() \tavg_sensi.mean() \tavg_sensi.std()'
                f'\tavg_speci.mean() \tavg_speci.std() \tavg_asd.mean() \tavg_asd.std() \tavg_assd.mean() \tavg_assd.std() \tavg_jc.mean() \tavg_jc.std()')

            with torch.no_grad():
                for i, data in enumerate(test_generator):
                    if brief:
                        if i > 5:
                            break
                    val_X, val_Y, val_X_label, val_Y_label = data[0].to(device), data[1].to(device), data[2].to(
                                    device), data[3].to(device)
                    # break
                    # normalize image to [0, 1]
                    norm = data_norm
                    if norm:
                        val_X = imgnorm(val_X)
                        val_Y = imgnorm(val_Y)
                        # print(f'fixed_img.shape : {fixed_img.shape}')


                    reg_code = torch.tensor([reg_input1], dtype=val_X.dtype, device=val_X.device).unsqueeze(dim=0)
                    val_warpped, val_deform, _, _, _, _ = model(val_X, val_Y, reg_code)
                    val_warpped_label = spatial_transform(val_X_label, val_deform, mode="nearest")

                    val_X_cpu = val_X.data.cpu().numpy()[0, 0, :, :, :]
                    val_Y_cpu = val_Y.data.cpu().numpy()[0, 0, :, :, :]
                    val_X_label_cpu = val_X_label.data.cpu().numpy()[0, 0, :, :, :]
                    val_Y_label_cpu = val_Y_label.data.cpu().numpy()[0, 0, :, :, :]

                    val_warpped_cpu = val_warpped.data.cpu().numpy()[0, 0, :, :, :]
                    val_deform_cpu = val_deform.data.cpu().numpy()[0, :, :, :, :].transpose(1, 2, 3, 0)
                    val_warpped_label_cpu = val_warpped_label.data.cpu().numpy()[0, 0, :, :, :]


                    if i <= 2:
                        dir_fix = os.path.join(dir_reg_input1, f'{i}_fixed.nii.gz')
                        save_img(val_Y_cpu, dir_fix)

                        dir_fix_label = os.path.join(dir_reg_input1, f'{i}_fixed_label.nii.gz')
                        save_img(val_Y_label_cpu, dir_fix_label)

                        dir_mov = os.path.join(dir_reg_input1, f'{i}_moving.nii.gz')
                        save_img(val_X_cpu, dir_mov)

                        dir_mov_label = os.path.join(dir_reg_input1, f'{i}_moving_label.nii.gz')
                        save_img(val_X_label_cpu, dir_mov_label)

                        dir_vectorfield = os.path.join(dir_reg_input1, f'{i}_vector_field.nii.gz')
                        save_flow(val_deform_cpu, dir_vectorfield)

                        dir_warppedMov = os.path.join(dir_reg_input1, f'{i}_warped_moving.nii.gz')
                        save_img(val_warpped_cpu, dir_warppedMov)

                        dir_warppedlabel = os.path.join(dir_reg_input1, f'{i}_warped_label.nii.gz')
                        save_img(val_warpped_label_cpu, dir_warppedlabel)
                    dsc, volume = dice_multi(val_warpped_label.data.int().cpu().numpy(),
                                             val_Y_label.cpu().int().numpy())
                    dsc_list.append(dsc)

                    dice_ = np.nanmean(dsc)
                    dice_nanmean_list.append(dice_)
                    dice_sum = dice(val_warpped_label.data.int().cpu().numpy()[0, 0, :, :, :],
                                    val_Y_label.cpu().int().numpy()[0, 0, :, :, :])
                    dice_sum_list.append(dice_sum)
                    print(f'i : {i}, reg_input1 : {reg_input1} dice_ : {dice_} dice_sum: {dice_sum}')

                    # a = 1.0 / 100
                    a = 1.0
                    spacing = [a, a, a]
                    avg_dc.append(dc(val_warpped_label_cpu, val_Y_label_cpu))
                    avg_prc.append(precision(val_warpped_label_cpu, val_Y_label_cpu))
                    avg_rcl.append(recall(val_warpped_label_cpu, val_Y_label_cpu))
                    avg_sensi.append(sensitivity(val_warpped_label_cpu, val_Y_label_cpu))
                    avg_speci.append(specificity(val_warpped_label_cpu, val_Y_label_cpu))

                    avg_hd.append(hd(result=val_warpped_cpu, reference=val_Y_cpu, voxelspacing=spacing))
                    avg_hd95.append(
                        hd95(result=val_warpped_cpu, reference=val_Y_cpu, voxelspacing=spacing))
                    avg_asd.append(
                        asd(result=val_warpped_cpu, reference=val_Y_cpu, voxelspacing=spacing))
                    avg_assd.append(assd(result=val_warpped_cpu, reference=val_Y_cpu, voxelspacing=spacing))
                    avg_jc.append(jc(val_warpped_cpu, val_Y_cpu))

                avg_dc = np.array(avg_dc)
                avg_hd = np.array(avg_hd)
                avg_hd95 = np.array(avg_hd95)
                avg_prc = np.array(avg_prc)
                avg_rcl = np.array(avg_rcl)
                avg_sensi = np.array(avg_sensi)
                avg_speci = np.array(avg_speci)
                avg_asd = np.array(avg_asd)
                avg_assd = np.array(avg_assd)
                avg_jc = np.array(avg_jc)

                dsc_list = np.array(dsc_list)
                dice_nanmean_list = np.array(dice_nanmean_list)
                dice_sum_list = np.array(dice_sum_list)

                np.save(os.path.join(dir_reg_input1, 'avg_dc.npy'), avg_dc)
                np.save(os.path.join(dir_reg_input1, 'avg_hd.npy'), avg_hd)
                np.save(os.path.join(dir_reg_input1, 'avg_hd95.npy'), avg_hd95)
                np.save(os.path.join(dir_reg_input1, 'avg_prc.npy'), avg_prc)
                np.save(os.path.join(dir_reg_input1, 'avg_rcl.npy'), avg_rcl)
                np.save(os.path.join(dir_reg_input1, 'avg_sensi.npy'), avg_sensi)
                np.save(os.path.join(dir_reg_input1, 'avg_speci.npy'), avg_speci)
                np.save(os.path.join(dir_reg_input1, 'avg_asd.npy'), avg_asd)
                np.save(os.path.join(dir_reg_input1, 'avg_assd.npy'), avg_assd)
                np.save(os.path.join(dir_reg_input1, 'avg_jc.npy'), avg_jc)
                np.save(os.path.join(dir_reg_input1, 'dsc_list.npy'), dsc_list)
                np.save(os.path.join(dir_reg_input1, 'dice_nanmean_list.npy'), dice_nanmean_list)
                np.save(os.path.join(dir_reg_input1, 'dice_sum_list.npy'), dice_sum_list)

                str_mean = ''
                for i, val in enumerate(dsc_list.sum(axis=0)):
                    str_mean += '\t{:.5f}'.format(val)

                str_std = ''
                for i, val in enumerate(dsc_list.std(axis=0)):
                    str_std += '\t{:.5f}'.format(val)

                logger_test.info(
                    f'\t{epoch} \t{reg_input1}'
                    f'\t{str_mean} \t{str_std} \t'
                    f'\t{dice_nanmean_list.mean()} \t{dice_nanmean_list.std()} \t\t{dice_sum_list.mean()} \t{dice_sum_list.std()}'
                    f'\t\t{avg_dc.mean()} \t{avg_dc.std()} \t{avg_hd.mean()} \t{avg_hd.std()} \t{avg_hd95.mean()}  \t{avg_hd95.std()}'
                    f'\t{avg_prc.mean()} \t{avg_prc.std()} \t{avg_rcl.mean()} \t{avg_rcl.std()} \t{avg_sensi.mean()} \t{avg_sensi.std()}'
                    f'\t{avg_speci.mean()} \t{avg_speci.std()} \t{avg_asd.mean()} \t{avg_asd.std()} \t{avg_assd.mean()} \t{avg_assd.std()}\t{avg_jc.mean()} \t{avg_jc.std()}')

                logger_test_total.info(
                    f'\t{epoch} \t{epoch_file} \t{reg_input1}'
                    f'\t{str_mean} \t{str_std} \t'
                    f'\t{dice_nanmean_list.mean()} \t{dice_nanmean_list.std()} \t\t{dice_sum_list.mean()} \t{dice_sum_list.std()}'
                    f'\t\t{avg_dc.mean()} \t{avg_dc.std()} \t{avg_hd.mean()} \t{avg_hd.std()} \t{avg_hd95.mean()}  \t{avg_hd95.std()}'
                    f'\t{avg_prc.mean()} \t{avg_prc.std()} \t{avg_rcl.mean()} \t{avg_rcl.std()} \t{avg_sensi.mean()} \t{avg_sensi.std()}'
                    f'\t{avg_speci.mean()} \t{avg_speci.std()} \t{avg_asd.mean()} \t{avg_asd.std()} \t{avg_assd.mean()} \t{avg_assd.std()} \t{avg_jc.mean()} \t{avg_jc.std()}')

                logger_test.handlers.clear()
    logger_test_total.handlers.clear()

def Loss_plots(dir_save):
    '''
    Read .npy file that saves loss values
    save .png file that shows loss values
    '''
    dir_loss_plots = os.path.join(dir_save, 'loss_plots')
    createFolder(dir_loss_plots)

    files_npy = [pth_file for pth_file in os.listdir(dir_save) if 'npy' in pth_file]
    num_max = max([int(re.findall(r'\d+', number)[-1]) for number in files_npy])
    [max_npy] = [file for file in files_npy if str(num_max) in file]
    dir_npyl = os.path.join(dir_save, max_npy)
    np_l1 = np.load(dir_npyl)
    np_l1_mean = np.mean(np_l1, 1)

    dir_losses = os.path.join(dir_loss_plots, 'image_losses.png')
    plt.plot(np_l1_mean[0], label='Train loss total')
    plt.plot(np_l1_mean[1], label=loss_names[0][0] + ' loss')
    plt.plot(np_l1_mean[2], label=loss_names[1][0] + ' loss')
    plt.plot(np_l1_mean[3], label=loss_names[2][0] + ' loss')
    plt.legend(frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(dir_losses)
    plt.close()

    dir_total_losses = os.path.join(dir_loss_plots, 'total_losses.png')
    plt.plot(np_l1_mean[0], label='Train loss')
    plt.legend(frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig(dir_total_losses)
    plt.close()

    # this is similarity loss
    dir_loss1 = os.path.join(dir_loss_plots, loss_names[0][0] + '_loss.png')
    plt.plot(np_l1_mean[1], label=loss_names[0][0] + ' loss')
    plt.legend(frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel(loss_names[0][0] + ' loss')
    plt.savefig(dir_loss1)
    plt.close()

    # this is loss1
    dir_loss2 = os.path.join(dir_loss_plots, loss_names[1][0] + '_losses.png')
    plt.plot(np_l1_mean[2], label=loss_names[1][0] + ' loss')
    plt.legend(frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel(loss_names[1][0] + ' Loss')
    plt.savefig(dir_loss2)
    plt.close()

    # this is loss2
    dir_loss3 = os.path.join(dir_loss_plots, loss_names[2][0] + '_losses.png')
    plt.plot(np_l1_mean[3], label=loss_names[2][0] + ' loss')
    plt.legend(frameon=False)
    plt.xlabel('Epoch')
    plt.ylabel(loss_names[2][0] + ' Loss')
    plt.savefig(dir_loss3)
    plt.close()


if __name__ == "__main__":
    print(f'main code starts')
    start_t = datetime.now()
    train_con1Unet(data_name = cfg.DATASET.DATATYPE, dir_save = dir_losses,  doubleF=True, doubleB=True,batchsize=batch_size)

    # time
    end_t = datetime.now()
    total_t = end_t - start_t

    print("Time: ", total_t.total_seconds())
    logger_eval.info("Time: {}".format(total_t.total_seconds()))

    days = int(total_t.total_seconds()) // (60 * 60 * 24)
    days_left = int(total_t.total_seconds()) % (60 * 60 * 24)

    hours = days_left // (60 * 60)
    hours_left = days_left % (60 * 60)

    mins = hours_left // (60)
    mins_left = hours_left % (60)

    print("{}days {}hours {}mins {}secs".format(days, hours, mins, mins_left))
    logger_eval.info("{}days {}hours {}mins {}secs\n\n".format(days, hours, mins, mins_left))
    start_test = datetime.now()

    test(dir_epoch, max_only=False, brief=False, reg_input1s=[1, 10])

    end_test = datetime.now()
    total_test = end_test - start_test
    logger_eval.info("total_test time: {}".format(total_test))