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
    dir_CLcLapIRN = os.getcwd()

    dir_CLcLapIRN_Code = os.path.join(dir_CLcLapIRN, 'Code')
    dir_ver3 = os.path.join(dir_CLcLapIRN_Code, 'version3')
    dir_LPBA40 = os.path.join(dir_ver3, 'LPBA40')
    dir_Train = os.path.join(dir_LPBA40, 'Train')

    dir_Data = os.path.join(dir_CLcLapIRN, 'Data')
    dir_Result = os.path.join(dir_CLcLapIRN, 'Result')

    num_worker = 0
    basefile = 'Pycharm_Console'

    assert os.path.isdir(dir_CLcLapIRN_Code), "{} is not exist".format(dir_CLcLapIRN_Code)
    assert os.path.isdir(dir_ver3), "{} is not exist".format(dir_ver3)

else:
    dir_Train = os.getcwd()
    dir_LPBA40 = os.path.dirname(dir_Train)
    dir_ver3 = os.path.dirname(dir_LPBA40)

    dir_CLcLapIRN_Code = os.path.dirname(dir_ver3)
    dir_CLcLapIRN = os.path.dirname(dir_CLcLapIRN_Code)
    dir_Data = os.path.join(dir_CLcLapIRN, 'Data')
    dir_Result = os.path.join(dir_CLcLapIRN, 'Result')
    num_worker = 4
    basefile = os.path.basename(__file__)[:os.path.basename(__file__).find(".")]
    version = re.findall(r'\d+', basefile)[-1]
    print(f'version : {version}')

    assert os.path.isdir(dir_CLcLapIRN_Code), "{} is not exist".format(dir_CLcLapIRN_Code)
    assert os.path.isdir(dir_ver3), "{} is not exist".format(dir_ver3)

print(f'dir_ConLapIRN_Code : {dir_CLcLapIRN_Code}')
print(f'dir_Train : {dir_Train}')
print(f'basefile : {basefile}')

sys.path.append(dir_CLcLapIRN_Code)
sys.path.append(dir_CLcLapIRN)
sys.path.append(dir_ver3)



from config import _C as cfg
from Functions import generate_grid, flow_unit, \
    generate_grid_unit, createFolder, fileNameUpdate, setup_logger, cuda_check, init_env,save_img, save_flow, transform_unit_flow_to_flow, load_4D, imgnorm, best_dc, tr_val_test, dice, AverageMeter, dice_multi
from Data_augmentation import CustomCompose,RandomFlip, RandomRotate, AddGaussianNoise,RandomCrop, CenterCrop, ToFloatTensor
from loader import Dataset_LPBA40,Dataset_LPBA40_ValTest,  Dataset_epoch, Dataset_epoch_validation
# from model_Unet_conditional1 import model_Unet_conditional1, SpatialTransform_unit, SpatialTransformNearest_unit
from model_con1UNet_attn import VoxelMorph3dAttn_con1, SpatialTransformer
from losses import smoothloss, neg_Jdet_loss, NCC, multi_resolution_NCC, contrastive_loss, kl_loss, lamda_mse_loss


'''
LPBA40 is appied to 40(fix)*40(moving) = 1600
'''
DATA_PATH = 'E:\\Brain\\LPBA40\\from_CLMorph'
DATA_PATH_IMGS = DATA_PATH + '/LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten/l*_to_l*.nii'
DATA_PATH_LABELS = DATA_PATH + '/LPBA40_rigidly_registered_label_pairs_small_GoodLabel/l*_to_l*.nii'
PAIR = False

names = sorted(glob.glob(DATA_PATH_IMGS))
labels = sorted(glob.glob(DATA_PATH_LABELS))
fixed_test_path = sorted(glob.glob(DATA_PATH_IMGS))[0]

atlas_exist = False
label_list = None
imgshape = (72, 96, 72)

gpu_num=0
data_norm=False

# init_env(str(gpu_num))
device = torch.device(f'cuda:{gpu_num}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(device)  # change allocation of current GPU







#여기까지 def가 아닌 부분이 5번 돌아가는데 이유가 따로 있나??

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"




def test(dir_save,max_only=False, brief = True, reg_input1s = [0.1,1,10], num_brief = 10, num_save=10):
    # dir_save= dir_epoch
    # reg_input1 = i
    # model_name_front = cfg.MODEL.NAME_FRONT
    # model_name_back = cfg.MODEL.NAME_BACK
    # doubleF = doubleF
    # doubleB = doubleB

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
        f'\tavg_speci.mean() \tavg_speci.std() \tavg_asd.mean() \tavg_asd.std() '
        f'\tavg_assd.mean() \tavg_assd.std() \tavg_jc.mean() \tavg_jc.std()')

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

        model =VoxelMorph3dAttn_con1(1, start_channel, ch_magnitude=ch_magnitude, use_gpu=True, img_size=imgshape, is_training= False)
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
                f'\tavg_speci.mean() \tavg_speci.std() \tavg_asd.mean() \tavg_asd.std() '
                f'\tavg_assd.mean() \tavg_assd.std() \tavg_jc.mean() \tavg_jc.std()')

            with torch.no_grad():
                for i, data in enumerate(test_generator):
                    if brief:
                        if i > num_brief:
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


                    if i <= num_save:
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
                    f'\t{avg_speci.mean()} \t{avg_speci.std()} \t{avg_asd.mean()} \t{avg_asd.std()} '
                    f'\t{avg_assd.mean()} \t{avg_assd.std()}\t{avg_jc.mean()} \t{avg_jc.std()}')

                logger_test_total.info(
                    f'\t{epoch} \t{epoch_file} \t{reg_input1}'
                    f'\t{str_mean} \t{str_std} \t'
                    f'\t{dice_nanmean_list.mean()} \t{dice_nanmean_list.std()} \t\t{dice_sum_list.mean()} \t{dice_sum_list.std()}'
                    f'\t\t{avg_dc.mean()} \t{avg_dc.std()} \t{avg_hd.mean()} \t{avg_hd.std()} \t{avg_hd95.mean()}  \t{avg_hd95.std()}'
                    f'\t{avg_prc.mean()} \t{avg_prc.std()} \t{avg_rcl.mean()} \t{avg_rcl.std()} \t{avg_sensi.mean()} \t{avg_sensi.std()}'
                    f'\t{avg_speci.mean()} \t{avg_speci.std()} \t{avg_asd.mean()} \t{avg_asd.std()} '
                    f'\t{avg_assd.mean()} \t{avg_assd.std()} \t{avg_jc.mean()} \t{avg_jc.std()}')

                logger_test.handlers.clear()
    logger_test_total.handlers.clear()



if __name__ == "__main__":
    print(f'main code starts')

    start_test = datetime.now()

    start_channel = 4
    ch_magnitude = 4

    dir_epoch = 'D:\\Yoonguu\\PycharmProjects\\Registration\\CCMorph_YG\\Result\\version3\\LPBA40_small\\Train_con1UnetAttn_loss3_LPBA40_v3\\NCC_con1_0.001KL_1e-06CL\\mid4_4\\gmadmin\\Epoch_300'
    test(dir_epoch,max_only=False, brief = False, reg_input1s = [0.1],num_brief = 10, num_save=10)

    # dir_epoch = 'D:\\Yoonguu\\PycharmProjects\\Registration\\CCMorph_YG\\Result\\version3\\LPBA40_small\\Train_con1UnetAttn_loss3_LPBA40_v3\\NCC_con1_0.001KL_1e-06CL\\mid4_4\\gmadmin\\Epoch_300'
    # test(dir_epoch, max_only=False, brief=False, reg_input1s=[0.1, 1, 10])
    # start_channel = 3
    # ch_magnitude = 2
    # dir_epoch = 'D:\\Yoonguu\\PycharmProjects\\Registration\\CCMorph_YG\\Result\\version3\\LPBA40_small\\Train_con1UnetAttn_loss3_LPBA40_v3\\NCC_con1_0.01KL_0.1CL\\mid3_2\\gmadmin\\Epoch_300'
    # test(dir_epoch, max_only=False, brief=False, reg_input1s=[0.1, 1, 10])
    # end_test = datetime.now()
    # total_test = end_test - start_test
