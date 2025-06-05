import numpy as np
import torch.utils.data as Data

import torch
import itertools
import os
from Functions import load_4D, imgnorm, load_3D
from Data_augmentation import get_transform, get_transform_test


class Dataset(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, iterations, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.iterations = iterations

    def __len__(self):
        'Denotes the total number of samples'
        return self.iterations

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        index_pair = np.random.permutation(len(self.names))[0:2]
        img_A = load_4D(self.names[index_pair[0]]) #type numpy
        img_B = load_4D(self.names[index_pair[1]])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()


class Dataset_epoch(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, norm=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.index_pair = list(itertools.permutations(names, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.index_pair[step][0])
        img_B = load_4D(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float()



class Dataset_epoch_validation(Data.Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, imgs, labels, norm=False):
        'Initialization'
        super(Dataset_epoch_validation, self).__init__()



        self.imgs = imgs
        self.labels = labels
        self.norm = norm
        self.imgs_pair = list(itertools.permutations(imgs, 2))
        self.labels_pair = list(itertools.permutations(labels, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.imgs_pair)

    def __getitem__(self, step):
        # print(f'Dataset_epoch_validation --step : {step}')
        'Generates one sample of data'
        # Select sample
        img_A = load_4D(self.imgs_pair[step][0])
        img_B = load_4D(self.imgs_pair[step][1])

        label_A = load_4D(self.labels_pair[step][0])
        label_B = load_4D(self.labels_pair[step][1])

        # print(self.index_pair[step][0])
        # print(self.index_pair[step][1])

        if self.norm:
            return torch.from_numpy(imgnorm(img_A)).float(), torch.from_numpy(imgnorm(img_B)).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()
        else:
            return torch.from_numpy(img_A).float(), torch.from_numpy(img_B).float(), torch.from_numpy(label_A).float(), torch.from_numpy(label_B).float()

class Dataset_Mindboggle(Data.Dataset):
    '''
    Data loader for Mindboggle
    #isTest == False -> for Training
    #isTest == True -> for test
    '''

    def __init__(self, names, norm=False, img_size=(72, 96, 72), isTest=False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.img_size = img_size

        if isTest:
            self.transform = get_transform_test(img_size=self.img_size)
            self.constrain = list(range(1, 43, 1))
        else:
            self.transform = get_transform(img_size=self.img_size)
            self.constrain = list(range(43, 63, 1))

        self.img_paths = []
        for file in self.names:
            file_name = os.path.basename(os.path.normpath(file))
            if (file_name.endswith(".nii.gz") and file_name.startswith("brain")):
                for c in self.constrain:
                    if "brain_{0:02d}".format(c) in str(file_name) :
                        break
                else:
                    self.img_paths.append(file)

        self.index_pair = list(itertools.permutations(self.img_paths, 2))

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        self.img_mov_path = self.index_pair[step][0]
        self.img_fix_path = self.index_pair[step][1]

        img_mov = load_3D(self.index_pair[step][0])
        img_fix = load_3D(self.index_pair[step][1])

        self.img_mov_atlas_path = self.img_mov_path.replace("brain", f"atlas")
        self.img_fix_atlas_path = self.img_fix_path.replace("brain", f"atlas")

        img_mov_atlas = load_3D(self.img_mov_atlas_path)
        img_fix_atlas = load_3D(self.img_fix_atlas_path)

        # print(f"img_mov_path : {self.img_mov_path}")
        # print(f"img_mov_atlas_path : {self.img_mov_atlas_path}")
        # print(f"img_fix_path : {self.img_fix_path}")
        # print(f"img_fix_atlas_path : {self.img_fix_atlas_path}")
        # print()
        fixed_img_pytorch, moving_img_pytorch, img_fix_atlas_pytorch, img_mov_atlas_pytorch = self.transform(
            [img_fix, img_mov, img_fix_atlas, img_mov_atlas])

        return moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch


class Dataset_OASIS(Data.Dataset):
    '''
    Data loader for OASIS dataset
    #set_name을 바꾸면 Train, Test, Val로 처리됨

    '''
    def __init__(self, names, norm=False, img_size=(72, 96, 72), num_seg=35, isTest = False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.img_size = img_size
        self.num_seg = num_seg
        if isTest:
            self.transform = get_transform_test(img_size=self.img_size)
        else:
            self.transform = get_transform(img_size=self.img_size)
        self.index_pair= list(itertools.permutations(self.names, 2))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        self.img_mov_path = self.index_pair[step][0]
        self.img_fix_path = self.index_pair[step][1]

        img_mov = load_3D(self.index_pair[step][0])
        img_fix = load_3D(self.index_pair[step][1])

        self.img_mov_atlas_path = self.img_mov_path.replace(
                "aligned_norm",f"aligned_seg{self.num_seg}")
        self.img_fix_atlas_path = self.img_fix_path.replace(
            "aligned_norm", f"aligned_seg{self.num_seg}")



        img_mov_atlas = load_3D(self.img_mov_atlas_path)
        img_fix_atlas = load_3D(self.img_fix_atlas_path)



        # print(f"img_mov_path : {self.img_mov_path}")
        # print(f"img_mov_atlas_path : {self.img_mov_atlas_path}")
        # print(f"img_fix_path : {self.img_fix_path}")
        # print(f"img_fix_atlas_path : {self.img_fix_atlas_path}")
        # print()
        fixed_img_pytorch, moving_img_pytorch, img_fix_atlas_pytorch, img_mov_atlas_pytorch   = self.transform(
            [img_fix, img_mov, img_fix_atlas, img_mov_atlas])

        return moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch




class Dataset_OASIS_split(Data.Dataset):
    '''
    Data loader for OASIS dataset
    #set_name을 바꾸면 Train, Test, Val로 처리됨

    '''
    def __init__(self, names, norm=False, img_size=(72, 96, 72), tr_ratio = 0.8, set_name ='Train', num_seg=35):
        'Initialization'
        self.names = names
        self.norm = norm
        self.img_size = img_size
        self.tr_ratio = tr_ratio
        self.val_ratio = (1 - self.tr_ratio) / 2
        self.test_ratio = (1 - self.tr_ratio) / 2
        self.set_name = set_name
        self.num_seg = num_seg

        self.transform = get_transform(img_size=self.img_size)
        if self.set_name=='Train':
            self.filenames = names[0: int(len(names)*self.tr_ratio)]

        elif self.set_name == 'Val':
            self.filenames = names[int(len(names) * self.tr_ratio): int(len(names)*(self.tr_ratio+self.val_ratio))]

        elif self.set_name=='Test':
            self.filenames = names[int(len(names)*(self.tr_ratio+self.val_ratio)): len(names)]
        # print(f'len(self.filenames) : {len(self.filenames)}')
        self.index_pair= list(itertools.permutations(self.filenames, 2))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        self.img_mov_path = self.index_pair[step][0]
        self.img_fix_path = self.index_pair[step][1]

        img_mov = load_3D(self.index_pair[step][0])
        img_fix = load_3D(self.index_pair[step][1])

        self.img_mov_atlas_path = self.img_mov_path.replace(
                "aligned_norm",f"aligned_seg{self.num_seg}")
        self.img_fix_atlas_path = self.img_fix_path.replace(
            "aligned_norm", f"aligned_seg{self.num_seg}")



        img_mov_atlas = load_3D(self.img_mov_atlas_path)
        img_fix_atlas = load_3D(self.img_fix_atlas_path)



        # print(f"img_mov_path : {self.img_mov_path}")
        # print(f"img_mov_atlas_path : {self.img_mov_atlas_path}")
        # print(f"img_fix_path : {self.img_fix_path}")
        # print(f"img_fix_atlas_path : {self.img_fix_atlas_path}")
        # print()
        fixed_img_pytorch, moving_img_pytorch, img_fix_atlas_pytorch, img_mov_atlas_pytorch   = self.transform(
            [img_fix, img_mov, img_fix_atlas, img_mov_atlas])

        return moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch



class Dataset_LPBA40(Data.Dataset):
    'Characterizes a dataset for PyTorch - for training'

    def __init__(self, names, img_size=(72, 96, 72), norm=False):
        '''
        Initialization
        :param names: file name list
        :param norm: even tho there is varaible norm,
                    but we only use the normalized LPBA40 files through preprocessing code.
                    so we don't use this variable
        '''
        self.names = names
        self.norm = norm
        self.img_size = img_size

    def initialize(self):
        '''
        to make the list of moving_paths(moving), dictionary of moving_fixed(fixed)
            dictionary of moving_fixed(fixed) --> {moving_path(key) : moving_fix(value)}
`
        :param is_training: bool--> to make train dataloader, validation dataloader
        :return:
        '''



        self.constrain = list(range(31, 41, 1))
        self.transform = get_transform(img_size=self.img_size)


        self.moving_paths = []
        for file in self.names:
            file_name = os.path.basename(os.path.normpath(file))
            if (file_name.endswith(".hdr") or file_name.endswith(".nii")) and file_name.startswith("l"):
                for c in self.constrain:
                    if "l{}_".format(str(c)) in str(file_name) or "_l{}.".format(str(c)) in str(file_name):
                        break
                else:
                    self.moving_paths.append(file)

        self.fixed_paths = {}
        self.moving_atlas_paths = {}
        self.fixed_atlas_paths = {}
        for file_moving in self.moving_paths:
            path_parents = os.path.abspath(os.path.join(file_moving, os.pardir))
            file_name_moving = os.path.basename(os.path.normpath(file_moving))
            fixed_name = file_name_moving.split(".")[-2].split("_")[-1]
            fixed_suffix = file_name_moving.split(".")[-1]

            if not (fixed_suffix == "hdr" or fixed_suffix == "nii"):
                raise Exception("Suffix not hdr or nii.")
            self.fixed_paths[file_moving] = os.path.join(path_parents,
                                                          "{}_to_{}.{}".format(str(fixed_name), str(fixed_name),
                                                                               str(fixed_suffix)))
            self.moving_atlas_paths[file_moving] = file_moving.replace(
                "LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten",
                "LPBA40_rigidly_registered_label_pairs_small_GoodLabel")
            self.fixed_atlas_paths[file_moving] = self.fixed_paths[file_moving].replace(
                "LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten",
                "LPBA40_rigidly_registered_label_pairs_small_GoodLabel")

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.fixed_paths)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        dir_moving = self.moving_paths[step]
        dir_fix = self.fixed_paths[self.moving_paths[step]]

        img_mov = load_3D(dir_moving) # type numpy
        img_fix = load_3D(dir_fix) # type numpy

        dir_moving_atlas = self.moving_atlas_paths[self.moving_paths[step]]
        dir_fixed_atlas = self.fixed_atlas_paths[self.moving_paths[step]]

        img_mov_atlas = load_3D(dir_moving_atlas)  # type numpy
        img_fix_atlas = load_3D(dir_fixed_atlas)  # type numpy

        if self.norm:
            img_mov = imgnorm(img_mov)
            img_fix = imgnorm(img_fix)

        # img_mov_torch = torch.from_numpy(img_mov).float()
        # img_fix_torch = torch.from_numpy(img_fix).float()
        # print(f'img_mov_torch.shape : {img_mov_torch.shape}')
        # print(f'img_fix_torch.shape : {img_fix_torch.shape}')
        # img_mov_atlas_torch = torch.from_numpy(img_mov_atlas).float()
        # img_fix_atlas_torch = torch.from_numpy(img_fix_atlas).float()

        #self.transform_train is change based on is_training variable
        moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch = self.transform(
            [img_mov, img_fix, img_mov_atlas, img_fix_atlas])
        # print(f'type(img_mov) : {type(img_mov)}') #class 'numpy.memmap
        # print(f'img_mov.shape : {img_mov.shape}') #(80, 106, 80)
        # print(f'type(moving_img_pytorch) : {type(moving_img_pytorch)}') #class 'torch.Tensor'
        # print(f'moving_img_pytorch.shape : {moving_img_pytorch.shape}') #torch.Size([1, 72, 96, 72])

        return moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch


class Dataset_LPBA40_ValTest(Data.Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, names, img_size=(72, 96, 72), norm=False):
        '''
        Initialization
        :param names: file name list
        :param norm: even tho there is varaible norm,
                    but we only use the normalized LPBA40 files through preprocessing code.
                    so we don't use this variable
        '''
        self.names = names
        self.norm = norm
        self.img_size = img_size

    def initialize(self, is_valid):
        '''
        to make the list of moving_paths(moving), dictionary of moving_fixed(fixed)
            dictionary of moving_fixed(fixed) --> {moving_path(key) : moving_fix(value)}
`
        :param is_training: bool--> to make train dataloader, validation dataloader
        :return:
        '''
        self.is_valid = is_valid

        self.transform = get_transform_test(img_size=self.img_size)
        if self.is_valid =='test':
            self.constrain = list(range(1, 31, 1))


            self.moving_paths = []
            for file in self.names:
                file_name = os.path.basename(os.path.normpath(file))
                if (file_name.endswith(".hdr") or file_name.endswith(".nii")) and file_name.startswith("l"):
                    for c in self.constrain:
                        if "l{}_".format(str(c)) in str(file_name) or "_l{}.".format(str(c)) in str(file_name):
                            break
                    else:
                        self.moving_paths.append(file)

        elif self.is_valid=='val':
            constrain1 = list(range(1, 31, 1))
            constrain2 = list(range(31, 41, 1))

            self.moving_paths = []
            for file in self.names:
                file_name = os.path.basename(os.path.normpath(file))
                if (file_name.endswith(".hdr") or file_name.endswith(".nii")) and file_name.startswith("l"):
                    for c1 in constrain1:
                        for c2 in constrain2:
                            if "l{}_".format(str(c1)) in str(file_name) and "_l{}.".format(str(c2)) in str(file_name):
                                self.moving_paths.append(file)
                            if "l{}_".format(str(c2)) in str(file_name) and "_l{}.".format(str(c1)) in str(file_name):
                                self.moving_paths.append(file)

        # self.moving_fixed = {}
        # self.moving_atlases = {}
        # self.fixed_atlases = {}


        self.fixed_paths = {}
        self.moving_atlas_paths = {}
        self.fixed_atlas_paths = {}
        for file_moving in self.moving_paths:
            path_parents = os.path.abspath(os.path.join(file_moving, os.pardir))
            file_name_moving = os.path.basename(os.path.normpath(file_moving))
            fixed_name = file_name_moving.split(".")[-2].split("_")[-1]
            fixed_suffix = file_name_moving.split(".")[-1]

            if not (fixed_suffix == "hdr" or fixed_suffix == "nii"):
                raise Exception("Suffix not hdr or nii.")
            self.fixed_paths[file_moving] = os.path.join(path_parents,
                                                          "{}_to_{}.{}".format(str(fixed_name), str(fixed_name),
                                                                               str(fixed_suffix)))
            self.moving_atlas_paths[file_moving] = file_moving.replace(
                "LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten",
                "LPBA40_rigidly_registered_label_pairs_small_GoodLabel")
            self.fixed_atlas_paths[file_moving] = self.fixed_paths[file_moving].replace(
                "LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten",
                "LPBA40_rigidly_registered_label_pairs_small_GoodLabel")



        # print(f'len(fixed_paths) : {len(self.fixed_paths)}')

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.fixed_paths)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        dir_moving = self.moving_paths[step]
        dir_fix = self.fixed_paths[self.moving_paths[step]]

        img_mov = load_3D(dir_moving) # type numpy
        img_fix = load_3D(dir_fix) # type numpy

        dir_moving_atlas = self.moving_atlas_paths[self.moving_paths[step]]
        dir_fixed_atlas = self.fixed_atlas_paths[self.moving_paths[step]]

        img_mov_atlas = load_3D(dir_moving_atlas)  # type numpy
        img_fix_atlas = load_3D(dir_fixed_atlas)  # type numpy

        if self.norm:
            img_mov = imgnorm(img_mov)
            img_fix = imgnorm(img_fix)

        # img_mov_torch = torch.from_numpy(img_mov).float()
        # img_fix_torch = torch.from_numpy(img_fix).float()
        #
        # img_mov_atlas_torch = torch.from_numpy(img_mov_atlas).float()
        # img_fix_atlas_torch = torch.from_numpy(img_fix_atlas).float()

        #self.transform_train is change based on is_training variable
        img_mov_pytorch, img_fix_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch = self.transform(
            [img_mov, img_fix, img_mov_atlas, img_fix_atlas])

        return img_mov_pytorch, img_fix_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch

# from Data_augmentation import get_transform, get_transform_test


class Predict_dataset(Data.Dataset):
    def __init__(self, fixed_list, move_list, fixed_label_list, move_label_list, norm=False):
        super(Predict_dataset, self).__init__()
        self.fixed_list = fixed_list
        self.move_list = move_list
        self.fixed_label_list = fixed_label_list
        self.move_label_list = move_label_list
        self.norm = norm

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.move_list)

    def __getitem__(self, index):
        fixed_img = load_4D(self.fixed_list)
        moved_img = load_4D(self.move_list[index])
        fixed_label = load_4D(self.fixed_label_list)
        moved_label = load_4D(self.move_label_list[index])

        if self.norm:
            fixed_img = imgnorm(fixed_img)
            moved_img = imgnorm(moved_img)

        fixed_img = torch.from_numpy(fixed_img)
        moved_img = torch.from_numpy(moved_img)
        fixed_label = torch.from_numpy(fixed_label)
        moved_label = torch.from_numpy(moved_label)

        if self.norm:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output
        else:
            output = {'fixed': fixed_img.float(), 'move': moved_img.float(),
                      'fixed_label': fixed_label.float(), 'move_label': moved_label.float(), 'index': index}
            return output



class Dataset_LungCT(Data.Dataset):
    '''
    Data loader for OASIS dataset
    #set_name을 바꾸면 Train, Test, Val로 처리됨

    '''
    def __init__(self, names, norm=False, img_size=(112, 96, 112), num_seg=1, isTest = False):
        'Initialization'
        self.names = names
        self.norm = norm
        self.img_size = img_size
        self.num_seg = num_seg
        if isTest:
            self.transform = get_transform_test(img_size=self.img_size)
        else:
            self.transform = get_transform(img_size=self.img_size)
        self.index_pair= list(itertools.permutations(self.names, 2))


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.index_pair)

    def __getitem__(self, step):
        'Generates one sample of data'
        # Select sample
        self.img_mov_path = self.index_pair[step][0]
        self.img_fix_path = self.index_pair[step][1]

        img_mov = load_3D(self.index_pair[step][0])
        img_fix = load_3D(self.index_pair[step][1])

        self.img_mov_atlas_path = self.img_mov_path.replace(
                "imagesTr","masksTr")
        self.img_fix_atlas_path = self.img_fix_path.replace(
            "imagesTr","masksTr")



        img_mov_atlas = load_3D(self.img_mov_atlas_path)
        img_fix_atlas = load_3D(self.img_fix_atlas_path)
        # print(f'self.img_mov_path : {self.img_mov_path}')
        # print(f'self.img_fix_path : {self.img_fix_path}')
        #
        # print(f'self.img_mov_atlas_path : {self.img_mov_atlas_path}')
        # print(f'self.img_fix_atlas_path : {self.img_fix_atlas_path}\n')


        # print(f"img_mov_path : {self.img_mov_path}")
        # print(f"img_mov_atlas_path : {self.img_mov_atlas_path}")
        # print(f"img_fix_path : {self.img_fix_path}")
        # print(f"img_fix_atlas_path : {self.img_fix_atlas_path}")
        # print()
        fixed_img_pytorch, moving_img_pytorch, img_fix_atlas_pytorch, img_mov_atlas_pytorch   = self.transform(
            [img_fix, img_mov, img_fix_atlas, img_mov_atlas])

        return moving_img_pytorch, fixed_img_pytorch, img_mov_atlas_pytorch, img_fix_atlas_pytorch
