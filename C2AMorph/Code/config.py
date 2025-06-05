from yacs.config import CfgNode as CN
import platform

_C = CN()
_C.PLATFORM = CN()
_C.DATASET = CN()

_C.DATASET.DATATYPE = 'LPBA40_small'
#run in python console or os.path.basename(__file__)

if "Win" in platform.system():
    _C.PLATFORM.isWin = True
    _C.PLATFORM.server = 'Computer'
else:
    _C.PLATFORM.isWin = False
    _C.PLATFORM.server = 'Server'
    # print("_C.PLATFORM.isWin", _C.PLATFORM.isWin)






_C.DATASET.NUM_EPOCHS = 5 #for LPBA40

# _C.DATASET.TR_VAL_RATIO = 0.9
_C.DATALOADER = CN()
_C.DATALOADER.BATCH_SIZE = 1
_C.DATALOADER.NUM_WORKERS = 6
_C.DATALOADER.NORM = False #on LPBA40

_C.MODEL = CN()
_C.MODEL.DROPOUT = 0.2
_C.MODEL.GPU_NUM = 0
_C.MODEL.PARALLEL_GPUS = None

_C.MODEL.NAME_FRONT = 'Unet'
_C.MODEL.NAME_BACK = 'both'
# 'unet_diff' 'both'

_C.MODEL.FDoubleConv= True
_C.MODEL.BDoubleConv= True
_C.MODEL.BestModelCheckRatio= 0.98

_C.MODEL.Start_Channel=4
_C.MODEL.Ch_magnitude=4

if _C.DATASET.DATATYPE == 'LPBA40_small':
    _C.DATASET.DATA_PATH = 'E:\\Brain\\LPBA40\\from_CLMorph'
    _C.DATASET.DATA_PATH_IMGS = _C.DATASET.DATA_PATH + '/LPBA40_rigidly_registered_pairs_histogram_standardization_small_Whiten/l*_to_l*.nii'
    _C.DATASET.DATA_PATH_LABELS = _C.DATASET.DATA_PATH + '/LPBA40_rigidly_registered_label_pairs_small_GoodLabel/l*_to_l*.nii'
    _C.DATASET.PAIR = False


_C.SOLVER = CN()
_C.SOLVER.LEARNING_RATE = 1e-4

_C.SOLVER.CHECKPOINT = 1000
_C.SOLVER.LOAD_MODEL = False
_C.SOLVER.RANGE_FLOW = 0.7
_C.SOLVER.NCC_EPS = 1e-8 #1e-8



#'MSE', 'NCC'
_C.SOLVER.LOSS1 = 'NCC' #similarity loss

# 'REG', 'KL', 'REG', 'Jacobian', 'CL'
_C.SOLVER.LOSS2 ='KL'#' #Regularization
_C.SOLVER.LOSS3 = 'CL'
_C.SOLVER.LOSS4 = 'REG'
_C.SOLVER.LOSS5 = 'Jacobian'


_C.SOLVER.HYP_KL = 0.01 #hyp_KL
_C.SOLVER.MAX_KL = 0.01#10 #max_KL

_C.SOLVER.HYP_SMOOTH = 0#3.5 #hyp_smooth
_C.SOLVER.MAX_SMOOTH = 0#100 #max_smooth

_C.SOLVER.HYP_CL = 0.1 #hyp_contra
_C.SOLVER.MAX_CL = 1000 #max_contra

_C.SOLVER.HYP_ANTIFOLD = 0.01 #hyp_antifold
_C.SOLVER.MAX_ANTIFOLD = 10 #max_antifold







