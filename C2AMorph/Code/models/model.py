import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.autograd import Variable
from torchvision.models import vgg16, resnet50
from torch.distributions import normal
from torch.nn import init
import SimpleITK as sitk
from torch.distributions.normal import Normal

class ConditionalInstanceNorm1(nn.Module):
    def __init__(self, in_channel, latent_dim=64):
        super().__init__()

        self.norm = nn.InstanceNorm3d(in_channel)

        self.style = nn.Linear(latent_dim, in_channel * 2)

        self.style.bias.data[:in_channel] = 1
        self.style.bias.data[in_channel:] = 0

    def forward(self, input, latent_code):
        # style [batch_size, in_channels*2] => [batch_size, in_channels*2, 1, 1, 1]
        # print(f'\t\tinput.shape {input.shape}')
        # print(f'\t\tlatent_code.shape {latent_code.shape}')
        style = self.style(latent_code).unsqueeze(dim=-1).unsqueeze(dim=-1).unsqueeze(dim=-1)
        # print(f'\t\tstyle.shape {style.shape}')
        gamma, beta = style.chunk(2, dim=1)
        # print(f'\t\tgamma.shape {gamma.shape}')
        # print(f'\t\tbeta.shape {beta.shape}')
        out = self.norm(input)
        # print(f'\t\tout.shape {out.shape}')
        # out = input
        out = (1. + gamma) * out + beta

        return out


class Attention_block1D(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        '''
        원래 1x1 Conv가 들어가야되는거임
        '''
        super(Attention_block1D, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi



class PreActBlock_Conditional1(nn.Module):
    """Pre-activation version of the BasicBlock + Conditional instance normalization"""
    expansion = 1

    def __init__(self, in_planes, planes, num_group=4, stride=1, bias=False, num_hype=1, latent_dim=64,
                 mapping_fmaps=64, num_latent=1):
        super(PreActBlock_Conditional1, self).__init__()
        self.ai1 = ConditionalInstanceNorm1(in_planes, latent_dim=latent_dim)
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias)
        self.ai2 = ConditionalInstanceNorm1(in_planes, latent_dim=latent_dim)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=bias)

        self.mapping = nn.Sequential(
            nn.Linear(1, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, mapping_fmaps),
            nn.LeakyReLU(0.2),
            nn.Linear(mapping_fmaps, latent_dim),
            nn.LeakyReLU(0.2)
        )

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=bias)
            )

    def forward(self, x, reg_code):
        # print(f'\tx.shape : {x.shape}')
        # print(f'\treg_code.shape : {reg_code.shape}')
        latent_fea = self.mapping(reg_code)
        # print(f'\tlatent_fea.shape : {latent_fea.shape}')

        out = F.leaky_relu(self.ai1(x, latent_fea), negative_slope=0.2)
        # print(f'\tself.ai1(x, latent_fea).shape : {self.ai1(x, latent_fea).shape}')
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        # print(f'\tshortcut.shape : {shortcut.shape}')
        out = self.conv1(out)
        # print(f'\tout.shape : {out.shape}')
        out = self.conv2(F.leaky_relu(self.ai2(out, latent_fea), negative_slope=0.2))
        # print(f'\tself.ai2(out, latent_fea).shape : {self.ai2(out, latent_fea).shape}')
        # out += shortcut
        out = out + shortcut
        return out

class CCAUnet(nn.Module):
    def __init__(self, in_channel, start_channel, ch_magnitude, bias_opt = True, batchsize=1):

        super(CCAUnet, self).__init__()
        self.in_channel = in_channel
        self.start_channel = start_channel
        self.ch_magnitude = ch_magnitude
        self.batchsize = batchsize
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=self.start_channel * self.ch_magnitude)
        self.conv_maxpool1 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode2 = self.contracting_block(self.start_channel * self.ch_magnitude, self.start_channel * self.ch_magnitude*2)
        self.conv_maxpool2 = torch.nn.MaxPool3d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(self.start_channel * self.ch_magnitude*2, self.start_channel * self.ch_magnitude*4)
        self.conv_maxpool3 = torch.nn.MaxPool3d(kernel_size=2)

        self.avgpool = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
        self.linear_x = torch.nn.Linear(self.start_channel * ch_magnitude * 4, self.start_channel * ch_magnitude *4)
        self.linear_y = torch.nn.Linear(self.start_channel * ch_magnitude * 4, self.start_channel * ch_magnitude * 4)

        self.con_Resblock_group = self.resblock_seq(self.start_channel * ch_magnitude * 8, bias_opt=bias_opt)

        self.attention1 = Attention_block(self.start_channel *self.ch_magnitude * 8, self.start_channel *self.ch_magnitude * 8, self.start_channel *self.ch_magnitude * 16)
        self.attention2 = Attention_block(self.start_channel *self.ch_magnitude * 4, self.start_channel *self.ch_magnitude * 4, self.start_channel *self.ch_magnitude * 8)
        self.attention3 = Attention_block(self.start_channel *self.ch_magnitude*2, self.start_channel *self.ch_magnitude*2 , self.start_channel *self.ch_magnitude * 4)

        # Bottleneck
        mid_channel = self.start_channel * self.ch_magnitude*8

        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1),
            torch.nn.BatchNorm3d(mid_channel * 2),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        )

        # Decode
        self.conv_decode3 = self.expansive_block(self.start_channel * ch_magnitude * 8, self.start_channel * ch_magnitude * 8, self.start_channel * ch_magnitude * 4)
        self.conv_decode2 = self.expansive_block(self.start_channel * ch_magnitude * 4, self.start_channel * ch_magnitude *4, self.start_channel * ch_magnitude * 2)
        self.final_layer = self.final_block(self.start_channel * ch_magnitude * 2, self.start_channel * ch_magnitude * 2, 3)
        self.up3 = torch.nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self._init_weight()

    def resblock_seq(self, in_channels, bias_opt=False, inplaceBool=False):
        layer = nn.ModuleList([
            PreActBlock_Conditional1(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2, inplace=inplaceBool),
            PreActBlock_Conditional1(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2,inplace=inplaceBool),
            PreActBlock_Conditional1(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2, inplace=inplaceBool),
            PreActBlock_Conditional1(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2, inplace=inplaceBool),
            PreActBlock_Conditional1(in_channels, in_channels, bias=bias_opt),
            nn.LeakyReLU(0.2, inplace=inplaceBool)])
        return layer

    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose3d(in_channels=mid_channel, out_channels=out_channels, kernel_size=3, stride=2,
                                     padding=1, output_padding=1),
            torch.nn.BatchNorm3d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1),
            torch.nn.BatchNorm3d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv3d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1),
            torch.nn.BatchNorm3d(out_channels),
            # For Probablisitic Model - Softmax is Better
            torch.nn.Softmax()
            # # For Non-Probablisitic Model - Relu is Better
            # torch.nn.ReLU()
        )
        return block

    def crop_and_concat(self, upsampled, bypass, crop=False):
        """
        This layer crop the layer from contraction block and concat it with expansive block vector
        """
        if crop:
            c = (bypass.size()[2] - upsampled.size()[2]) // 2
            bypass = F.pad(bypass, (-c, -c, -c, -c))
        return torch.cat((upsampled, bypass), 1)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose3d):
                init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.BatchNorm3d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.uniform_(0.0, 1.0)
                m.bias.data.fill_(0)

    def forward(self, x, y, reg_code):
        # Encode 1
        # print(f'x.shape : {x.shape}')
        # print(f'y.shape : {y.shape}')
        encode_block1_x = self.conv_encode1(x)
        # print(f'encode_block1_x.shape : {encode_block1_x.shape}')
        encode_pool1_x = self.conv_maxpool1(encode_block1_x)
        # print(f'encode_pool1_x.shape : {encode_pool1_x.shape}')
        encode_block2_x = self.conv_encode2(encode_pool1_x)
        # print(f'encode_pool1_x.shape : {encode_pool1_x.shape}')
        encode_pool2_x = self.conv_maxpool2(encode_block2_x)
        # print(f'encode_pool2_x.shape : {encode_pool2_x.shape}')
        encode_block3_x = self.conv_encode3(encode_pool2_x)
        # print(f'encode_block3_x.shape : {encode_block3_x.shape}')
        encode_pool3_x = self.conv_maxpool3(encode_block3_x)
        # print(f'encode_pool3_x.shape : {encode_pool3_x.shape}')
        f_x = self.avgpool(encode_pool3_x)
        # print(f'f_x.shape : {f_x.shape}')
        f_x = f_x.squeeze()
        # print(f'f_x.shape : {f_x.shape}')
        print()
        if self.batchsize ==1:
            f_x = f_x.unsqueeze(dim=0)  # [64]를 [batch, 64]로 만들어줘야 되기 때문에 내가 붙인거
            # print(f'\t f_x.shape : {f_x.shape}')
        f_x = self.linear_x(f_x)
        # print(f'f_x.shape : {f_x.shape}')
        f_x = f_x / f_x.norm(dim=-1, keepdim=True)
        # print(f'f_x.shape : {f_x.shape}')

        # Encode 2
        encode_block1_y = self.conv_encode1(y)
        encode_pool1_y = self.conv_maxpool1(encode_block1_y)
        encode_block2_y = self.conv_encode2(encode_pool1_y)
        encode_pool2_y = self.conv_maxpool2(encode_block2_y)
        encode_block3_y = self.conv_encode3(encode_pool2_y)
        encode_pool3_y = self.conv_maxpool3(encode_block3_y)
        f_y = self.avgpool(encode_pool3_y)
        f_y = f_y.squeeze()
        if self.batchsize == 1:
            f_y = f_y.unsqueeze(dim=0)  # [64]를 [batch, 64]로 만들어줘야 되기 때문에 내가 붙인거
        # print(f'f_x.shape : {f_x.shape}')  # [1, 48]
        f_y = self.linear_y(f_y)
        f_y = f_y / f_y.norm(dim=-1, keepdim=True)

        # Bottleneck
        # bottleneck1 = self.bottleneck(torch.cat((encode_pool3_x, encode_pool3_y), 1))

        # print(f'encode_pool3_x.shape : {encode_pool3_x.shape}')
        # print(f'encode_pool3_y.shape : {encode_pool3_y.shape}')
        concat_max = torch.cat([encode_pool3_x, encode_pool3_y], dim=1)
        # print(f'concat_max.shape : {concat_max.shape}')
        for i in range(len(self.con_Resblock_group)):
            if i % 2 == 0:
                res_block = self.con_Resblock_group[i](concat_max, reg_code)
                # print(f'res_block.shape : {res_block.shape}')
            else:
                res_block = self.con_Resblock_group[i](concat_max)
                # print(f'res_block.shape : {res_block.shape}')
        # print(f'res_block.shape : {res_block.shape}')

        res_block_up = self.up3(res_block)
        # print(f'res_block_up.shape : {res_block_up.shape}')
        # print()
        # Decode
        # decode_block3 = self.crop_and_concat(res_block_up, torch.cat((encode_block3_x, encode_block3_y), 1))
        decode_block3 = self.attention1(res_block_up, torch.cat((encode_block3_x, encode_block3_y), 1))
        # print(f'res_block_up.shape : {res_block_up.shape}')
        # print(f'encode_block3_x.shape : {encode_block3_x.shape}')
        # print(f'encode_block3_y.shape : {encode_block3_y.shape}')
        # print(f'decode_block3.shape : {decode_block3.shape}')
        # print()
        cat_layer2 = self.conv_decode3(decode_block3)
        # print(f'cat_layer2.shape : {cat_layer2.shape}')
        # print()

        # decode_block2 = self.crop_and_concat(cat_layer2, torch.cat([encode_block2_x, encode_block2_y], 1))
        decode_block2 = self.attention2(cat_layer2, torch.cat((encode_block2_x, encode_block2_y), 1))
        # print(f'cat_layer2.shape : {cat_layer2.shape}')
        # print(f'encode_block2_x.shape : {encode_block2_x.shape}')
        # print(f'encode_block2_y.shape : {encode_block2_y.shape}')
        # print(f'decode_block2.shape : {decode_block2.shape}')
        # print()
        cat_layer1 = self.conv_decode2(decode_block2)
        # print(f'cat_layer1.shape : {cat_layer1.shape}')
        # print()
        # decode_block1 = self.crop_and_concat(cat_layer1, torch.cat([encode_block1_x, encode_block1_y], 1))
        decode_block1 = self.attention3(cat_layer1, torch.cat((encode_block1_x, encode_block1_y), 1))
        # print(f'cat_layer1.shape : {cat_layer1.shape}')
        # print(f'encode_block1_x.shape : {encode_block1_x.shape}')
        # print(f'encode_block1_y.shape : {encode_block1_y.shape}')
        # print(f'decode_block1.shape : {decode_block1.shape}')
        # print()
        final_layer = self.final_layer(decode_block1)
        # print(f'final_layer.shape : {final_layer.shape}')

        return final_layer, f_x, f_y


class ProbabilisticModel(nn.Module):
    def __init__(self, is_training=True):
        super(ProbabilisticModel, self).__init__()

        self.mean = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)
        self.log_sigma = torch.nn.Conv3d(in_channels=3, out_channels=3, kernel_size=3, padding=1)

        # Manual Initialization
        self.mean.weight.data.normal_(0, 1e-5)
        self.log_sigma.weight.data.normal_(0, 1e-10)
        self.log_sigma.bias.data.fill_(-10.)


        self.is_training=is_training

    def forward(self, final_layer):
        flow_mean = self.mean(final_layer)
        flow_log_sigma = self.log_sigma(final_layer)
        noise = torch.randn_like(flow_mean).cuda()

        if self.is_training:
            flow = flow_mean + flow_log_sigma * noise
        else:
            flow = flow_mean + flow_log_sigma # No noise at testing time

        return flow, flow_mean, flow_log_sigma


class C2AMorph(nn.Module):
    '''
    This VoxelMorph3d_con1 is for the Unet based Voxelmorph using KL divergence loss, Contrastive loss
    in this case, we can use
    '''
    def __init__(self, in_channel=1, start_channel=3, ch_magnitude=2, bias_opt=True, use_gpu=False, is_training=True, img_size=(144, 192, 144), batchsize=1):

        super(C2AMorph, self).__init__()
        self.start_channel = start_channel
        self.ch_magnitude= ch_magnitude
        self.bias_opt = bias_opt
        self.batchsize = batchsize


        self.unet = CCAUnet(in_channel, start_channel, ch_magnitude, bias_opt = bias_opt, batchsize= self.batchsize)
        self.probabilistic_model = ProbabilisticModel(is_training=is_training)
        self.spatial_transform = SpatialTransformer(img_size)

        if use_gpu:
            self.unet = self.unet.cuda()
            self.probabilistic_model = self.probabilistic_model.cuda()
            self.spatial_transform = self.spatial_transform.cuda()

    def forward(self, moving_image, fixed_image, reg_code):
        # print(f'moving_image.shape : {moving_image.shape}')
        # print(f'fixed_image.shape : {fixed_image.shape}')


        flow, f_x, f_y = self.unet(moving_image, fixed_image, reg_code)
        # print(f'flow.shape : {flow.shape}')
        # print(f'f_x.shape : {f_x.shape}')
        # print(f'f_y.shape : {f_y.shape}')

        deformation_matrix, flow_mean, flow_log_sigma = self.probabilistic_model(flow)
        # print(f'deformation_matrix.shape : {deformation_matrix.shape}')
        # print(f'flow_mean.shape : {flow_mean.shape}')
        # print(f'flow_log_sigma.shape : {flow_log_sigma.shape}')
        warped_image = self.spatial_transform(moving_image, deformation_matrix)
        # print(f'warped_image.shape : {warped_image.shape}')
        # warped_image_atlas = self.spatial_transform(moving_atlas, deformation_matrix, mode="nearest")

        return warped_image, deformation_matrix, flow_mean, flow_log_sigma, f_x, f_y
class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv3d(F_g, F_int, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv3d(F_l, F_int, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.BatchNorm3d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv3d(F_int, 1, kernel_size=(1, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0), bias=True),
            nn.BatchNorm3d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # print(f'\tg.shape : {g.shape}')
        # print(f'\tx.shape : {x.shape}')
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # print(f'\tg1.shape : {g1.shape}')
        # print(f'\tx1.shape : {x1.shape}')
        psi = self.relu(g1 + x1)
        # print(f'\tpsi.shape : {psi.shape}')
        psi = self.psi(psi)
        # print(f'\tpsi.shape : {psi.shape}')
        # a= x * psi
        # print(f'\tpsi * x.shape : {a.shape}')
        return x * psi


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output from the UNet to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

    def forward(self, src, flow, mode='bilinear'):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return F.grid_sample(src, new_locs, mode=mode)



if __name__ == "__main__":
    print(f'model_Unet_conditional1 started in main')
    ori_imgshape = (160, 192, 224)
    imgshape = (128, 128, 128)
    imgshape_4 = (128 // 4, 128 // 4, 128 // 4)
    imgshape_2 = (128 // 2, 128 // 2, 128 // 2)
    start_channel = 6


    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    input1 = torch.randn(4, 1, 128, 128, 128).cuda()
    input2 = torch.randn(4, 1, 128, 128, 128).cuda()
    reg_code1 = torch.rand(1, dtype=input1.dtype, device=input1.device).unsqueeze(dim=0)
    reg_code2 = torch.rand(1, dtype=input1.dtype, device=input1.device).unsqueeze(dim=0)
    reg_code3 = torch.rand(1, dtype=input1.dtype, device=input1.device).unsqueeze(dim=0)

    X = input1.to(device).float()
    Y = input1.to(device).float()
    print("X.shape : {}".format(X.shape))
    print("Y.shape : {}".format(Y.shape))

    model_lvl1 = CCAUnet(1, 3, 2, batchsize=4).to(device)
    # output_disp_e0, warpped_inputx_lvl1_out, down_y, output_disp_e0_v, e0 = model_lvl1(X, Y)
    model_lvl1(X, Y, reg_code1)
    print('\n\n')