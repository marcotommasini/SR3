from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import sys


class SiLu(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class UpSample(nn.Module):      #upsample the block by 2 and maintain the same number of channels
    def __init__(self, input_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(input_channels, input_channels, 3, padding=1)
    def forward(self, x_input):
        return self.conv(self.up(x_input))


class DownSample(nn.Module):    #Downsample the block by 2 and maintain the same number of channels
    def __init__(self, input_channels):
        super().__init__()
        self.conv = nn.Conv2d(input_channels, input_channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x_input):
        return self.conv(x_input)
        

class ResnetBlock(nn.Module):
    pass

class smallBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, dim),
            SiLu(),
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )
    def forward(self, x):
        return self.block(x)

class SelfAttention(nn.Module):
    pass


class UNET_SR3(nn.Module):
    def __init__(self, in_channels, out_channels, final_image_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_image_size = final_image_size
        
        self.inner_channel = 32 #This is the ammount of channels after only the first conv2d block
        self.number_res_packs = 3   #Number of times the the agreggation that involver the residual blocks will be called
        self.channels_multipliers = [1, 2, 4, 8, 8]

        
        num_multipliers = len(self.channels_multipliers)
        current_channel_input = self.inner_channel
        featured_channels = [current_channel_input]
        current_resolution = final_image_size
        resolution_to_use_attention = 8
        

        #I need to define the time emmeding trasnform and how it affects the network
        
        #I will add each block to a list and then use the module function to make it into a sequential
        #Downs blocks

        DOWNS = [nn.Conv2d(self.in_channels, self.inner_channel, kernel_size=3, padding=1)]
        flag_last = False
        for index in range(num_multipliers):

            flag_last = (index == num_multipliers - 1)  #Checks if this is the final iteration
            current_channel_output = self.inner_channel * self.channels_multipliers[index]
            use_attention = (current_resolution == resolution_to_use_attention)

            for j in range(self.number_res_packs):
                DOWNS.append([ResnetBlock(current_channel_input, current_channel_output, noise_level_emb_dim=noise_level_channel,\
                     norm_groups=norm_groups, dropout=dropout, with_attn=use_attention)])
                featured_channels.append()  #This serves so that in the upsample block they know how to concatenate the skip connections
                current_channel_input = current_channel_output
                
                if flag_last == False:
                    DOWNS.append(DownSample(current_channel_input))
                    featured_channels.append(current_channel_input)
                    current_resolution = current_resolution/2

        self.downs = nn.ModuleList(DOWNS)

        self.middle = nn.ModuleList([ResnetBlock(current_channel_input, current_channel_input, noise_level_emb_dim=noise_level_channel, \
                                        norm_groups=norm_groups, dropout=dropout, with_attn=True),

                                    ResnetBlock(current_channel_input, current_channel_input, noise_level_emb_dim=noise_level_channel, \
                                        norm_groups=norm_groups, dropout=dropout, with_attn=False)])
        
        UPS = []
        for index in reversed(range(num_multipliers)):
            flag_last = (index == 0)
            use_attention = (current_resolution == resolution_to_use_attention)
            current_channel_output = self.inner_channel * self.channels_multipliers[index]
            for j in range(0, self.number_res_packs +1):

                UPS.append(ResnetBlock(current_channel_input+featured_channels.pop(), current_channel_output, \
                    noise_level_emb_dim=noise_level_channel, norm_groups=norm_groups, dropout=dropout, with_attn=use_attention))
                current_channel_input = current_channel_output
                if flag_last == False:
                    UPS.append(UpSample(current_channel_input))
                    current_resolution = current_resolution * 2
        self.ups = nn.ModuleList(UPS)

        self.output_conv = smallBlock(current_channel_input, out_channels, groups=norm_groups)

    def forward(self, x, time):

        time = 0  #We need to define the network to adapt the size of the time embeding to the number of channels

        skip_features = []
        for layer in self.downs:
            if isinstance(layer, ResnetBlock):
                x = layer(x, time)
            else:
                x = layer(x)
            skip_features.append(x)  #This will save all the intermediate values so they can be passed to the upsample
        
        for layer in self.middle:
            if isinstance(layer, ResnetBlock):
                x = layer(x, time)
            else:
                x = layer(x)
            skip_features.append(x)  #This will save all the intermediate values so they can be passed to the upsample
        
        for layer in self.ups:
            if isinstance(layer, ResnetBlock):
                x = layer(torch.cat((x, skip_features.pop()), dim=1), time)
            else:
                x = layer(x)

        return self.output_conv(x)
        
