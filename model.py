from turtle import forward
import torch
from torch import nn
import torch.nn.functional as F
import sys


class TimeDimensionMatching(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.emb_layer = nn.Sequential(nn.SiLU(),
            nn.Linear(in_channels, out_channels)) #this sequential matches the embedding dimension with the number of channels

    def forward(self, x, time_embedded):
        batch = x.shape[0]
        out = self.emb_layer(time_embedded)[:, :, None, None].repeat(1,1,x.size(-2), x.size(-1))
        out_final = x + self.emb_layer(time_embedded).view(batch, -1, 1, 1)
        return out_final



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
    def __init__(self, dim, dim_out, time_embedding_dimension, dropout = 0, norm_groups = 32, use_attention = False):
        super().__init__()
        
        self.addTime = TimeDimensionMatching(time_embedding_dimension, dim_out)
        self.use_attention = use_attention
        self.block1 = smallBlock(dim, dim_out, norm_groups)
        self.block2 = smallBlock(dim_out, dim_out, norm_groups, dropout = dropout)
        self.final_conv = nn.Conv2d(dim, dim_out, 1)

        if use_attention:
            self.att_block = SelfAttention(dim_out)

    def forward(self, x_input, time_embedded):
        x1 = self.block1(x_input)
        x_emb = self.addTime(x1, time_embedded)
        x2 = self.block2(x_emb)
        x_output = x2 + self.final_conv(x_input)

        if not self.use_attention:
            return x_output
        else:
            return self.att_block(x_output)



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



class SelfAttention(nn.Module):     #Standard attention block
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.mha = nn.MultiheadAttention(in_channels, 4, batch_first=True)
        self.layer_norm = nn.LayerNorm([in_channels])

        self.feed_forward = nn.Sequential(
            nn.LayerNorm([in_channels]),
            nn.Linear(in_channels, in_channels),
            nn.GELU(),
            nn.Linear(in_channels, in_channels),)
    def forward(self, x):
        batch, channels, size_y, size_x = x.size()
        x = x.view(-1, self.in_channels, int(size_y**2)).swapaxes(1,2)

        norm_x = self.layer_norm(x)
        attention_value, _ = self.mha(norm_x, norm_x, norm_x)
        attention_value = attention_value + x
        attention_value = self.feed_forward(attention_value) + attention_value
        output = attention_value.swapaxes(2, 1).view(-1, self.in_channels, size_y, size_x)
        return output


class UNET_SR3(nn.Module):
    def __init__(self, in_channels = 6, \
                out_channels = 3, \
                channel_multipliers = [1, 2, 4, 8, 8], \
                final_image_size = 128, \
                number_res_blocks = 3, \
                time_embedding_dimension = 256):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.final_image_size = final_image_size
        
        self.inner_channel = 32 #This is the ammount of channels after only the first conv2d block
        self.number_res_packs = number_res_blocks   #Number of times the the agreggation that involver the residual blocks will be called
        self.channels_multipliers = channel_multipliers
        self.time_embedding_dimension = time_embedding_dimension

        
        num_multipliers = len(self.channels_multipliers)
        current_channel_input = self.inner_channel
        featured_channels = [current_channel_input]
        current_resolution = final_image_size
        resolution_to_use_attention = 8
        dropout = 0
        

        #I will add each block to a list and then use the module function to make it into a sequential

        #Downs blocks

        DOWNS = [nn.Conv2d(self.in_channels, self.inner_channel, kernel_size=3, padding=1)]
        flag_last = False
        for index in range(num_multipliers):
            flag_last = (index == num_multipliers - 1)  #Checks if this is the final iteration
            current_channel_output = self.inner_channel * self.channels_multipliers[index]
            use_attention = (current_resolution == resolution_to_use_attention)

            for j in range(self.number_res_packs):
                DOWNS.append(ResnetBlock(current_channel_input, current_channel_output,\
                                time_embedding_dimension = self.time_embedding_dimension, \
                                dropout=dropout, use_attention=use_attention))
                                
                featured_channels.append(current_channel_output)  #This serves so that in the upsample block they know how to concatenate the skip connections
                current_channel_input = current_channel_output
                
            if not flag_last:   
                DOWNS.append(DownSample(current_channel_input))
                featured_channels.append(current_channel_input)
                current_resolution = current_resolution/2
        self.downs = nn.ModuleList(DOWNS)   #Creating the sequential with the downblocks
    	
        #Middle blocks
        self.middle = nn.ModuleList([ResnetBlock(current_channel_input, current_channel_input,\
                                     time_embedding_dimension = self.time_embedding_dimension, dropout=dropout, use_attention=True),

                                    ResnetBlock(current_channel_input, current_channel_input,\
                                        time_embedding_dimension = self.time_embedding_dimension, dropout=dropout, use_attention=False)])
        
        #Up blocks
        UPS = []
        for index in reversed(range(num_multipliers)):
            flag_last = (index == 0)
            use_attention = (current_resolution == resolution_to_use_attention)
            current_channel_output = self.inner_channel * self.channels_multipliers[index]
            for j in range(0, self.number_res_packs + 1):
                UPS.append(ResnetBlock(current_channel_input+featured_channels.pop(), current_channel_output, \
                            time_embedding_dimension= self.time_embedding_dimension,\
                            dropout=dropout, use_attention=use_attention))
                current_channel_input = current_channel_output
            if flag_last == False:
                UPS.append(UpSample(current_channel_input))
                current_resolution = current_resolution * 2
        self.ups = nn.ModuleList(UPS)
        
        self.output_conv = smallBlock(current_channel_input, out_channels)

    def forward(self, x, time):

        skip_features = []
        for position_down, layer in enumerate(self.downs):
            if isinstance(layer, ResnetBlock):
                x = layer(x, time)
            else:
                x = layer(x)
            skip_features.append(x)  #This will save all the intermediate values so they can be passed to the upsample
        
        for position_mid, layer in enumerate(self.middle):
            if isinstance(layer, ResnetBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        
        for position_up, layer in enumerate(self.ups):
            if isinstance(layer, ResnetBlock):
                x = layer(torch.cat((x, skip_features.pop()), dim=1), time)
            else:

                x = layer(x)

        return self.output_conv(x)

