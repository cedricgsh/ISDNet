import torch
import torch.nn as nn
from torch.nn import init
import math
import time
from mmcv.cnn import ConvModule
from collections import OrderedDict

class ConvX(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1, sync=True):
        super(ConvX, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride, padding=kernel//2, bias=False)
        if sync:
            self.bn = nn.SyncBatchNorm(out_planes)
        else:
            self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn(self.conv(x)))
        return out



class AddBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(AddBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.Sequential(
                nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=1, groups=in_planes, bias=False),
                nn.BatchNorm2d(in_planes),
                nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_planes),
            )
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))

    def forward(self, x):
        out_list = []
        out = x

        for idx, conv in enumerate(self.conv_list):
            if idx == 0 and self.stride == 2:
                out = self.avd_layer(conv(out))
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            x = self.skip(x)

        return torch.cat(out_list, dim=1) + x



class CatBottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, block_num=3, stride=1):
        super(CatBottleneck, self).__init__()
        assert block_num > 1, print("block number should be larger than 1.")
        self.conv_list = nn.ModuleList()
        self.stride = stride
        if stride == 2:
            self.avd_layer = nn.Sequential(
                nn.Conv2d(out_planes//2, out_planes//2, kernel_size=3, stride=2, padding=1, groups=out_planes//2, bias=False),
                nn.BatchNorm2d(out_planes//2),
            )
            self.skip = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
            stride = 1

        for idx in range(block_num):
            if idx == 0:
                self.conv_list.append(ConvX(in_planes, out_planes//2, kernel=1))
            elif idx == 1 and block_num == 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//2, stride=stride))
            elif idx == 1 and block_num > 2:
                self.conv_list.append(ConvX(out_planes//2, out_planes//4, stride=stride))
            elif idx < block_num - 1:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx+1))))
            else:
                self.conv_list.append(ConvX(out_planes//int(math.pow(2, idx)), out_planes//int(math.pow(2, idx))))
            
    def forward(self, x):
        out_list = []
        out1 = self.conv_list[0](x)

        for idx, conv in enumerate(self.conv_list[1:]):
            if idx == 0:
                if self.stride == 2:
                    out = conv(self.avd_layer(out1))
                else:
                    out = conv(out1)
            else:
                out = conv(out)
            out_list.append(out)

        if self.stride == 2:
            out1 = self.skip(out1)
        out_list.insert(0, out1)

        out = torch.cat(out_list, dim=1)
        return out

class ShallowNet(nn.Module):
    def __init__(self, base=64, in_channels=3,  layers=[2,2], block_num=4, type="cat", dropout=0.20, pretrain_model=''):
        super(ShallowNet, self).__init__()
        if type == "cat":
            block = CatBottleneck
        elif type == "add":
            block = AddBottleneck
        self.in_channels = in_channels
        self.features = self._make_layers(base, layers, block_num, block)
        self.x2 = nn.Sequential(self.features[:1])
        self.x4 = nn.Sequential(self.features[1:2])
        self.x8 = nn.Sequential(self.features[2:4])
        self.x16 = nn.Sequential(self.features[4:6])
        if pretrain_model:
            print('use pretrain model {}'.format(pretrain_model))
            self.init_weight(pretrain_model)
        else:
            self.init_params()

    def init_weight(self, pretrain_model):

        state_dict = torch.load(pretrain_model)["state_dict"]
        self_state_dict = self.state_dict()
        for k, v in state_dict.items():
            if k == 'features.0.conv.weight' and self.in_channels != 3:
                v = torch.cat([v, v], dim=1)
            self_state_dict.update({k: v})
        self.load_state_dict(self_state_dict, strict=False)

    def init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def _make_layers(self, base, layers, block_num, block):
        features = []
        features += [ConvX(self.in_channels, base//2, 3, 2)]
        features += [ConvX(base//2, base, 3, 2)]

        for i, layer in enumerate(layers):
            for j in range(layer):
                if i == 0 and j == 0:
                    features.append(block(base, base*4, block_num, 2))
                elif j == 0:
                    features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
                else:
                    features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))

        return nn.Sequential(*features)

    def forward(self, x, cas3=False):
        feat2 = self.x2(x)
        feat4 = self.x4(feat2)
        feat8 = self.x8(feat4)
        feat16 = self.x16(feat8)
        if cas3:
            return feat4, feat8, feat16
        else:
            return feat8, feat16

#
# class STDCNet1446Full(nn.Module):
#     def __init__(self, base=64, layers=[4,5], block_num=4, in_channels=3 ,type="cat", dropout=0.20, pretrain_model=''):
#         super(STDCNet1446Full, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.in_channels = in_channels
#         self.features = self._make_layers(base, layers, block_num, block)
#
#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.x16 = nn.Sequential(self.features[6:11])
#         # self.x32 = nn.Sequential(self.features[11:])
#
#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()
#
#     def init_weight(self, pretrain_model):
#
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             if k == 'features.0.conv.weight' and self.in_channels != 3:
#                 v = torch.cat([v, v], dim=1)
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict, strict=False)
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(self.in_channels, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]
#
#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))
#
#         return nn.Sequential(*features)
#
#     def forward(self, x, cas3=False):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat16 = self.x16(feat8)
#         if cas3:
#             return feat4, feat8, feat16
#         else:
#             return feat8, feat16
#
# class STDCNet1446FullLR(nn.Module):
#     def __init__(self, base=64, layers=[4,5], block_num=4, type="cat", dropout=0.20, pretrain_model=''):
#         super(STDCNet1446FullLR, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.features = self._make_layers(base, layers, block_num, block)
#
#         # self.x2 = nn.Sequential(self.features[:1])
#         # self.x4 = nn.Sequential(self.features[1:2])
#         # self.x8 = nn.Sequential(self.features[2:6])
#         # self.x16 = nn.Sequential(self.features[6:11])
#         # self.x32 = nn.Sequential(self.features[11:])
#
#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()
#
#     def init_weight(self, pretrain_model):
#
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             if k == 'features.0.conv.weight':
#                 v = torch.cat([v, v], dim=1)
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict, strict=False)
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(6, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]
#
#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))
#
#         return nn.Sequential(*features)
#
#     def forward(self, x, cas3=False):
#         feat2 = self.features[:1](x)
#         feat4 = self.features[1:2](feat2)
#         feat8 = self.features[2:6](feat4)
#         feat16 = self.features[6:11](feat8)
#         if cas:
#             return feat4, feat8, feat16
#         else:
#             return feat8, feat16
#
# class TVAtt(nn.Module):
#     def __init__(self, in_channels, out_channels, conv_cfg=None, norm_cfg=dict(type='SyncBN', requires_grad=True), act_cfg=dict(type='ReLU')):
#         super(TVAtt, self).__init__()
#         self.conv_bn_relu = ConvModule(in_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
#         #self.conv_head = ConvModule(out_channels, out_channels, 3, stride=1, padding=1, conv_cfg=conv_cfg, norm_cfg=norm_cfg, act_cfg=act_cfg)
#         self.conv_1x1 = ConvModule(out_channels, out_channels, 1, stride=1, padding=0, conv_cfg=conv_cfg, norm_cfg=norm_cfg,act_cfg=None)
#         self.init_params()
#
#     def forward(self, x):
#         """Forward function."""
#         feat = self.conv_bn_relu(x)
#         h, w = feat.size()[2:]
#         h_tv = torch.pow(feat[..., 1:, :] - feat[..., :h-1, :], 2)
#         w_tv = torch.pow(feat[..., 1:] - feat[..., :w-1], 2)
#         atten = torch.mean(h_tv, dim=(2, 3), keepdim=True) + torch.mean(w_tv, dim=(2, 3), keepdim=True)
#         atten = self.conv_1x1(atten)
#         atten = atten.sigmoid()
#         out = torch.mul(feat, atten)
#         #out = self.conv_head(out)
#         return out
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
# class STDCNet1446TV(nn.Module):
#     def __init__(self, base=64, layers=[4,5], block_num=4, in_channels=3 ,type="cat", dropout=0.20, pretrain_model=''):
#         super(STDCNet1446TV, self).__init__()
#         if type == "cat":
#             block = CatBottleneck
#         elif type == "add":
#             block = AddBottleneck
#         self.in_channels = in_channels
#         self.features = self._make_layers(base, layers, block_num, block)
#         self.tv8 = TVAtt(256, 256)
#         self.tv16 = TVAtt(512, 512)
#         self.x2 = nn.Sequential(self.features[:1])
#         self.x4 = nn.Sequential(self.features[1:2])
#         self.x8 = nn.Sequential(self.features[2:6])
#         self.x16 = nn.Sequential(self.features[6:11])
#         # self.x32 = nn.Sequential(self.features[11:])
#
#         if pretrain_model:
#             print('use pretrain model {}'.format(pretrain_model))
#             self.init_weight(pretrain_model)
#         else:
#             self.init_params()
#
#     def init_weight(self, pretrain_model):
#
#         state_dict = torch.load(pretrain_model)["state_dict"]
#         self_state_dict = self.state_dict()
#         for k, v in state_dict.items():
#             if k == 'features.0.conv.weight' and self.in_channels != 3:
#                 v = torch.cat([v, v], dim=1)
#             self_state_dict.update({k: v})
#         self.load_state_dict(self_state_dict, strict=False)
#
#     def init_params(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 init.kaiming_normal_(m.weight, mode='fan_out')
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 init.constant_(m.weight, 1)
#                 init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 init.normal_(m.weight, std=0.001)
#                 if m.bias is not None:
#                     init.constant_(m.bias, 0)
#
#     def _make_layers(self, base, layers, block_num, block):
#         features = []
#         features += [ConvX(self.in_channels, base//2, 3, 2)]
#         features += [ConvX(base//2, base, 3, 2)]
#
#         for i, layer in enumerate(layers):
#             for j in range(layer):
#                 if i == 0 and j == 0:
#                     features.append(block(base, base*4, block_num, 2))
#                 elif j == 0:
#                     features.append(block(base*int(math.pow(2,i+1)), base*int(math.pow(2,i+2)), block_num, 2))
#                 else:
#                     features.append(block(base*int(math.pow(2,i+2)), base*int(math.pow(2,i+2)), block_num, 1))
#
#         return nn.Sequential(*features)
#
#     def forward(self, x, cas3=False):
#         feat2 = self.x2(x)
#         feat4 = self.x4(feat2)
#         feat8 = self.x8(feat4)
#         feat8 = self.tv8(feat8)
#         feat16 = self.x16(feat8)
#         feat16 = self.tv16(feat16)
#         if cas3:
#             return feat4, feat8, feat16
#         else:
#             return feat8, feat16
#
#
# if __name__ == "__main__":
#
#
#     # pretrain_model_g = "/apdcephfs/private_v_huaziguo/code/mmckpts/deeplabv3_R_14_retraining_old_wo_chlreduce_160k/latest.pth"
#     # pretrain_model_gl = "/apdcephfs/private_v_huaziguo/code/mmckpts/youtu_local_refine_adam_14/iter_16000.pth"
#     # g_dict = torch.load(pretrain_model_g)["state_dict"]
#     # gl_dict = torch.load(pretrain_model_gl)["state_dict"]
#     # for k in list(g_dict.keys()):
#     #     # print(g_dict[k].shape)
#     #     # print(gl_dict[k].shape)
#     #     if torch.sum(g_dict[k] - gl_dict[k]).item() != 0.:
#     #         print(k)
#     #         print(torch.sum(g_dict[k] - gl_dict[k]).item())
#     #     # print(torch.sum(g_dict[k] - gl_dict[k]))
#     #     # print(g_dict[k] - gl_dict[k])
#     #     # if g_dict[k] != gl_dict[k]:
#     #     #     print(k)
#
#     # print('---------')
#     # # refine_head.stdc_block.
#     # new_state = OrderedDict()
#     # for k, v in state_dict.items():
#     #     s = k.replace("refine_head.stdc_block.", "")
#     #     new_state[s] = v
#     #     #print(v)
#     # out = OrderedDict()
#     # out["state_dict"] = new_state
#     # torch.save(out, "./pretrain_stdc.pth")
#     # for k, v in new_state.items():
#     #     print(k)
#     #     print(v)
#     # total_iters = 300
#     x = torch.randn(1, 6, 256, 256)
#     net = STDCNet1446FullLR()
#     z = net(x)
#     for i in z:
#         print(i.shape)
#     #print(net)
#     # dummy = torch.randn(1, 6, 128, 128)
#     # output1, output2, output3 = net(dummy)
#     # print(output1.shape)
#     # print(output2.shape)
#     # print(output3.shape)
#     # num_warmup = 50
#     # pure_inf_time = 0
#     # dummy = torch.randn(1, 3, 1024, 2048)
#     # model = STDCNetLight(block_num=4, pretrain_model="/apdcephfs/private_v_huaziguo/code/mmckpts/STDCNet813M_73.91.tar")
#     # for name, param in model.named_parameters():
#     #     print(name)
#     #print(model)
#
#     #z1, z2 = model(dummy)
#     # print(model)
#     # print(z1.shape)
#     # print(z2.shape)
#     #model = STDCNetTiny(block_num=4, layers=[1, 2]).cuda()
#     # print(model)
#     # y =  model(dummy)
#     # for y_i in y:
#     #     print(y_i.shape)
#     # print(model)
#     # for param in
#
#     # for j in range(total_iters + num_warmup):
#
#     #     torch.cuda.synchronize()
#     #     start_time = time.perf_counter()
#
#     #     with torch.no_grad():
#     #         out = model(dummy)
#
#     #     torch.cuda.synchronize()
#     #     elapsed = time.perf_counter() - start_time
#
#     #     if (j + 1) > num_warmup:
#     #         pure_inf_time += elapsed
#
#     # fps = total_iters / pure_inf_time
#     # print(f'fps: {fps:.2f} img / s')
#
#     #      print(param)
#     # x = torch.randn(1,3,224,224)
#     # y = model(x)
#     # for i in y:
#     #     print(i.shape)
#     # quit(0)
