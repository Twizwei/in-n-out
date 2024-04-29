import torch
from torch import nn
import torchvision
from configs.paths_config import model_paths
from CLIPStyle.mapper.facial_recognition.model_irse import Backbone

"""
Original implementation: https://github.com/omertov/encoder4editing/blob/99ea50578695d2e8a1cf7259d8ee89b23eea942b/criteria/id_loss.py
Modified from [StyleCLIP](https://github.com/orpatashnik/StyleCLIP/blob/main/criteria/id_loss.py).
"""

class IDLoss(nn.Module):
    def __init__(self):
        super(IDLoss, self).__init__()
        print('Loading ResNet ArcFace')
        self.facenet = Backbone(input_size=112, num_layers=50, drop_ratio=0.6, mode='ir_se')
        self.facenet.load_state_dict(torch.load(model_paths['ir_se50']))
        self.face_pool = torch.nn.AdaptiveAvgPool2d((112, 112))
        self.facenet.eval()
        interp_mode = torchvision.transforms.transforms.InterpolationMode.BICUBIC
        self.preprocess = torchvision.transforms.Compose([
            torchvision.transforms.Resize(256, interp_mode),
            torchvision.transforms.CenterCrop(191),  # explicit: StyleGAN alignment -> ArcFace alignment
            torchvision.transforms.Resize(112, interp_mode),
            torchvision.transforms.Normalize([0.5] * 3, [0.5 * 256 / 255] * 3),
        ])
        self.criterion = torch.nn.CosineSimilarity(dim=1)
        for module in [self.facenet, self.face_pool]:
            for param in module.parameters():
                param.requires_grad = False

    def extract_feats(self, x):
        # x = x[:, :, 35:223, 32:220]  # Crop interesting region
        # x = self.face_pool(x)
        x = self.preprocess(x)
        x_feats = self.facenet(x)
        return x_feats

    def calc_similarity(self, y_hat, y):
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        similarity = self.criterion(y_feats, y_hat_feats)

        return similarity

    def forward(self, y_hat, y):
        n_samples = y.shape[0]
        y_feats = self.extract_feats(y)  # Otherwise use the feature from there
        y_hat_feats = self.extract_feats(y_hat)
        y_feats = y_feats.detach()
        loss = 0
        # sim_improvement = 0
        count = 0
        for i in range(n_samples):
            diff_target = y_hat_feats[i].dot(y_feats[i])
            loss += 1 - diff_target
            count += 1

        # return loss / count, sim_improvement / count
        return loss / count

        # y_hat_feats = self.facenet(self.preprocess(y_hat))
        # y_feats = self.facenet(self.preprocess(y)).detach()
        # y_hat_feats /= y_hat_feats.norm(dim=-1, keepdim=True)
        # y_feats /= y_feats.norm(dim=-1, keepdim=True)
        # loss = (1.0 - self.criterion(y_hat_feats, y_feats)).mean()
        # return loss        

# import insightface
# class ArcFaceLoss(torch.nn.Module):
#     def __init__(self,
#                  model_name="iresnet50",  # choices: "iresnet34", "iresnet50", "iresnet100"
#                  ):
#         super().__init__()
#         self.arcface = getattr(insightface, model_name)(pretrained=True).eval().requires_grad_(False)
#         mean = [0.5] * 3
#         std = [0.5 * 256 / 255] * 3
#         input_resolution = 112
#         interp_mode = torchvision.transforms.transforms.InterpolationMode.BICUBIC
#         self.preprocess = torchvision.transforms.Compose([
#             torchvision.transforms.Resize(256, interp_mode),
#             torchvision.transforms.CenterCrop(191),  # explicit: StyleGAN alignment -> ArcFace alignment
#             torchvision.transforms.Resize(input_resolution, interp_mode),
#             torchvision.transforms.Normalize(mean, std),
#         ])
#         self.criterion = torch.nn.CosineSimilarity(dim=1)

#     def forward(self, input, target):
#         input = self.arcface(self.preprocess(input))
#         target = self.arcface(self.preprocess(target))
#         input /= input.norm(dim=-1, keepdim=True)
#         target /= target.norm(dim=-1, keepdim=True)
#         loss = -self.criterion(input, target).mean()
#         return loss