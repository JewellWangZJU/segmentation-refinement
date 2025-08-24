import torch
import torch.nn as nn
import torch.nn.functional as F

class VigSegLoss(nn.Module):
    def __init__(self, args):
        super(VigSegLoss, self).__init__()
        self.args = args
        self.num_class = args.num_class
        
        # 使用标准的语义分割损失
        if self.num_class == 1:
            # 二分类使用二元交叉熵损失和Dice损失
            self.bce_loss = nn.BCEWithLogitsLoss()
            self.dice_loss = DiceLoss()
        else:
            # 多分类使用交叉熵损失和Dice损失
            self.ce_loss = nn.CrossEntropyLoss()
            self.dice_loss = DiceLoss(multiclass=True)

    def forward(self, output, target):
        if self.num_class == 1:
            # 二分类
            bce = self.bce_loss(output, target)
            dice = self.dice_loss(output, target)
            loss = bce + dice
        else:
            # 多分类
            # 确保目标格式正确
            if target.dim() == 4 and target.size(1) == 1:
                target = target.squeeze(1).long()
            
            ce = self.ce_loss(output, target)
            dice = self.dice_loss(output, target)
            loss = ce + dice
            
        return loss


class DiceLoss(nn.Module):
    def __init__(self, multiclass=False):
        super(DiceLoss, self).__init__()
        self.multiclass = multiclass

    def forward(self, output, target):
        smooth = 1e-5
        
        if self.multiclass:
            # 多分类Dice损失
            if output.dim() == 4:
                # 输出是 (B, C, H, W)，目标应该是 (B, H, W)
                if target.dim() == 4 and target.size(1) == 1:
                    target = target.squeeze(1)
                
                # 转换为one-hot编码
                num_classes = output.size(1)
                target_onehot = torch.zeros_like(output)
                target_onehot.scatter_(1, target.unsqueeze(1), 1)
                
                # 计算每个类的Dice系数
                intersection = torch.sum(output * target_onehot, dim=(2, 3))
                union = torch.sum(output, dim=(2, 3)) + torch.sum(target_onehot, dim=(2, 3))
                dice = (2. * intersection + smooth) / (union + smooth)
                
                # 平均所有类的Dice损失（不包括背景类，索引0）
                dice_loss = 1 - dice[:, 1:].mean()
            else:
                dice_loss = torch.tensor(0.0, device=output.device)
        else:
            # 二分类Dice损失
            output = torch.sigmoid(output)
            intersection = torch.sum(output * target)
            union = torch.sum(output) + torch.sum(target)
            dice = (2. * intersection + smooth) / (union + smooth)
            dice_loss = 1 - dice
            
        return dice_loss


