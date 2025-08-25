from random import shuffle
import numpy as np
import vig_loss 
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from torch.optim import lr_scheduler
from lr_update import get_lr
from metrics.cldice import clDice
import os
from torch.cuda.amp import autocast, GradScaler
import sklearn
import torchvision.utils as utils
from sklearn.metrics import precision_score
from skimage.io import imread, imsave



class Solver(object):
    def __init__(self, args,optim=torch.optim.Adam):
        self.args = args
        self.optim = optim
        self.NumClass = self.args.num_class
        self.lr = self.args.lr
        H, W = args.resize


        self.hori_translation = torch.zeros([1,self.NumClass,W,W])
        for i in range(W-1):
            self.hori_translation[:,:,i,i+1] = torch.tensor(1.0)
        self.verti_translation = torch.zeros([1,self.NumClass,H,H])
        for j in range(H-1):
            self.verti_translation[:,:,j,j+1] = torch.tensor(1.0)
        self.hori_translation = self.hori_translation.float()
        self.verti_translation = self.verti_translation.float()
        self.loss_func = vig_loss.VigSegLoss(args).cuda()

    def create_exp_directory(self,exp_id):
        if not os.path.exists('models/' + str(exp_id)):
            os.makedirs('models/' + str(exp_id))

        csv = 'results_'+str(exp_id)+'.csv'
        with open(os.path.join(self.args.save, csv), 'w') as f:
            f.write('epoch, dice, Jac, clDice \n')



    def get_density(self, pos_cnt,bins = 50):
        ### only used for Retouch in this code
        val_in_bin_ = [[],[],[]]
        density_ = [[],[],[]]
        bin_wide_ = []

        ### check
        for n in range(3):
            density = []
            val_in_bin = []
            c1 = [i for i in pos_cnt[n] if i != 0]
            c1_t = torch.tensor(c1)
            bin_wide = (c1_t.max()+50)/bins
            bin_wide_.append(bin_wide)

            edges = torch.arange(bins + 1).float()*bin_wide
            for i in range(bins):
                val = [c1[j] for j in range(len(c1)) if ((c1[j] >= edges[i]) & (c1[j] < edges[i + 1]))]
                # print(val)
                val_in_bin.append(val)
                inds = (c1_t >= edges[i]) & (c1_t < edges[i + 1]) #& valid
                num_in_bin = inds.sum().item()
                # print(num_in_bin)
                density.append(num_in_bin)

            denominator = torch.tensor(density).sum()
            # print(val_in_bin)

            #### get density ####
            density = torch.tensor(density)/denominator
            density_[n]=density
            val_in_bin_[n] = val_in_bin
        print(density_)

        return density_, val_in_bin_,bin_wide_



    def train(self, model, train_loader, val_loader,exp_id, num_epochs=10):

        optim = self.optim(model.parameters(), lr=self.lr)

        print('START TRAIN.')
        
        self.create_exp_directory(exp_id)

        net = model.cuda() 
        optimizer = self.optim(net.parameters(), lr=self.lr)
        scaler = GradScaler()


        best_p = 0
        best_epo = 0
        scheduled = ['CosineAnnealingWarmRestarts']
        if self.args.lr_update in scheduled:
            scheduled = True
            if self.args.lr_update == 'CosineAnnealingWarmRestarts':
                scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2, eta_min = 0.00001)
        else:
            scheduled = False

        if self.args.test_only:
            self.test_epoch(net,val_loader,0,exp_id)
        else:
            for epoch in range(self.args.epochs):
                net.train()

                if scheduled:
                    scheduler.step()
                else:
                    curr_lr = get_lr(self.lr,self.args.lr_update, epoch, num_epochs, gamma=self.args.gamma,step=self.args.lr_step)
                    for param_group in optim.param_groups:
                        param_group['lr'] = curr_lr
                

                for i_batch, sample_batched in enumerate(train_loader):
                    X = Variable(sample_batched[0]).cuda()
                    if self.args.num_class == 1:
                        y = Variable(sample_batched[1]).float().cuda() 
                        if y.dim == 3:
                            y = y.unsqueeze(1).float()
                    else:
                        y = Variable(sample_batched[1]).long().cuda() # [N,H,W]


                    optimizer.zero_grad()
                    with autocast():
                        output = net(X)

                        # 空间尺寸对齐 (很多头是1/4或1/8输出)
                        if output.shape[-2:] != y.shape[-2:]:
                            output = F.interpolate(output, size=y.shape[-2:], mode='bilinear', align_corners=False)

                        loss = self.loss_func(output, y)
                    
                    if torch.isinf(loss) or torch.isnan(loss):
                        print("Warning: Loss Inf or NaN. Stop Training!")
                        exit()

                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    print('[epoch:'+str(epoch)+'][Iteration : ' + str(i_batch) + '/' + str(len(train_loader)) + '] Total:%.3f' %(
                        loss.item()))



                dice_p = self.test_epoch(net,val_loader,epoch,exp_id)
                if best_p<dice_p:
                    best_p = dice_p
                    best_epo = epoch
                    torch.save(model.state_dict(), 'models/' + str(exp_id) + '/best_model.pth')
                if (epoch+1) % self.args.save_per_epochs == 0:
                    torch.save(model.state_dict(), 'models/' + str(exp_id) + '/'+str(epoch+1)+'_model.pth')
                print('[Epoch :%d] total loss:%.3f ' %(epoch,loss.item()))

                # if epoch%self.args.save_per_epochs==0:
                #     torch.save(model.state_dict(), 'models/' + str(exp_id) + '/epoch' + str(epoch + 1)+'.pth')
            csv = 'results_'+str(exp_id)+'.csv'
            with open(os.path.join(self.args.save, csv), 'a') as f:
                f.write('%03d,%0.6f \n' % (
                    best_epo,
                    best_p
                ))
            # writer.close()
            print('FINISH.')
            
    def test_epoch(self, model, loader, epoch, exp_id):
        model.eval()
        self.dice_ls, self.Jac_ls, self.cldc_ls = [], [], []

        with torch.no_grad():
            for j_batch, test_data in enumerate(loader):
                X_test = test_data[0].cuda()
                y_test = test_data[1].cuda()

                output_test = model(X_test)  # [N,C,h,w] or [N,1,h,w]

                # 尺寸对齐到 label
                if output_test.shape[-2:] != y_test.shape[-2:]:
                    output_test = F.interpolate(output_test, size=y_test.shape[-2:], mode='bilinear', align_corners=False)

                if self.args.num_class == 1:
                    # ---- 二分类：sigmoid + 阈值 ----
                    # 期望：y_test 为 0/1，形状 [N,1,H,W]（若 [N,H,W] 则先升维）
                    if y_test.dim() == 3:
                        y_test = y_test.unsqueeze(1)
                    y_test = y_test.float()                    # [N,1,H,W]

                    prob = torch.sigmoid(output_test)          # [N,1,H,W]
                    pred = (prob > 0.5).float()                # [N,1,H,W]

                    # clDice（若你需要）
                    try:
                        from metrics.cldice import clDice
                        pred_np   = pred.squeeze(1).cpu().numpy()
                        target_np = y_test.squeeze(1).cpu().numpy()
                        cldc = clDice(pred_np, target_np)
                    except Exception:
                        cldc = 0.0
                    self.cldc_ls.append(cldc)

                    # 计算Dice/Jaccard（对齐你现有的 per_class_dice 接口：用 one-hot）
                    y_oh = torch.cat([1 - y_test, y_test], dim=1)     # [N,2,H,W]
                    p_oh = torch.cat([1 - pred,   pred  ], dim=1)     # [N,2,H,W]
                    dice, Jac = self.per_class_dice(p_oh, y_oh)        # 返回 per-class

                    # 记录前景类（索引1）
                    self.dice_ls += dice[:, 1].tolist()
                    self.Jac_ls  += Jac[:, 1].tolist()

                else:
                    # ---- 多分类：softmax + argmax ----
                    # 期望：y_test 为 Long，形状 [N,H,W]（若 [N,1,H,W] 则 squeeze）
                    if y_test.dim() == 4 and y_test.size(1) == 1:
                        y_test = y_test[:, 0, ...]
                    y_test = y_test.long()                      # [N,H,W]

                    prob = F.softmax(output_test, dim=1)        # [N,C,H,W]
                    pred_idx = prob.argmax(dim=1)               # [N,H,W]

                    # one-hot 后计算Dice/Jaccard
                    pred = self.one_hot(pred_idx.unsqueeze(1), X_test.shape)  # [N,C,H,W]
                    y_oh = self.one_hot(y_test.unsqueeze(1),  X_test.shape)   # [N,C,H,W]

                    dice, Jac = self.per_class_dice(pred, y_oh)
                    # 记录前景均值（排除背景索引0）
                    self.dice_ls += torch.mean(dice[:, 1:], 1).tolist()
                    self.Jac_ls  += torch.mean(Jac[:, 1:], 1).tolist()

                if j_batch % max(1, len(loader)//5) == 0:
                    print(f'[Iteration : {j_batch}/{len(loader)}] Total DSC:{np.mean(self.dice_ls):.3f}')

            Jac_ls   = np.array(self.Jac_ls)
            dice_ls  = np.array(self.dice_ls)
            total_dice = float(np.mean(dice_ls)) if dice_ls.size > 0 else 0.0

            # 记录到 csv
            csv = f'results_{exp_id}.csv'
            with open(os.path.join(self.args.save, csv), 'a') as f:
                f.write('%03d,%0.6f,%0.6f,%0.6f \n' % (
                    (epoch + 1),
                    total_dice,
                    float(np.mean(Jac_ls)) if Jac_ls.size > 0 else 0.0,
                    float(np.mean(self.cldc_ls)) if (self.args.num_class == 1 and len(self.cldc_ls)>0) else 0.0
                ))
            return total_dice

    def per_class_dice(self,y_pred, y_true):
        eps = 0.0001
        y_pred = y_pred
        y_true = y_true

        FN = torch.sum((1-y_pred)*y_true,dim=(2,3)) 
        FP = torch.sum((1-y_true)*y_pred,dim=(2,3)) 
        Pred = y_pred
        GT = y_true
        inter = torch.sum(GT* Pred,dim=(2,3)) 


        union = torch.sum(GT,dim=(2,3)) + torch.sum(Pred,dim=(2,3)) 
        dice = (2*inter+eps)/(union+eps)
        Jac = (inter+eps)/(inter+FP+FN+eps)

        return dice, Jac

    def one_hot(self,target,shape):

        one_hot_mat = torch.zeros([shape[0],self.args.num_class,shape[2],shape[3]]).cuda()
        target = target.cuda()
        one_hot_mat.scatter_(1, target, 1)
        return one_hot_mat

def get_mask(output):
    output = F.softmax(output,dim=1)
    _,pred = output.topk(1, dim=1)
    #pred = pred.squeeze()
    
    return pred

