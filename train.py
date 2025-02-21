import sys
import os
import new_reader as reader
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import importlib
import yaml
import ctools
import gtools
import argparse
# import wandb
from tensorboardX import SummaryWriter
from easydict import EasyDict
import torch.nn.functional as F
from pathlib import Path
import torchvision.transforms as transforms
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model.change_CBAM2 import Model  # ---------------------------------------------------------------------------------------

base_dir = os.getcwd()
sys.path.append(base_dir)
log_dir = Path(
    './logs/...')  # --------------------------------------------------------------------------------
log_dir.mkdir(exist_ok=True, parents=True)


class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()

    def forward(self, input, target):
        cosine_similarity = F.cosine_similarity(input, target, dim=1)
        loss = 1 - cosine_similarity
        return loss.mean()


class AngularLoss(nn.Module):
    def __init__(self):
        super(AngularLoss, self).__init__()

    def forward(self, input, target):
        cosine_similarity = F.cosine_similarity(input, target, dim=1)
        angular_loss = torch.acos(cosine_similarity)
        return angular_loss.mean()


def CropEyeWithCenter(batch_size, face_, center_):
    im_ = []
    for i in range(batch_size):
        face = face_[i]
        center = center_[i]
        center_x = center[0]
        center_y = center[1]

        width = 60 * 1.2
        height = 36 * 1.2

        x1 = [max(center_x - width / 2, 0), max(center_y - height / 2, 0)]
        x2 = [min(x1[0] + width, 224), min(x1[1] + height, 224)]  # image size = [224, 224]
        im = face[:, int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
        # print("im:", im.size())
        im = F.interpolate(im.unsqueeze(0), size=(36, 60), mode='bilinear', align_corners=False).squeeze(0)
        # print('reimg1.eye.shape:', im.shape)
        im_.append(im)
    im_ = torch.stack(im_)
    return im_


def RestFaceWithCenter(im_face, lcenter, rcenter):
    im = im_face
    center_lx, center_ly, center_rx, center_ry = lcenter[0], lcenter[1], rcenter[0], rcenter[1]

    lwidth = 60 * 1.2
    lheight = 36 * 1.2
    lx1 = [max(center_lx - lwidth / 2, 0), max(center_ly - lheight / 2, 0)]
    lx2 = [min(lx1[0] + lwidth, 224), min(lx1[1] + lheight, 224)]

    rwidth = 60 * 1.2
    rheight = 36 * 1.2
    rx1 = [max(center_rx - rwidth / 2, 0), max(center_ry - rheight / 2, 0)]
    rx2 = [min(rx1[0] + rwidth, 224), min(rx1[1] + rheight, 224)]

    top_region = im[:, :int(min(lx1[1], rx1[1])), :]
    # print("top", top_region.shape)

    middle_start = int(min(lx1[1], rx1[1]))
    middle_end = int(max(lx2[1], rx2[1]))

    mid_region = torch.cat([
        im[:, middle_start:middle_end, :int(lx1[0])],
        im[:, middle_start:middle_end, int(lx2[0]):int(rx1[0])],
        im[:, middle_start:middle_end, int(rx2[0]):]
    ], dim=2)
    # print("middle", mid_region.shape)

    bottom_region = im[:, int(max(lx2[1], rx2[1])):, :]
    # print('bottom', bottom_region.shape)

    return top_region, mid_region, bottom_region


def Loss_RemoveEyesFromFaceWithCenter(batch_size, org_face_, im_face_, lcenter_, rcenter_):
    mse_loss = nn.MSELoss()
    sum_loss = 0
    # org_top, org_mid, org_bot = [], [], []
    # im_top, im_mid, im_bot = [], [], []
    for i in range(batch_size):
        org_face = org_face_[i]
        im_face = im_face_[i]
        lcenter, rcenter = lcenter_[i], rcenter_[i]
        otop, omid, obot = RestFaceWithCenter(org_face, lcenter, rcenter)
        top, mid, bot = RestFaceWithCenter(im_face, lcenter, rcenter)
        loss = mse_loss(top, otop) + mse_loss(mid, omid) + mse_loss(bot, obot)
        sum_loss = sum_loss + loss
    sum_loss = sum_loss / batch_size
    return sum_loss


def CropEye(batch_size, face_, lcorner_, rcorner_):
    # print("face_:", face_.size())
    im_ = []
    for i in range(batch_size):
        face = face_[i]
        lcorner = lcorner_[i]
        rcorner = rcorner_[i]
        x, y = list(zip(lcorner, rcorner))

        center_x = np.mean(x)
        center_y = np.mean(y)

        width = np.abs(x[0] - x[1]) * 1.5
        times = width / 60
        height = 36 * times

        x1 = [max(center_x - width / 2, 0), max(center_y - height / 2, 0)]
        x2 = [min(x1[0] + width, 224), min(x1[1] + height, 224)]  # image size = [224, 224]
        im = face[:, int(x1[1]):int(x2[1]), int(x1[0]):int(x2[0])]
        # print("im:", im.size())
        im = F.interpolate(im.unsqueeze(0), size=(36, 60), mode='bilinear', align_corners=False).squeeze(0)
        # print('reimg1.eye.shape:', im.shape)
        im_.append(im)
    im_ = torch.stack(im_)
    return im_


def RestFace(im_face, llcorner, lrcorner, rlcorner, rrcorner):
    im = im_face
    lx, ly = list(zip(llcorner, lrcorner))
    rx, ry = list(zip(rlcorner, rrcorner))

    center_lx, center_ly, center_rx, center_ry = np.mean(lx), np.mean(ly), np.mean(rx), np.mean(ry)

    lwidth = np.abs(lx[0] - lx[1]) * 1.5
    ltimes = lwidth / 60
    lheight = 36 * ltimes
    lx1 = [max(center_lx - lwidth / 2, 0), max(center_ly - lheight / 2, 0)]
    lx2 = [min(lx1[0] + lwidth, 224), min(lx1[1] + lheight, 224)]

    rwidth = np.abs(rx[0] - rx[1]) * 1.5
    rtimes = rwidth / 60
    rheight = 36 * rtimes
    rx1 = [max(center_rx - rwidth / 2, 0), max(center_ry - rheight / 2, 0)]
    rx2 = [min(rx1[0] + rwidth, 224), min(rx1[1] + rheight, 224)]

    top_region = im[:, :int(min(lx1[1], rx1[1])), :]
    # print("top", top_region.shape)

    middle_start = int(min(lx1[1], rx1[1]))
    middle_end = int(max(lx2[1], rx2[1]))

    mid_region = torch.cat([
        im[:, middle_start:middle_end, :int(lx1[0])],
        im[:, middle_start:middle_end, int(lx2[0]):int(rx1[0])],
        im[:, middle_start:middle_end, int(rx2[0]):]
    ], dim=2)
    # print("middle", mid_region.shape)

    bottom_region = im[:, int(max(lx2[1], rx2[1])):, :]
    # print('bottom', bottom_region.shape)

    return top_region, mid_region, bottom_region


def Loss_RemoveEyesFromFace(batch_size, org_face_, im_face_, llc_, lrc_, rlc_, rrc_):
    mse_loss = nn.MSELoss()
    sum_loss = 0
    for i in range(batch_size):
        org_face = org_face_[i]
        im_face = im_face_[i]
        llc, lrc, rlc, rrc = llc_[i], lrc_[i], rlc_[i], rrc_[i],
        otop, omid, obot = RestFace(org_face, llc, lrc, rlc, rrc)
        top, mid, bot = RestFace(im_face, llc, lrc, rlc, rrc)
        loss = mse_loss(top, otop) + mse_loss(mid, omid) + mse_loss(bot, obot)
        sum_loss = sum_loss + loss
    sum_loss = sum_loss / batch_size
    return sum_loss


def SaveVisualImg(person, epoch, reimg_, name):
    reimg = reimg_
    batch_size, _, _, _ = reimg.size()
    savepath = '...'  # ---------------------------------------------------------
    for i in range(batch_size):
        img = reimg[i]
        img = img.permute(1, 2, 0)  # c, h, w
        img_array = img.cpu().detach().numpy()
        b, g, r = cv2.split(img_array)
        img_array_rgb = cv2.merge([r, g, b])
        plt.imshow(img_array_rgb)  # h, w, c
        imgpath = Path(savepath + '/' + name + '/p' + str(person) + '/' + str(epoch))
        imgpath.mkdir(exist_ok=True, parents=True)
        # image = Image.fromarray(img_array)
        imgpath = savepath + '/' + name + '/p' + str(person) + '/' + str(epoch) + '/' + str(i) + '.png'
        plt.savefig(imgpath)
    return True


def train(config, person=None):
    save = config.save
    params = config.params
    data = config.data

    module = importlib.import_module(
        "model")  # -----------------------------------------------------------------
    model = getattr(module, config.model_name)()

    device = torch.device(config.device)

    model.to(device)

    print("===> Read data <===")
    if person is not None:
        print(f"===> Train excluding person {person} <===")
        specific = [i for i in range(0, 15) if i != person]  ---------------------------------------------
        data, _ = ctools.readfolder(data, specific)

        savepath = os.path.join(save.metapath, save.folder, str(person), f"checkpoint")
    else:
        if data.isFolder:
            data, _ = ctools.readfolder(data)
            savepath = os.path.join(save.metapath, save.folder, person, f"checkpoint")
        else:
            savepath = os.path.join(save.metapath, save.folder, f"checkpoint")

    dataloader = reader.loader(data, params.batch_size, shuffle=True, num_workers=8)  # [c, h, w]

    print("===> Model building <===")

    model = Model().to(config.device)  # ------------------------------------------------------------------------------

    pretrain = config.pretrain
    if pretrain.enable and pretrain.device:
        model.load_state_dict(
            torch.load(
                pretrain.path,
                map_location={f"cuda:{pretrain.device}": f"cuda:{config.device}"},
            )
        )
    elif pretrain.enable and not pretrain.device:
        model.load_state_dict(torch.load(pretrain.path))

    print("===> Optimizer building <===")

    optimizer = optim.AdamW(model.parameters(), params.lr, weight_decay=params.weight_decay)

    scheduler = optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=params.milestones, gamma=params.decay
    )

    loss_fn = nn.L1Loss().to(device)  # -----------------------------------------------------------------------------
    loss_recon = nn.MSELoss().to(device)

    if not os.path.exists(savepath):
        os.makedirs(savepath)

    print("===> Training <===")

    model.train()

    length = len(dataloader)
    total = length * params.epoch
    timer = ctools.TimeCounter(total)

    optimizer.zero_grad()
    optimizer.step()
    scheduler.step()

    loss_meter = ctools.AverageMeter()
    angle_error_meter = ctools.AverageMeter()

    with open(os.path.join(savepath, "train.log"), "w") as f:
        f.write(ctools.DictDumps(config) + "\n")

        for epoch in range(1, params.epoch + 1):
            loss_meter.reset()
            angle_error_meter.reset()

            log_epoch_dir = Path(str(log_dir) + '/' + str(epoch))
            log_epoch_dir.mkdir(exist_ok=True, parents=True)
            writer = SummaryWriter(log_dir=str(log_epoch_dir))

            for i, (data, anno) in enumerate(dataloader):
                for key in data:
                    if key != "name":
                        data[key] = data[key].to(device)

                anno_2d = anno['_2d'].to(device)

                if epoch >= 61:  # [c, h, w]
                    SaveVisualImg(person, epoch, data['left'], 'left')
                    SaveVisualImg(person, epoch, data['right'], 'right')
                    SaveVisualImg(person, epoch, data['face'], 'face')

                yaw, pitch, reimg1, reimg2 = model(data['left'], data['right'], data['face'])

                gaze_2d = torch.cat((yaw, pitch), dim=1)
                loss_g = loss_fn(gaze_2d, anno_2d)

                if params.name == 'diap' or params.name == 'g360':
                    reimg1_left = CropEyeWithCenter(params.batch_size, reimg1, anno['_lcenter_2d'])
                    reimg1_right = CropEyeWithCenter(params.batch_size, reimg1, anno['_rcenter_2d'])

                    loss_re1 = loss_recon(reimg1_left, data['left']) + loss_recon(reimg1_right, data['right'])
                    loss_re2 = Loss_RemoveEyesFromFaceWithCenter(params.batch_size, data['face'], reimg2,
                                                                 anno['_lcenter_2d'], anno['_rcenter_2d'])
                else:
                    reimg1_left = CropEye(params.batch_size, reimg1, anno['_llc'], anno['_lrc'])
                    reimg1_right = CropEye(params.batch_size, reimg1, anno['_rlc'], anno['_rrc'])

                    if epoch == 61:
                        SaveVisualImg(person, epoch, reimg1, 'reimg1')
                        SaveVisualImg(person, epoch, reimg1_left, 'reimg1_left')
                        SaveVisualImg(person, epoch, reimg1_right, 'reimg1_right')
                        SaveVisualImg(person, epoch, reimg2, 'reimg2')

                    # reconstruct eyes
                    loss_re1 = loss_recon(reimg1_left, data['left']) + loss_recon(reimg1_right, data['right'])
                    # reconstruct non-eyes
                    loss_re2 = Loss_RemoveEyesFromFace(params.batch_size, data['face'], reimg2, anno['_llc'], anno['_lrc'],
                                                       anno['_rlc'], anno['_rrc'])

                optimizer.zero_grad()

                loss = loss_g + loss_re1 + loss_re2

                gaze_3d = gtools.gazeto3d_batch(gaze_2d)
                # anno_3d = anno['_3d'].to(device)
                anno_3d = gtools.gazeto3d_batch(anno_2d)
                angle_error = gtools.angular_batch(gaze_3d, anno_3d).mean()

                loss.backward()
                torch.nn.utils.clip_grad_norm(model.parameters(), max_norm=1.0)
                # print(model.multiscaledecouple.cbam_0.channel_attention.sigma.grad)
                optimizer.step()

                rest = timer.step() / 3600

                num = data['face'].size(0)
                loss_meter.update(loss.item(), num)
                angle_error_meter.update(angle_error.item(), num)

                writer.add_scalar('Loss', loss_meter.val, i + 1)
                writer.add_scalar('Angle Error', angle_error_meter.val, i + 1)
                if i % 20 == 0:
                    log = (f"Epoch: {epoch} | Iter: {i}/{length} | LR: {ctools.GetLR(optimizer)} | "
                           f"Loss: {loss_meter.val:.6f} | Avg Loss: {loss_meter.avg:.6f} | "
                           f"Angle Error: {angle_error_meter.val:.3f} | Avg Angle Error: {angle_error_meter.avg:.3f} | "
                           f"Time: {rest:.2f}h")
                    print(log)
                    f.write(log + "\n")
                    sys.stdout.flush()
                    f.flush()

            writer.close()

            scheduler.step()

            if epoch % save.step == 0:
                torch.save(
                    model.state_dict(),
                    os.path.join(savepath, f"Iter_{epoch}_{save.model_name}.pth")
                )


def main(config):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    cudnn.benchmark = config.cudnn.benchmark
    cudnn.deterministic = config.cudnn.deterministic

    data = config.data
    if data.name == 'mpii':
        for i in range(0, 15):
            # config = EasyDict(yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)).train
            data.label = "..."
            train(config, person=i)

    else:
        train(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="New Training")  # -----------------------------------
    parser.add_argument("--config", type=str, help="Path to the config file")
    args = parser.parse_args()
    config = EasyDict(yaml.load(open(args.config, "r"), Loader=yaml.FullLoader))
    config = config.train

    main(config)