import os
import cv2
import torch
import numpy as np
from easydict import EasyDict as edict
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


def Decode_MPII(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[5], line[6]
    anno.gaze2d, anno.head2d = line[7], line[8]
    anno.llc = line[11]
    anno.lrc = line[12]
    anno.rlc = line[13]
    anno.rrc = line[14]
    return anno


def Decode_Diap(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d, anno.head3d = line[4], line[5]
    anno.gaze2d, anno.head2d = line[6], line[7]

    anno.lcenter_2d, anno.rcenter_2d = line[8], line[9]
    return anno


def Decode_Gaze360(line):
    anno = edict()
    anno.face, anno.lefteye, anno.righteye = line[0], line[1], line[2]
    anno.name = line[3]

    anno.gaze3d = line[4]
    anno.gaze2d = line[5]
    # anno.llc = line[6]
    # anno.lrc = line[7]
    # anno.rlc = line[8]
    # anno.rrc = line[9]
    anno.lcenter_2d = line[6]
    anno.rcenter_2d = line[7]
    return anno


def Decode_ETH(line):
    anno = edict()
    anno.face = line[0]
    anno.gaze2d = line[1]
    anno.head2d = line[2]
    anno.name = line[3]
    return anno


def Decode_RTGene(line):
    anno = edict()
    anno.face = line[0]
    anno.lefteye = line[1]
    anno.righteye = line[2]
    anno.gaze3d = line[4]
    anno.gaze2d = line[6]
    anno.head2d = line[7]
    anno.llc = line[8]
    anno.lrc = line[9]
    anno.rlc = line[10]
    anno.rrc = line[11]
    anno.name = line[0]
    return anno


def Decode_Dict():
    mapping = edict()
    mapping.mpiigaze = Decode_MPII
    mapping.eyediap = Decode_Diap
    mapping.gaze360 = Decode_Gaze360
    mapping.ethtrain = Decode_ETH
    mapping.rtgene = Decode_RTGene
    return mapping


def long_substr(str1, str2):
    substr = ""
    for i in range(len(str1)):
        for j in range(len(str1) - i + 1):
            if j > len(substr) and (str1[i : i + j] in str2):
                substr = str1[i : i + j]
    return len(substr)


def Get_Decode(name):
    mapping = Decode_Dict()
    keys = list(mapping.keys())
    name = name.lower()
    score = [long_substr(name, i) for i in keys]
    key = keys[score.index(max(score))]
    return mapping[key]


def gray_to_rgb(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img


class Trainloader(Dataset):
    def __init__(self, dataset):

        # Read source data
        self.data = edict()
        self.data.line = []
        self.data.root = dataset.image
        self.data.decode = Get_Decode(dataset.name)

        if isinstance(dataset.label, list):
            for i in dataset.label:
                with open(i) as f:
                    line = f.readlines()

                if dataset.header:
                    line.pop(0)

                self.data.line.extend(line)

        else:

            with open(dataset.label) as f:
                self.data.line = f.readlines()

            if dataset.header:
                self.data.line.pop(0)

        # build transforms
        self.eye_transforms = transforms.Compose(
            [
                # transforms.Lambda(lambda x: gray_to_rgb(x)),
                transforms.ToTensor(),
                # transforms.Resize((224, 224)),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ]
        )

    def __len__(self):

        return len(self.data.line)

    def __getitem__(self, idx):

        # Read souce information
        line = self.data.line[idx]
        line = line.strip().split(" ")
        anno = self.data.decode(line)

        face_img = cv2.imread(os.path.join(self.data.root, anno.face))
        face_img = self.transforms(face_img)                            # [c, h, w]
        left_img = cv2.imread(os.path.join(self.data.root, anno.lefteye))
        left_img = self.eye_transforms(left_img)                        # [c, h, w]
        right_img = cv2.imread(os.path.join(self.data.root, anno.righteye))
        right_img = self.eye_transforms(right_img)                      # [c, h, w]

        label_2d = np.array(anno.gaze2d.split(",")).astype("float")
        label_2d = torch.from_numpy(label_2d).type(torch.FloatTensor)

        label_3d = np.array(anno.gaze3d.split(",")).astype("float")
        label_3d = torch.from_numpy(label_3d).type(torch.FloatTensor)

        if 'llc' in anno:
            label_llc = np.array(anno.llc.split(",")).astype("float")
            label_llc = torch.from_numpy(label_llc).type(torch.FloatTensor)

            label_lrc = np.array(anno.lrc.split(",")).astype("float")
            label_lrc = torch.from_numpy(label_lrc).type(torch.FloatTensor)

            label_rlc = np.array(anno.rlc.split(",")).astype("float")
            label_rlc = torch.from_numpy(label_rlc).type(torch.FloatTensor)

            label_rrc = np.array(anno.rrc.split(",")).astype("float")
            label_rrc = torch.from_numpy(label_rrc).type(torch.FloatTensor)
        else:
            label_lcenter_2d = np.array(anno.lcenter_2d.split(",")).astype("float")
            label_lcenter_2d = torch.from_numpy(label_lcenter_2d).type(torch.FloatTensor)

            label_rcenter_2d = np.array(anno.rcenter_2d.split(",")).astype("float")
            label_rcenter_2d = torch.from_numpy(label_rcenter_2d).type(torch.FloatTensor)

        data = edict()
        data.face = face_img
        data.left = left_img
        data.right = right_img
        data.name = anno.name
        label = edict()
        label._2d = label_2d
        label._3d = label_3d

        if 'llc' in anno:
            label._llc = label_llc
            label._lrc = label_lrc
            label._rlc = label_rlc
            label._rrc = label_rrc
        else:
            label._lcenter_2d = label_lcenter_2d
            label._rcenter_2d = label_rcenter_2d
        return data, label


def loader(source, batch_size, shuffle=True, num_workers=0):
    dataset = Trainloader(source)
    print(f"-- [Read Data]: Source: {source.label}")
    print(f"-- [Read Data]: Total num: {len(dataset)}")
    load = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=True
    )
    return load
