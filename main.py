"""
(Training, Generating edge maps)
Pixel Difference Networks for Efficient Edge Detection (accepted as an ICCV 2021 oral)
See paper in https://arxiv.org/abs/2108.07009

Author: Zhuo Su, Wenzhe Liu
Date: Aug 22, 2020

"""

from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import print_function
from __future__ import division

import argparse
import os
import time
import models
from models.convert_pidinet import convert_pidinet
from torch.utils import data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
from PIL import Image
import numpy as np
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='PyTorch Pixel Difference Convolutional Networks')
args = parser.parse_args()

args.seed = int(time.time())
args.use_cuda = torch.cuda.is_available()
args.gpu = 'cpu'
args.datadir = 'samples'
args.model = 'pidinet_converted'
args.config = 'carv4'
args.sa = True
args.dil = True
args.evaluate = 'table5_pidinet.pth'
args.evaluate_converted = True
args.workers = 4
args.savedir = 'results'

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu


def fold_files(foldname):
    """All files in the fold should have the same extern"""
    allfiles = os.listdir(foldname)
    if len(allfiles) < 1:
        raise ValueError('No images in the data folder')
        return None
    else:
        return allfiles


class Custom_Loader(data.Dataset):
    """
    Custom Dataloader
    """

    def __init__(self, root='data/'):
        self.root = root
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])
        self.imgList = fold_files(os.path.join(root))

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, index):
        with open(os.path.join(self.root, self.imgList[index]), 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        img = self.transform(img)

        filename = Path(self.imgList[index]).stem

        return img, filename


def main():
    global args

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    ### Create model
    model = getattr(models, args.model)(args.config, args.sa, args.dil)

    ### Transfer to cuda devices
    if args.use_cuda:
        model = torch.nn.DataParallel(model).cuda()
        print('cuda is used, with %d gpu devices' % torch.cuda.device_count())
    else:
        print('cuda is not used, the running might be slow')

    # cudnn.benchmark = True

    ### Load Data
    test_dataset = Custom_Loader(root=args.datadir)

    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=args.workers, shuffle=False)

    args.start_epoch = 0

    ### Evaluate directly if required
    model_filename = args.evaluate
    checkpoint = torch.load(model_filename, map_location='cpu')
    loadinfo2 = "=> loaded checkpoint '{}' successfully".format(model_filename)
    print(loadinfo2)

    if checkpoint is not None:
        args.start_epoch = checkpoint['epoch'] + 1
        if args.evaluate_converted:
            model.load_state_dict(convert_pidinet(checkpoint['state_dict'], args.config))
        else:
            model.load_state_dict(checkpoint['state_dict'])
    else:
        raise ValueError('no checkpoint loaded')
    test(test_loader, model, args.start_epoch, args)
    print('##########Time########## %s' % (time.strftime('%Y-%m-%d %H:%M:%S')))
    return


def test(test_loader, model, epoch, args):
    from PIL import Image
    import scipy.io as sio
    model.eval()
    img_dir = os.path.join(args.savedir, 'eval_results', 'imgs_epoch_%03d' % (epoch - 1))
    mat_dir = os.path.join(args.savedir, 'eval_results', 'mats_epoch_%03d' % (epoch - 1))
    eval_info = '\nBegin to eval...\nImg generated in %s\n' % img_dir
    print(eval_info)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    else:
        print('%s already exits' % img_dir)
        # return
    if not os.path.exists(mat_dir):
        os.makedirs(mat_dir)

    for idx, (image, img_name) in enumerate(test_loader):

        img_name = img_name[0]
        with torch.no_grad():
            image = image.cuda() if args.use_cuda else image
            _, _, H, W = image.shape
            results = model(image)
            result = torch.squeeze(results[-1]).cpu().numpy()

        results_all = torch.zeros((len(results), 1, H, W))
        for i in range(len(results)):
            results_all[i, 0, :, :] = results[i]

        torchvision.utils.save_image(1 - results_all,
                                     os.path.join(img_dir, "%s.jpg" % img_name))
        sio.savemat(os.path.join(mat_dir, '%s.mat' % img_name), {'img': result})
        result = Image.fromarray((result * 255).astype(np.uint8))
        result.save(os.path.join(img_dir, "%s.png" % img_name))
        runinfo = "Running test [%d/%d]" % (idx + 1, len(test_loader))
        print(runinfo)


if __name__ == '__main__':
    main()
    print('done')
