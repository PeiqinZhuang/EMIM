from model import Uniformer


from collections import OrderedDict

import os
import cv2
import torch
import torchvision
import torch.nn
import torch.nn.functional as F
import argparse
import numpy as np
from PIL import Image
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from decord import VideoReader
from decord import cpu


import random
import pickle
import mmcv
from mmcv.fileio import FileClient


from transforms import GroupNormalize, GroupScale, GroupCenterCrop, Stack, ToTorchFormatTensor


def get_index(num_frames, num_segments, dense_sample_rate=8, method='dense'):
    if method == 'dense':
        sample_range = num_segments * dense_sample_rate
        sample_pos = max(1, 1 + num_frames - sample_range)
        t_stride = dense_sample_rate
        start_idx = 0 if sample_pos == 1 else sample_pos // 2
        offsets = np.array([
            (idx * t_stride + start_idx) %
            num_frames for idx in range(num_segments)
        ])
    else:
        if num_frames > num_segments:
            tick = num_frames / float(num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(num_segments)])
        else:
            offsets = np.zeros((num_segments,))
    return offsets


def load_images(video, selected_frames, transform1, transform2):
    crop_size = 224
    t_size = len(selected_frames)
    images = np.zeros((t_size, crop_size, crop_size, 3))
    orig_imgs = np.zeros_like(images)
    images_group = list()

    file_client = FileClient('petrel')
    for i, frame_index in enumerate(selected_frames):
        # img = Image.fromarray(video[frame_index].asnumpy())
        # images_group.append(img)
        img_str = file_client.get(os.path.join(video, "{:05d}.jpg".format(frame_index+1)))
        img = mmcv.imfrombytes(img_str)
        img = Image.fromarray(img)
        images_group.append(img)
        r_image = np.array(img)[:,:,::-1]
        orig_imgs[i] = transform2([Image.fromarray(r_image)])  
    torch_imgs = transform1(images_group)
    return np.expand_dims(orig_imgs, 0), torch_imgs


def get_img(index, length=8):
    path_prefix = 'XXXX/Sth2Sthv1_256/' 


    train_list = list()
    with open('/mnt/lustre/zhuangpeiqin.vendor/Annotation/Sthv1/data_lists/somethingv1_rgb_val.txt', 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip()
            train_list.append(line.split())
    path, num_frames, label = train_list[index]
    video_path = os.path.join(path_prefix, path)
    vr = video_path
    # vr = VideoReader(video_path, ctx=cpu(0))
    # num_frames = len(vr)
    num_frames = int(num_frames)

    # load image
    crop_size = 224
    scale_size = 256
    input_mean = [0.485, 0.456, 0.406]
    input_std = [0.229, 0.224, 0.225]

    input_mean2 = [0.5, 0.5, 0.5]
    input_std2 = [0.5, 0.5, 0.5]

    transform1 = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
        ToTorchFormatTensor(),
        GroupNormalize(input_mean, input_std),
    ])

    transform2 = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(),
    ])
    frame_indices = get_index(num_frames, length, method='sparse')
    RGB_vid, vid = load_images(vr, frame_indices, transform1, transform2)
    import pdb
    pdb.set_trace()
    
    return RGB_vid, vid, path, int(label)

def get_heatmap(RGB_vid, vid, label, downsample=True):
    # get predictions, last convolution output and the weights of the prediction layer
    model = Uniformer()
    # state_dict = torch.load('/mnt/cache/zhuangpeiqin.vendor/workspace/Transformer/temp_dir/uniformer_3d_no_temporal_reduction/video_classification/exp/uniformer_s16_sthv1_pre1k_uniformer_extra_attn64_ATTN_ATTN_7_7_LG_DROPPATH_02_0875_pretrained/checkpoints_588/checkpoint_epoch_00060.pyth', map_location='cpu')['model_state']
    state_dict = torch.load('/mnt/cache/zhuangpeiqin.vendor/workspace/Transformer/temp_dir/uniformer_3d_no_temporal_reduction/video_classification/exp/uniformer_s16_sthv1_pre1k/checkpoints/checkpoint_epoch_00060.pyth', map_location='cpu')['model_state']
    model.load_state_dict(state_dict)
    model = model.cuda()
    model.eval()
    predictions, y = model(vid.cuda())
    layerout = y[-1] # B, C, T, H, W

    layerout = layerout[0].detach().cpu().permute(1, 2, 3, 0)
    pred_weights = model.head.weight.data.detach().cpu().numpy().transpose()

    pred = torch.argmax(predictions).item()

    cam = np.zeros(dtype = np.float32, shape = layerout.shape[0:3])
    for i, w in enumerate(pred_weights[:, label]):
        # Compute cam for every kernel
        cam += w * layerout[:, :, :, i].numpy()

    # Resize CAM to frame level
    cam = zoom(cam, (1, 32, 32)) # output map is 8x7x7, so multiply to get to 8x224x224 (original image size)
    if not downsample:
        # interpolate
        tmp_cam = torch.from_numpy(cam)
        T, H, W = tmp_cam.shape
        tmp_cam = tmp_cam.view(T, H*W).permute(1, 0).unsqueeze(0)
        tmp_cam = torch.nn.functional.interpolate(tmp_cam, size=T*2, mode='linear')
        tmp_cam = tmp_cam.view(H, W, T*2).permute(2, 0, 1)
        cam = tmp_cam.numpy()

    # normalize
    cam -= np.min(cam)
    cam /= np.max(cam) - np.min(cam)

    heatmaps = []
    for i in range(0, cam.shape[0]):
        #   Create colourmap
        heatmap = cv2.applyColorMap(np.uint8(255*cam[i]), cv2.COLORMAP_JET)

        # Create frame with heatmap
        heatframe = heatmap//2 + RGB_vid[0][i]//2
        heatmaps.append(heatframe[:, :, ::-1]/255.)
        
    return heatmaps, pred

def show_img(index):
    RGB_vid, vid, path, label = get_img(index, length=16)
    print(RGB_vid.shape)
    
    TC, H, W = vid.shape
    inputs = vid.reshape(1, TC//3, 3, H, W).permute(0, 2, 1, 3, 4)
    
    # RGB_vid = RGB_vid[:, 1::2]
    heatmaps, pred = get_heatmap(RGB_vid, inputs, label=label)

    label_file = '/mnt/lustre/zhuangpeiqin.vendor/Annotation/Sthv1/data_lists/new_category.txt'
    category = []
    for x in open(label_file):
        category.append(x.rstrip().split('\t')[0])
    num_class = len(category)
    print("path: {}".format(path))
    print("Visualizing for class\t{}-{}".format(label, category[label]))
    print(("CT-Net predicted class\t{}-{}".format(pred, category[pred])))
    # plt.rcParams['savefig.dpi'] = 200 #图片像素
    # plt.rcParams['figure.dpi'] = 200 #分辨率
    # plt.figure()
    # gs=gridspec.GridSpec(1,8)
    # for i in range(1):
    #     for j in range(8):
    #         plt.subplot(gs[i,j])
    #         temp = RGB_vid[0][i*4+j]
    #         plt.imshow(temp[:,:,::-1]/255.)
    #         plt.axis('off')
    # plt.title('Origin')
    # plt.show()

    # plt.rcParams['savefig.dpi'] = 200 #图片像素
    # plt.rcParams['figure.dpi'] = 200 #分辨率
    # plt.figure()
    # gs=gridspec.GridSpec(1,8)
    # for i in range(1):
    #     for j in range(8):
    #         plt.subplot(gs[i,j])
    #         plt.imshow(heatmaps[i*8+j])
    #         plt.axis('off')
    # plt.title('CT-Net')
    # plt.show()
    fig, axes = plt.subplots(4, 8, figsize=(10,6))
    for i in range(2):
        for j in range(8):
            img = RGB_vid[0][i*8+j][:,:,::-1]
            img = (img - img.min()) / (img.max() - img.min())
            axes[i, j].imshow(img)
            axes[i, j].set_axis_off()
    # plt.savefig('./EMIM/Original_{}.jpg'.format(index))


    # fig, axes = plt.subplots(2, 8)
    for i in range(2):
        for j in range(8):
            axes[i+2, j].imshow(heatmaps[i*8+j])
            axes[i+2, j].set_axis_off()
    # plt.savefig('./EMIM/EMIM_{}.jpg'.format(index))
    plt.savefig('./Baseline/Baseline_{}.jpg'.format(index))

    # for i in range(8):
    #     plt.imshow(heatmaps[i])
    #     plt.axis('off')
    #     plt.savefig(f'./cam/{i}.png', bbox_inches='tight', dpi=300)
    #     plt.show()


def main():
    # for i in range(809):
    #     show_img(i)
    # show_img(546)
    get_img(546, 16)

if __name__ == '__main__':
    main()