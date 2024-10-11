import warnings
import CLIP_.clip as clip
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import argparse
import math
import os
import xlwt
import sys
import time
import traceback
import numpy as np
import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from PIL import Image
from torchvision import models, transforms
from tqdm.auto import tqdm, trange
import dlib
import config
import sketch_utils as utils
from models.loss import Loss, LPIPS
from models.painter_params import XDoG_, Painter, PainterOptimizer, interpret
from IPython.display import display, SVG
from torchvision.utils import save_image
import cv2
import random
from collections import OrderedDict
def load_renderer(args, target_im=None, mask=None):
    renderer = Painter(num_strokes=args.num_strokes, args=args,
                       num_segments=args.num_segments,
                       imsize=args.image_scale,
                       device=args.device,
                       target_im=target_im)
    renderer = renderer.to(args.device)
    return renderer


def get_target(args):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_

def get_target_(args):
    target = Image.open(args.mask)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    if args.fix_scale:
        target = utils.fix_image_scale(target)

    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_

def get_roi(args):
    roi_ = []
    for round in range (args.region_round - 1):
        roi = f"./2-target_images/{args.target_file[:-4]}_roi{round+1}{args.target_file[-4:]}"
        print("roi_name===",roi)
        roi = Image.open(roi)
        if roi.mode == "RGBA":
            # Create a white rgba background
            new_image = Image.new("RGBA", roi.size, "WHITE")
            # Paste the image on the background.
            new_image.paste(roi, (0, 0), roi)
            roi = new_image
        roi = roi.convert("RGB")

        if args.fix_scale:
            target = utils.fix_image_scale(roi)

        transforms_ = []
        if roi.size[0] != roi.size[1]:
            transforms_.append(transforms.Resize(
                (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
        else:
            transforms_.append(transforms.Resize(
                args.image_scale, interpolation=PIL.Image.BICUBIC))
            transforms_.append(transforms.CenterCrop(args.image_scale))
        transforms_.append(transforms.ToTensor())
        data_transforms = transforms.Compose(transforms_)
        roi_.append(data_transforms(roi).unsqueeze(0).to(args.device))
    return roi_

def get_target_2(path):
    target = Image.open(path)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")

    transforms_ = []
    # if target.size[0] != target.size[1]:
    #     transforms_.append(transforms.Resize(
    #         (args.image_scale/2, args.image_scale), interpolation=PIL.Image.BICUBIC))
    # else:
    #     transforms_.append(transforms.Resize(
    #         args.image_scale/2, interpolation=PIL.Image.BICUBIC))
    #     transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    target_ = data_transforms(target).unsqueeze(0).to(args.device)
    return target_

def get_masked(args,mask_path):
    target = Image.open(args.target)
    if target.mode == "RGBA":
        # Create a white rgba background
        new_image = Image.new("RGBA", target.size, "WHITE")
        # Paste the image on the background.
        new_image.paste(target, (0, 0), target)
        target = new_image
    target = target.convert("RGB")
    transforms_ = []
    if target.size[0] != target.size[1]:
        transforms_.append(transforms.Resize(
            (args.image_scale, args.image_scale), interpolation=PIL.Image.BICUBIC))
    else:
        transforms_.append(transforms.Resize(
            args.image_scale, interpolation=PIL.Image.BICUBIC))
        transforms_.append(transforms.CenterCrop(args.image_scale))
    transforms_.append(transforms.ToTensor())
    data_transforms = transforms.Compose(transforms_)
    # target_ = data_transforms(target).unsqueeze(0).to(args.device)
    target_ = data_transforms(target).to(args.device).permute(1,2,0)

    masked_im= utils.get_mask_img(args,target_ ,mask_path)

    masked_im = data_transforms(masked_im).unsqueeze(0).to(args.device)

    return masked_im

def tensor_to_image(tensor):
    if tensor.dim()==4:
        tensor=tensor.squeeze(0)  ###去掉batch维度
    tensor=tensor.permute(1,2,0) ##将c,h,w 转换为h,w,c
    tensor=tensor.mul(255).clamp(0,255)  ###将像素值转换为0-255之间
    tensor=tensor.cpu().detach().numpy().astype('uint8')  ###
    tensor=Image.fromarray(np.uint8(tensor))
    return tensor

def fps(points, num_points):
    """
    FPS采样算法实现
    Args:
        points: 点集，N x 2的二维数组，每行表示一个点的坐标
        num_points: 采样数量
    Returns:
        采样点的索引列表
    """
    n = len(points)
    distances = np.full(n, np.inf)  # 初始化每个点到已采样点集的最短距离为无穷大
    # distances = 0  # 初始化每个点到已采样点集的距离为0
    samples = []  # 采样点索引集
    samples_points = []  # 采样点集
    current = np.random.randint(n)  # 随机选择一个起始点索引
    samples.append(current)  # 将起始点索引加入采样点索引集
    samples_points.append(points[current+10])  # 将起始点加入采样点集

    while len(samples) < num_points:
        # 计算每个点到已选点集的最短距离
        for i in range(n):
            if i in samples:
                distances[i] = 0
            else:
                dist = np.linalg.norm(points[i] - points[current])
                distances[i] = min(dist, distances[i])
        # 找到距离已选点集最远的点，将它添加到采样点集中
        farthest = np.argmax(distances)#np.argmax() 是NumPy库中的一个函数，用于返回数组中最大元素的索引
        samples.append(farthest)
        samples_points.append(points[farthest])
        current = farthest
        distances[current] = np.inf  # 将新选点的最短距离设为无穷大

    return np.array(samples_points)

def get_edge_points(image):
    image = image.detach().cpu().numpy()
    image = (image * 255).astype(np.uint8)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.blur(gray, (3, 3))
    # 使用 Canny 边缘检测器检测线条
    edges = cv2.Canny(image, 20, 100)
    edge_points = np.argwhere(edges > 0)  # 检测的边缘上的点
    # 如果没有检测到边缘，返回空列表
    if edge_points.size == 0:
        return []
    cv2.imwrite("{}/edge_image.png".format(args.output_dir), edges)
    return edge_points
def extract_line_points(image_path, num_points):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224,224))
    # image = image.detach().cpu().numpy()
    # image = (image * 255).astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.blur(gray, (3, 3))
    # 使用 Canny 边缘检测器检测线条
    edges = cv2.Canny(blur, 20, 200)
    edge_points = np.argwhere(edges > 0)  # 检测的边缘上的点
    # 如果没有检测到边缘，返回空列表
    if edge_points.size == 0:
        return []
    cv2.imwrite("{}/edge_image.png".format(args.output_dir), edges)
    selected_points = fps(edge_points, num_points)
    selected_points = np.array(selected_points)
    print("selected_points2",selected_points)
    return selected_points



def extract_global_points(image_path, num_points):
    # 读取图像并转换为灰度图
    print('image_path===',image_path)
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224,224))
    # image = image.detach().cpu().numpy()
    # image = (image * 255).astype(np.uint8)
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # blur = cv2.blur(gray, (3, 3))
    # 使用 Canny 边缘检测器检测线条
    edges = cv2.Canny(image, 50, 250)
    edge_points = np.argwhere(edges > 0)  # 检测的边缘上的点
    # 如果没有检测到边缘，返回空列表
    if edge_points.size == 0:
        return []
    cv2.imwrite("{}/edge_image.png".format(args.output_dir), edges)
    if num_points == 0:
        selected_points = []
    else:
        selected_points = fps(edge_points, num_points)
    selected_points = np.array(selected_points)
    print("selected_points2",selected_points)
    return selected_points

def extract_region_points(args, num_points):
    region_points = []
    # 读取图像并转换为灰度图
    for round in range (args.region_round - 1):
        roi_path = f"./2-target_images/{args.target_file[:-4]}_roi{round+1}{args.target_file[-4:]}"
        image = cv2.imread(roi_path)
        image = cv2.resize(image, (224, 224))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (3, 3))
        # 使用 Canny 边缘检测器检测线条
        edges = cv2.Canny(blur, 20, 200)
        edge_points = np.argwhere(edges > 0)  # 检测的边缘上的点
        # 如果没有检测到边缘，返回空列表
        if edge_points.size == 0:
            return []
        cv2.imwrite("{}/edge_image.png".format(args.output_dir), edges)
        if num_points[round] == 0:
            selected_points = []
        else:
            selected_points = fps(edge_points, num_points[round])
        region_points.append(np.array(selected_points))
    print("region_points",region_points)
    return region_points

def get_edge_pathnum(args):
    # 读取图像并转换为灰度图
    target = cv2.imread(args.target)
    target = cv2.resize(target, (224, 224))
    # image = image.detach().cpu().numpy()
    # image = (image * 255).astype(np.uint8)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target_blur = cv2.blur(target_gray, (3, 3))
    # 使用 Canny 边缘检测器检测线条
    target_edges = cv2.Canny(target_blur, 20, 200)
    target_edge_points = np.argwhere(target_edges > 0)  # 检测的边缘上的点

    cv2.imwrite("{}/target_edge_image.png".format(args.output_dir), target_edges)
    len_target = target_edge_points.size
    print("Number of target edge:", len_target)

    len_roi= []
    roi_pathnum= []
    for i in range(args.region_round - 1):
        roi = f"./2-target_images/{args.target_file[:-4]}_roi{i+1}{args.target_file[-4:]}"
        # 读取图像并转换为灰度图
        roi = cv2.imread(roi)
        roi = cv2.resize(roi, (224, 224))
        # image = image.detach().cpu().numpy()
        # image = (image * 255).astype(np.uint8)
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_blur = cv2.blur(roi_gray, (3, 3))
        # 使用 Canny 边缘检测器检测线条
        roi_edges = cv2.Canny(roi_blur, 20, 200)
        roi_edge_points = np.argwhere(roi_edges > 0)  # 检测的边缘上的点
        cv2.imwrite("{}/roi{}_edge_image.png".format(args.output_dir, i), roi_edges)
        len_roi.append(roi_edge_points.size)

    print("Number of roi edge:", len_roi)
    target_roi = len_target / (len_target + sum(len_roi))
    print("target_roi:", target_roi)
    target_pathnum = round(target_roi * args.num_strokes)
    for i in range(args.region_round - 1):
        roi_roi = len_roi[i] / (len_target + sum(len_roi))
        roi_pathnum.append(round(roi_roi * args.num_strokes))

    pathsum = target_pathnum + sum(roi_pathnum)
    #调整笔画数
    if pathsum > args.num_strokes:
        target_pathnum = target_pathnum - (pathsum - args.num_strokes)
    return target_pathnum, roi_pathnum

def get_pixel_pathnum(args):
    # Load image
    target = cv2.imread(args.target)
    target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    target_thresh = cv2.threshold(target_gray, 200, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(r"{}/target_thresh.png".format(args.output_dir), cv2.bitwise_not(target_thresh))
    target_pixels = cv2.countNonZero(cv2.bitwise_not(target_thresh))
    print("Number of target pixels:", target_pixels)

    roi_pixels= []
    roi_pathnum= []
    for i in range(args.region_round - 1):
        roi = f"./2-target_images/{args.target_file[:-4]}_roi{i+1}{args.target_file[-4:]}"
        roi_round = cv2.imread(roi)
        roi_gray = cv2.cvtColor(roi_round, cv2.COLOR_BGR2GRAY)

        roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)[1]
        cv2.imwrite(r"{}/{}_roi_thresh.png".format(round, args.output_dir), cv2.bitwise_not(roi_thresh))

        roi_pixels.append(cv2.countNonZero(cv2.bitwise_not(roi_thresh)))

    print("Number of roi pixels:", roi_pixels)
    target_roi = target_pixels / (target_pixels + sum(roi_pixels))
    print("target_roi:", target_roi)
    target_pathnum = round(target_roi * args.num_strokes)
    for i in range(args.region_round - 1):
        roi_roi = roi_pixels[i] / (target_pixels + sum(roi_pixels))
        roi_pathnum.append(round(roi_roi * args.num_strokes))

    pathsum = target_pathnum + sum(roi_pathnum)
    #调整笔画数
    if pathsum > args.num_strokes:
        target_pathnum = target_pathnum - (pathsum - args.num_strokes)
    return target_pathnum, roi_pathnum
def softmax(x, tau=0.2):
    e_x = np.exp(x / tau)
    return e_x / e_x.sum()

def get_new_points(args, target_im):
    model, preprocess = clip.load(args.saliency_clip_model, device=args.device, jit=False)
    model.eval().to(args.device)

    data_transforms = transforms.Compose([
        preprocess.transforms[-1],
    ])
    image_input_attn_clip = data_transforms(target_im).to(args.device)

    attention_map = interpret(image_input_attn_clip, model, device=args.device)
    del model
    attn_map = (attention_map - attention_map.min()) / (
            attention_map.max() - attention_map.min())
    print("attn_map=========", type(attn_map))
    print("attn_map=========", attn_map.shape)
    if args.xdog_intersec:
        xdog = XDoG_()
        im_xdog = xdog(image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=15)
        intersec_map = (1 - im_xdog) * attn_map
        attn_map = intersec_map
    print("attn_map2=========", type(attn_map))
    print("attn_map2=========", attn_map.shape)
    cv2.imwrite(r"{}/attn_map.png".format(args.output_dir), attn_map)

    attn_map_soft = np.copy(attn_map)
    attn_map_soft[attn_map > 0] = softmax(attn_map[attn_map > 0], tau=args.softmax_temp)

    # k = int((self.num_stages * self.num_paths)/self.masknum)
    numsize = 1
    print("numsize========",numsize)
    k = args.num_stages * numsize
    print("range(attn_map.flatten().shape[0])======",range(attn_map.flatten().shape[0]))
    print("attn_map_soft.flatten()======", attn_map_soft.flatten())

    inds = np.random.choice(range(attn_map.flatten().shape[0]), size=1, replace=False,
                                     p=np.max(attn_map_soft.flatten()))
    inds = np.array(np.unravel_index(inds, attn_map.shape)).T
   
    # points, edge_points = extract_line_points(args.target, args.num_paths)
    # print("points====",points)
    inds_normalised_record = np.zeros(inds.shape)
    inds_normalised_record[:, 0] = inds[:, 1] / args.image_scale
    inds_normalised_record[:, 1] = inds[:, 0] / args.image_scale
    inds_normalised_record = inds_normalised_record.tolist()
    return inds_normalised_record

def random_points(image_path, num_points):
    # 打开图像
    image = Image.open(image_path)

    # width, height = image.size
    region_points = []
    # 创建一个新的图像，用于绘制点
    points_image = Image.new('RGB', (224, 224))

    # 生成随机坐标并绘制点
    for _ in range(num_points):
        x = random.randint(0, 224 - 1)
        y = random.randint(0, 224 - 1)
        region_points.append([x,y])

    return region_points

def main(args):
    loss_func = Loss(args)
    loss_lpips = LPIPS(args).to(args.device)
    inputs_ = get_target(args)
    masks= get_target_(args)
    # roi_ = get_roi(args)
    inputs = inputs_
    # face_masked = utils.get_mask_img(inputs, './target_images/00017_skin.png')
    # face_masked.save("{}/face_masked.png".format(args.output_dir))
    # inputs, face_inds, hull_mask_img, inversemask_img = get_points(args)
    utils.log_input(args.use_wandb, 0, inputs_, args.output_dir)
    # renderer = load_renderer(args, inputs)
    # target_pathnum, roi_pathnum = get_edge_pathnum(args)
    # print("target_pathnum===",target_pathnum)
    # print("roi_pathnum===",roi_pathnum)
    renderer = load_renderer(args, inputs)
    optimizer = PainterOptimizer(args, renderer)
    counter = 0
    configs_to_save = {"loss_eval": []}
    if args.num_global_paths != 0 :
        # Attention
        # global_inds = renderer.get_attn_global_points()
        #FPS
        # global_inds = extract_global_points(args.target, args.num_global_paths)
        global_inds = extract_global_points(args.mask, args.num_global_paths)
        #random
        # global_inds = random_points(args.target, args.num_global_paths)
        global_inds = np.array(global_inds)
    else :
        global_inds = []
    if args.num_face_paths != 0:
        # random
        # face_inds = random_points(args.target, args.num_face_paths)
        # face_inds = np.array(face_inds)
        #Attention
        # face_inds = renderer.get_attn_global_points(args.output_dir)
        # face_inds = np.array(face_inds)
        # FPS
        # face_inds = extract_global_points(args.target,args.num_face_paths)
        # Face
        face_inds = renderer.get_face_points(args.target, args.output_dir)
    else :
        face_inds = []
    # face_inds, hair_inds, attn_map = renderer.get_points(args.target, args.output_dir)

    utils.plot_global_geo(inputs, np.array(global_inds), "{}/{}_.jpg".format(
        args.output_dir, "global_points_map"))

    utils.plot_face_clip(inputs, np.array(face_inds), "{}/{}_.jpg".format(
        args.output_dir, "face_points_map"))

    utils.plot_target_map(inputs, np.array(global_inds), np.array(face_inds), "{}/{}_.jpg".format(
        args.output_dir, "all_points_map"))


    inds_num = 0
    for record in range(2):
        if record == 0:
            inds_record = face_inds
            num_paths = args.num_face_paths
            if num_paths == 0:
                continue
        else:
            inds_record = global_inds
            num_paths = args.num_global_paths
            if num_paths == 0:
                continue


        inds_num += num_paths
        inds_normalised_record = np.zeros(inds_record.shape)
        inds_normalised_record[:, 0] = inds_record[:, 1] / args.image_scale
        inds_normalised_record[:, 1] = inds_record[:, 0] / args.image_scale
        inds_normalised_record = inds_normalised_record.tolist()

        renderer.set_random_noise(0)
        renderer.init_image_1(inds_normalised_record, num_paths)
        optimizer.init_optimizers()

        # not using tdqm for jupyter demo
        if args.display:
            epoch_range = range(args.num_iter)
        else:
            epoch_range = tqdm(range(args.num_iter))


        for epoch in epoch_range:
            if not args.display:
                epoch_range.refresh()
            renderer.set_random_noise(epoch)
            if args.lr_scheduler:
                optimizer.update_lr(counter)

            start = time.time()
            optimizer.zero_grad_()
            # sketches = renderer.get_image(inds_normalised_record).to(args.device)
            sketches = renderer.get_image().to(args.device)
            loss_crop = 0
            crop_ = 0
            if not os.path.isdir("{}/sketches_region".format(args.output_dir)):
                os.makedirs("{}/sketches_region".format(args.output_dir))
            if not os.path.isdir("{}/inputs_region".format(args.output_dir)):
                os.makedirs("{}/inputs_region".format(args.output_dir))
            if args.crop_object:
                x = 0
                y = 0
                # 裁剪
                w = int(args.image_scale / args.crop_scale)
                h = int(args.image_scale / args.crop_scale)
                sketches_img = tensor_to_image(sketches)
                inputs_img = tensor_to_image(inputs)
                for k in range(args.crop_scale):
                    for v in range(args.crop_scale):
                        sketches_region = sketches_img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
                        inputs_region = inputs_img.crop((x + k * w, y + v * h, x + w * (k + 1), y + h * (v + 1)))
                        #####保存图片的位置以及图片名称###############
                        sketches_region.save("{}/sketches_region/iter_{}_{}.png".format(args.output_dir, k, v))
                        inputs_region.save("{}/inputs_region/iter_{}_{}.png".format(args.output_dir, k, v))
                        sketches_region = get_target_2("{}/sketches_region/iter_{}_{}.png".format(args.output_dir, k, v))
                        inputs_region = get_target_2("{}/inputs_region/iter_{}_{}.png".format(args.output_dir, k, v))
                        crop_ += (inputs_region - sketches_region).pow(2).mean()
                        
            loss_crop = crop_
            # loss_crop = crop_ / (args.crop_scale * args.crop_scale)
            loss_crop1 = (inputs - sketches).pow(2).mean()

            losses_dict = loss_func(sketches, inputs.detach(
            ), renderer.get_color_parameters(), renderer, counter, optimizer)
            loss_s = sum(list(losses_dict.values()))

            loss_l = loss_lpips(sketches, masks.detach())
            loss_l = torch.mean(loss_l)
            
            if record ==0:
                loss = 1* loss_s + 0 * loss_l + (0 * loss_crop1 + 0 * loss_crop)
            else:
                loss = 0 * loss_s + 1 * loss_l + (1 * loss_crop1 + 1 * loss_crop)

            loss.backward()
            optimizer.step_()
            if epoch % args.save_interval == 0:
                save_image(sketches, "{}/{}_iter_{}.png".format(args.output_dir, record, epoch))
                renderer.save_svg(
                    f"{args.output_dir}/svg_logs", f"{record}_svg_iter{epoch}")

            counter += 1


    renderer.save_svg(args.output_dir, "final_svg")
    return configs_to_save

if __name__ == "__main__":
    args = config.parse_arguments()
    final_config = vars(args)
    try:
        configs_to_save = main(args)
    except BaseException as err:
        print(f"Unexpected error occurred:\n {err}")
        print(traceback.format_exc())
        sys.exit(1)
    for k in configs_to_save.keys():
        final_config[k] = configs_to_save[k]
    np.save(f"{args.output_dir}/config.npy", final_config)
    if args.use_wandb:
        wandb.finish()