import os

import cv2.dnn
import imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pydiffvg
import skimage
import skimage.io
import torch
import wandb
import PIL
import random
from PIL import Image
from torchvision import transforms
from torchvision.utils import make_grid
from skimage.transform import resize
import os.path as osp
from U2Net_.model import U2NET
from torchvision.utils import save_image

def imwrite(img, filename, gamma=2.2, normalize=False, use_wandb=False, wandb_name="", step=0, input_im=None):
    directory = os.path.dirname(filename)
    if directory != '' and not os.path.exists(directory):
        os.makedirs(directory)

    if not isinstance(img, np.ndarray):
        img = img.data.numpy()
    if normalize:
        img_rng = np.max(img) - np.min(img)
        if img_rng > 0:
            img = (img - np.min(img)) / img_rng
    img = np.clip(img, 0.0, 1.0)
    if img.ndim == 2:
        # repeat along the third dimension
        img = np.expand_dims(img, 2)
    img[:, :, :3] = np.power(img[:, :, :3], 1.0/gamma)
    img = (img * 255).astype(np.uint8)

    skimage.io.imsave(filename, img, check_contrast=False)
    images = [wandb.Image(Image.fromarray(img), caption="output")]
    if input_im is not None and step == 0:
        images.append(wandb.Image(input_im, caption="input"))
    if use_wandb:
        wandb.log({wandb_name + "_": images}, step=step)


def plot_batch(inputs, outputs, output_dir, step, use_wandb, title):
    plt.figure()
    plt.subplot(2, 1, 1)
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("inputs")

    plt.subplot(2, 1, 2)
    grid = make_grid(outputs, normalize=False, pad_value=2)
    npgrid = grid.detach().cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title("outputs")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"output": wandb.Image(plt)}, step=step)
    plt.savefig("{}/{}".format(output_dir, title))
    plt.close()


def log_input(use_wandb, epoch, inputs, output_dir):
    grid = make_grid(inputs.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"input": wandb.Image(plt)}, step=epoch)
    plt.close()
    input_ = inputs[0].cpu().clone().detach().permute(1, 2, 0).numpy()
    input_ = (input_ - input_.min()) / (input_.max() - input_.min())
    input_ = (input_ * 255).astype(np.uint8)
    imageio.imwrite("{}/{}.png".format(output_dir, "input"), input_)


def log_sketch_summary_final(path_svg, use_wandb, device, epoch, loss, title):
    canvas_width, canvas_height, shapes, shape_groups = load_svg(path_svg)
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)

    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    plt.imshow(img.cpu().numpy())
    plt.axis("off")
    plt.title(f"{title} best res [{epoch}] [{loss}.]")
    if use_wandb:
        wandb.log({title: wandb.Image(plt)})
    plt.close()


def log_sketch_summary(sketch, title, use_wandb):
    plt.figure()
    grid = make_grid(sketch.clone().detach(), normalize=True, pad_value=2)
    npgrid = grid.cpu().numpy()
    plt.imshow(np.transpose(npgrid, (1, 2, 0)), interpolation='nearest')
    plt.axis("off")
    plt.title(title)
    plt.tight_layout()
    if use_wandb:
        wandb.run.summary["best_loss_im"] = wandb.Image(plt)
    plt.close()


def load_svg(path_svg):
    svg = os.path.join(path_svg)
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        svg)
    return canvas_width, canvas_height, shapes, shape_groups


def read_svg(path_svg, device, multiply=False):
    canvas_width, canvas_height, shapes, shape_groups = pydiffvg.svg_to_scene(
        path_svg)
    if multiply:
        canvas_width *= 2
        canvas_height *= 2
        for path in shapes:
            path.points *= 2
            path.stroke_width *= 2
    _render = pydiffvg.RenderFunction.apply
    scene_args = pydiffvg.RenderFunction.serialize_scene(
        canvas_width, canvas_height, shapes, shape_groups)
    img = _render(canvas_width,  # width
                  canvas_height,  # height
                  2,   # num_samples_x
                  2,   # num_samples_y
                  0,   # seed
                  None,
                  *scene_args)
    img = img[:, :, 3:4] * img[:, :, :3] + \
        torch.ones(img.shape[0], img.shape[1], 3,
                   device=device) * (1 - img[:, :, 3:4])
    img = img[:, :, :3]
    return img


def plot_global_geo(inputs, hair_inds, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(4, 4))

    plt.subplot()
    inputs = make_grid(inputs, normalize=True, pad_value=0)
    inputs = np.transpose(inputs.cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(inputs, interpolation='nearest', vmin=0, vmax=0)
    if len(hair_inds) != 0:
        # plt.scatter(hair_inds[:, 1], hair_inds[:, 0], s=40, c='red', marker='o')
        plt.scatter(hair_inds[:, 1], hair_inds[:, 0], s=40, c='red', marker='o')

    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

def plot_face_clip(inputs, hair_inds, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(4, 4))

    plt.subplot()
    inputs = make_grid(inputs, normalize=True, pad_value=0)
    inputs = np.transpose(inputs.cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(inputs, interpolation='nearest', vmin=0, vmax=0)
    if len(hair_inds) != 0:
        # plt.scatter(hair_inds[:, 1], hair_inds[:, 0], s=40, c='red', marker='o')
        plt.scatter(hair_inds[:, 1], hair_inds[:, 0], s=40, c='green', marker='o')

    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_target_map(inputs, global_inds, face_inds, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(4, 4))

    plt.subplot()
    inputs = make_grid(inputs, normalize=True, pad_value=0)
    inputs = np.transpose(inputs.cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(inputs, interpolation='nearest', vmin=0, vmax=0)
    if len(global_inds) != 0:
        # plt.scatter(hair_inds[:, 1], hair_inds[:, 0], s=40, c='red', marker='o')
        plt.scatter(global_inds[:, 1], global_inds[:, 0], s=40, c='red', marker='o')
    if len(face_inds) != 0:
        # plt.scatter(hair_inds[:, 1], hair_inds[:, 0], s=40, c='red', marker='o')
        plt.scatter(face_inds[:, 1], face_inds[:, 0], s=40, c='green', marker='o')

    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()



def plot_roi_clip(inputs, face_inds, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(4, 4))

    plt.subplot()
    inputs = make_grid(inputs, normalize=True, pad_value=0)
    inputs = np.transpose(inputs.cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(inputs, interpolation='nearest', vmin=0, vmax=0)
    if len(face_inds) != 0:
        # plt.scatter(hair_inds[:, 1], hair_inds[:, 0], s=40, c='red', marker='o')
        plt.scatter(face_inds[:, 1], face_inds[:, 0], s=40, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_attn_clip_2(attn, threshold_map, inds, inputs, sketches, use_wandb, output_path, display_logs):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(20, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 4, 2)
    inputs = make_grid(inputs, normalize=True, pad_value=2)
    inputs = np.transpose(inputs.cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(inputs, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")

    plt.subplot(1, 4, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
                     (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("inds")
    if len(inds) != 0:
        # plt.scatter(hair_inds[:, 1], hair_inds[:, 0], s=40, c='red', marker='o')
        plt.scatter(inds[:, 1], inds[:, 0], s=30, c='red', marker='o')
    plt.axis("off")

    plt.subplot(1, 4, 4)
    sketches_im = make_grid(sketches, normalize=True, pad_value=2)
    sketches_im = np.transpose(sketches_im.cpu().detach().numpy(), (1, 2, 0))
    plt.imshow(sketches_im, interpolation='nearest', vmin=0, vmax=1)
    plt.title("sketches")
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()

def plot_attn_clip_(attn, threshold_map, inputs, inds, use_wandb, output_path, display_logs):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='blue', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
        (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()

# def plot_atten_(attn, threshold_map, inputs, inds, use_wandb, output_path, saliency_model, display_logs):
#     if saliency_model == "dino":
#         plot_attn_dino(attn, threshold_map, inputs,
#                        inds, use_wandb, output_path)
#     elif saliency_model == "clip":
#         plot_attn_clip_(attn, threshold_map, inputs, inds,
#                        use_wandb, output_path, display_logs)

# def plot_atten_2(attn, threshold_map, points, inputs, sketches, use_wandb, output_path, saliency_model, display_logs):
#     if saliency_model == "dino":
#         plot_attn_dino(attn, threshold_map, points, inputs, sketches,use_wandb, output_path, display_logs)
#     elif saliency_model == "clip":
#         plot_attn_clip_2(attn, threshold_map, points, inputs, sketches,use_wandb, output_path, display_logs)
#
# def plot_atten(target_pathnum, roi_pathnum, inputs, sketches, face_inds, hair_inds, use_wandb, output_path, saliency_model, display_logs):
#     if saliency_model == "dino":
#         plot_attn_dino(inputs, sketches,face_inds, hair_inds, use_wandb, output_path, display_logs)
#     elif saliency_model == "clip":
#         plot_attn_clip(target_pathnum, roi_pathnum, inputs, sketches,face_inds, hair_inds, use_wandb, output_path, display_logs)
#

def fix_image_scale(im):
    im_np = np.array(im) / 255
    height, width = im_np.shape[0], im_np.shape[1]
    max_len = max(height, width) + 20
    new_background = np.ones((max_len, max_len, 3))
    y, x = max_len // 2 - height // 2, max_len // 2 - width // 2
    new_background[y: y + height, x: x + width] = im_np
    new_background = (new_background / new_background.max()
                      * 255).astype(np.uint8)
    new_im = Image.fromarray(new_background)
    return new_im

# min(320, im_size
def get_mask_u2net(args, pil_im):
    w, h = pil_im.size[0], pil_im.size[1]
    im_size = min(w, h)
    data_transforms = transforms.Compose([
        transforms.Resize(min(320, im_size), interpolation=PIL.Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(
            0.26862954, 0.26130258, 0.27577711)),
    ])

    input_im_trans = data_transforms(pil_im).unsqueeze(0).to(args.device)

    model_dir = os.path.join("./U2Net_/saved_models/u2net.pth")
    net = U2NET(3, 1)
    if torch.cuda.is_available() and args.use_gpu:
        net.load_state_dict(torch.load(model_dir))
        net.to(args.device)
    else:
        net.load_state_dict(torch.load(model_dir, map_location='cpu'))
    net.eval()
    with torch.no_grad():
        d1, d2, d3, d4, d5, d6, d7 = net(input_im_trans.detach())
    pred = d1[:, 0, :, :]
    pred = (pred - pred.min()) / (pred.max() - pred.min())
    predict = pred
    predict[predict < 0.5] = 0
    predict[predict >= 0.5] = 1
    mask = torch.cat([predict, predict, predict], axis=0).permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask = resize(mask, (h, w), anti_aliasing=False)
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    # print("mask1==========", type(mask))
    # # predict_np = predict.clone().cpu().data.numpy()
    # mask = cv2.imread("./mask/woman12_mask_1.png")
    # print("mask2==========", type(mask))
    # mask = resize(mask, (h, w), anti_aliasing=False)
    im = Image.fromarray((mask[:, :, 0]*255).astype(np.uint8)).convert('RGB')
    # print("im==========",im.size)
    im.save(f"{args.output_dir}/mask.png")

    im_np = np.array(pil_im)
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1
    im_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_final = Image.fromarray(im_final)
    im_final.save(f"{args.output_dir}/mask_img.png")

    return im_final, predict

def get_image_augmentation(use_normalized_clip):
    augment_trans = transforms.Compose([
        transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
        transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
    ])

    if use_normalized_clip:
        augment_trans = transforms.Compose([
            transforms.RandomPerspective(fill=1, p=1, distortion_scale=0.5),
            transforms.RandomResizedCrop(224, scale=(0.7, 0.9)),
            transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
        ])
    return augment_trans

def get_mask_img(pil_im ,mask_path):
    mask = Image.open(mask_path)
    # matrix = 255 - np.asarray(mask)#mask反转
    matrix = np.asarray(mask)
    mask = Image.fromarray(matrix)  # 矩阵转图像
    # 处理mask
    print("mask_type========",type(mask))
    # print("mask_type========",type(mask_1))
    img_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        # transforms.Normalize(norm_mean, norm_std),
    ])
    mask = img_transforms(mask)
    print("mask_type========", type(mask))
    print("mask111========", mask.size())
    # mask = torch.cat([mask, mask, mask], axis=0).permute(1, 2, 0)
    mask = mask.permute(1, 2, 0)
    mask = mask.cpu().numpy()
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1

    pil_im = pil_im.squeeze().permute(1, 2, 0)
    im_np = np.array(pil_im.cpu())
    im_np = im_np / im_np.max()
    im_np = mask * im_np
    im_np[mask == 0] = 1

    im_masked_final = (im_np / im_np.max() * 255).astype(np.uint8)
    im_masked_final = Image.fromarray(im_masked_final)


    return im_masked_final


def check_and_create_dir(path):
    pathdir = osp.split(path)[0]
    if osp.isdir(pathdir):
        pass
    else:
        os.makedirs(pathdir)

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
    # 创建一个随机数生成器，使用默认的种子值
    rng = np.random.RandomState()
    current = rng.randint(n)  # 随机选择一个起始点索引
    samples.append(current)  # 将起始点索引加入采样点索引集
    samples_points.append(points[current])  # 将起始点加入采样点集

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

    return samples_points


def get_image_hull_mask(image, image_landmarks, ie_polys=None):
    # get the mask of the image
    if image_landmarks.shape[0] != 68:
        raise Exception(
            'get_image_hull_mask works only with 68 landmarks')
    int_lmrks = np.array(image_landmarks, dtype=np.int)
    print("int_lmrks====", int_lmrks)
    print("image_shape====",image.shape[0])
    print("image_shape====",image.shape[1])
    hull_mask = np.full(image.shape[0:2] + (1,), 0, dtype=np.float32)

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[0:9],
                        int_lmrks[18:19]))), (1,))
    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[8:17],
                        int_lmrks[26:27]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:20],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[24:27],
                        int_lmrks[8:9]))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[19:25],
                        int_lmrks[8:9],
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[17:22],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    cv2.fillConvexPoly(hull_mask, cv2.convexHull(
        np.concatenate((int_lmrks[22:27],
                        int_lmrks[27:28],
                        int_lmrks[31:36],
                        int_lmrks[8:9]
                        ))), (1,))

    # nose
    cv2.fillConvexPoly(
        hull_mask, cv2.convexHull(int_lmrks[27:36]), (1,))
    if ie_polys is not None:
        ie_polys.overlay_mask(hull_mask)
    hull_mask = 255 * np.uint8(hull_mask)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 40))
    # hull_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, kernel, 1)
    # hull_mask = cv2.GaussianBlur(hull_mask, (15, 15), cv2.BORDER_DEFAULT)
    inversemask = cv2.bitwise_not(hull_mask)
    return hull_mask, inversemask

def merge_add_alpha(img, mask):
    # merge rgb and mask into a rgba image
    color_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    alpha_channel = img * color_mask
    # alpha_channel = cv2.bitwise_and(img, img, mask=mask)
    return alpha_channel

def plot_attn_dino(attn, threshold_map, inputs, inds, use_wandb, output_path):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(2, attn.shape[0] + 2, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=10, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, 2)
    plt.imshow(attn.sum(0).numpy(), interpolation='nearest')
    plt.title("atn map sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 3)
    plt.imshow(threshold_map[-1].numpy(), interpolation='nearest')
    plt.title("prob sum")
    plt.axis("off")

    plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 4)
    plt.imshow(threshold_map[:-1].sum(0).numpy(), interpolation='nearest')
    plt.title("thresh sum")
    plt.axis("off")

    for i in range(attn.shape[0]):
        plt.subplot(2, attn.shape[0] + 2, i + 3)
        plt.imshow(attn[i].numpy())
        plt.axis("off")
        plt.subplot(2, attn.shape[0] + 2, attn.shape[0] + 1 + i + 4)
        plt.imshow(threshold_map[i].numpy())
        plt.axis("off")
    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()


def plot_attn_clip(attn, threshold_map, inputs, inds, use_wandb, output_path, display_logs):
    # currently supports one image (and not a batch)
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 3, 1)
    main_im = make_grid(inputs, normalize=True, pad_value=2)
    main_im = np.transpose(main_im.cpu().numpy(), (1, 2, 0))
    plt.imshow(main_im, interpolation='nearest')
    plt.scatter(inds[:, 1], inds[:, 0], s=40, c='red', marker='o')
    plt.title("input im")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(attn, interpolation='nearest', vmin=0, vmax=1)
    plt.title("atn map")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    threshold_map_ = (threshold_map - threshold_map.min()) / \
        (threshold_map.max() - threshold_map.min())
    plt.imshow(threshold_map_, interpolation='nearest', vmin=0, vmax=1)
    plt.title("prob softmax")
    plt.scatter(inds[:, 1], inds[:, 0], s=40, c='red', marker='o')
    plt.axis("off")

    plt.tight_layout()
    if use_wandb:
        wandb.log({"attention_map": wandb.Image(plt)})
    plt.savefig(output_path)
    plt.close()

def plot_atten(attn, threshold_map, inputs, inds, use_wandb, output_path, saliency_model, display_logs):
    if saliency_model == "dino":
        plot_attn_dino(attn, threshold_map, inputs,
                       inds, use_wandb, output_path)
    elif saliency_model == "clip":
        plot_attn_clip(attn, threshold_map, inputs, inds,
                       use_wandb, output_path, display_logs)