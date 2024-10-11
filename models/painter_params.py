#加入打乱顺序选取五官点
import random
import CLIP_.clip as clip
import numpy as np
import pydiffvg
import sketch_utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage.filters import gaussian_filter
from skimage.color import rgb2gray
from skimage.filters import threshold_otsu
from torchvision import transforms
import dlib
import cv2
from collections import OrderedDict
import numpy.random as npr
import math
import copy
import svgpathtools
def shape_to_np(shape, dtype="int"):
    #创建68*2
    coords = np.zeros((shape.num_parts, 2), dtype=dtype)#shape.num_parts=68，用np.zeros创建全0的numpy.ndarry，大小为(68,2)
    #遍历每一个关键点
    #得到坐标
    for i in range(0, shape.num_parts):
        coords[i] = (shape.part(i).x, shape.part(i).y)#取出每个点的坐标，然后赋值给coords对应的位置
    return coords

class Painter(torch.nn.Module):
    def __init__(self, args,
                 num_strokes=4,
                 num_segments=4,
                 imsize=224,
                 device=None,
                 target_im=None):
        super(Painter, self).__init__()

        self.args = args
        self.num_segments = num_segments
        self.width = args.width
        self.control_points_per_seg = args.control_points_per_seg
        self.opacity_optim = args.force_sparse
        self.num_stages = args.num_stages
        self.add_random_noise = "noise" in args.augemntations
        self.noise_thresh = args.noise_thresh
        self.softmax_temp = args.softmax_temp

        self.num_face_paths = args.num_face_paths
        self.num_global_paths = args.num_global_paths
        self.num_paths = args.num_face_paths+args.num_global_paths

        self.shapes = []
        self.shape_groups = []
        self.shapes_ = []
        self.shape_groups_ = []
        self.shapes_record = []
        self.shape_groups_record = []
        self.device = device
        self.canvas_width, self.canvas_height = imsize, imsize
        self.points_vars = []
        self.color_vars = []
        # self.stroke_width_var = []
        self.color_vars_threshold = args.color_vars_threshold

        self.path_svg = args.path_svg
        self.strokes_per_stage = self.num_paths
        self.optimize_flag = []

        # attention related for strokes initialisation
        self.attention_init = args.attention_init
        self.target_path = args.target
        self.saliency_model = args.saliency_model
        self.xdog_intersec = args.xdog_intersec
        self.mask_object = args.mask_object_attention
        self.image_scale = args.image_scale

        self.text_target = ""  # for clip gradients
        self.saliency_clip_model = args.saliency_clip_model
        self.define_attention_input(target_im)
        self.attention_map = self.set_attention_map() if self.attention_init else None

        self.thresh = self.set_attention_threshold_map() if self.attention_init else None
        self.strokes_counter = 0  # counts the number of calls to "get_path"
        self.epoch = 0
        self.final_epoch = args.num_iter - 1

        self.image_scale = args.image_scale
        self.device = args.device
    def clipasso_init_image(self, stage=0):
        if stage > 0:
            # if multi stages training than add new strokes on existing ones
            # don't optimize on previous strokes
            self.optimize_flag = [False for i in range(len(self.shapes))]
            for i in range(self.strokes_per_stage):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                 fill_color=None,
                                                 stroke_color=stroke_color)
                self.shape_groups.append(path_group)
                self.optimize_flag.append(True)

        else:
            num_paths_exists = 0
            if self.path_svg != "none":
                self.canvas_width, self.canvas_height, self.shapes, self.shape_groups = utils.load_svg(self.path_svg)
                # if you want to add more strokes to existing ones and optimize on all of them
                num_paths_exists = len(self.shapes)

            for i in range(num_paths_exists, self.num_paths):
                stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
                path = self.get_path()
                self.shapes.append(path)
                path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                                 fill_color=None,
                                                 stroke_color=stroke_color)
                self.shape_groups.append(path_group)
            print("shape_groups====",self.shape_groups)
            self.optimize_flag = [True for i in range(len(self.shapes))]

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (
                    1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def get_image(self):
        img = self.render_warp()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW

        return img

    def get_image_2(self):
        img = self.render_warp_2()
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW

        return img
    # def get_image(self, inds_record):#起点不动
    #     num_paths_exists = 0
    #     stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
    #     self.shapes_ = []
    #     self.shape_groups_ = []
    #     i = 0
    #     for path in self.shapes:
    #         # new_points = []
    #         p0 = inds_record[i]
    #         p0_tensor = torch.tensor([p0]).to(self.device)
    #         p0_tensor[: 0] *= self.image_scale
    #         p0_tensor[: 1] *= self.image_scale
    #         new_points = torch.cat((p0_tensor, path.points), dim=0)
    #         # new_points.append(new_p)
    #         path_ = pydiffvg.Path(num_control_points=self.num_control_points,
    #                                 points=new_points,
    #                                 stroke_width=torch.tensor(self.width),
    #                                 is_closed=False)
    #
    #         self.shapes_.append(path_)
    #         path_group_ = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes_) - 1]),
    #                                         fill_color=None,
    #                                         stroke_color=stroke_color)
    #
    #         self.shape_groups_.append(path_group_)
    #         i += 1
    #
    #     img = self.render_warp()
    #     opacity = img[:, :, 3:4]
    #     img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
    #     img = img[:, :, :3]
    #     # Convert img from HWC to NCHW
    #     img = img.unsqueeze(0)
    #     img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
    #
    #     return img

    def get_sketch(self, shapes_record, shape_groups_record):
        print("!!!!!!!!!!!!v1")
        _render = pydiffvg.RenderFunction.apply
        print("!!!!!!!!!!!!v2")
        # uncomment if you want to add random noise
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            self.canvas_width, self.canvas_height, shapes_record, shape_groups_record)
        print("!!!!!!!!!!!!v3")
        img = _render(self.canvas_width,  # width
                      self.canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      0,  # seed
                      None,
                      *scene_args)
        print("!!!!!!!!!!!!v4")
        opacity = img[:, :, 3:4]
        img = opacity * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (1 - opacity)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW

        return img
    def get_path(self):
        points = []
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        p0 = self.inds_normalised[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        print("p0=======", p0)
        print("p0=======", type(p0))
        points.append(p0)
        print("points=======",points)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.canvas_width
        points[:, 1] *= self.canvas_height

        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)
        self.strokes_counter += 1
        print("points1=", points)
        print("path=======", path)
        print("self.strokes_counter=======", self.strokes_counter)

        return path

    def get_path_(self,inds_record):
        points = []
        print("inds_record==========",len(inds_record))
        print("inds_record==========",inds_record)
        print("num_segments==========",self.num_segments)
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        # p0 = inds_record[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        p0 = inds_record
        print("p0=======", p0)
        print("p0=", type(p0))
        print("strokes_counter=", self.strokes_counter)
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.image_scale
        points[:, 1] *= self.image_scale
        print('num_control_points',self.num_control_points)
        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)

        # single 去掉stroke_counter
        # self.strokes_counter += 1
        return path


    def get_path_2(self,inds_record):
        points = []
        print("inds_record==========",len(inds_record))
        print("inds_record==========",inds_record)
        print("num_segments==========",self.num_segments)
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        # p0 = inds_record[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        p0 = inds_record
        print("p0=======", p0)
        print("p0=", type(p0))
        print("strokes_counter=", self.strokes_counter)
        points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.image_scale
        points[:, 1] *= self.image_scale
        print('num_control_points',self.num_control_points)
        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)

        # single 去掉stroke_counter
        self.strokes_counter += 1
        return path
    def get_path_1(self,inds_record):
        points = []#不含起点
        self.num_control_points = torch.zeros(self.num_segments, dtype=torch.int32) + (self.control_points_per_seg - 2)
        p0 = inds_record[self.strokes_counter] if self.attention_init else (random.random(), random.random())
        # points.append(p0)

        for j in range(self.num_segments):
            radius = 0.05
            for k in range(self.control_points_per_seg - 1):
                p1 = (p0[0] + radius * (random.random() - 0.5), p0[1] + radius * (random.random() - 0.5))
                points.append(p1)
                p0 = p1
        points = torch.tensor(points).to(self.device)
        points[:, 0] *= self.image_scale
        points[:, 1] *= self.image_scale

        path = pydiffvg.Path(num_control_points=self.num_control_points,
                             points=points,
                             stroke_width=torch.tensor(self.width),
                             is_closed=False)

        self.strokes_counter += 1
        return path

    def init_image_1(self,inds_record, num_paths):
        num_paths_exists = 0
        self.shapes = []
        self.shape_groups = []
        # single 加
        # for i in range(num_paths_exists, self.num_paths):
        # for i in range(num_paths_exists, (self.num_hair_paths + self.num_face_paths)):
        for i in range(num_paths_exists, num_paths):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = self.get_path_2(inds_record[i])
            # path = self.get_path_1(inds_record)#起点不动
            # self.shapes.append(path)
            self.shapes.append(path)
            self.shapes_record.append(path)

            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                             fill_color=None,
                                             stroke_color=stroke_color)
            path_group_record = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes_record) - 1]),
                                                    fill_color=None,
                                                    stroke_color=stroke_color)
            self.shape_groups.append(path_group)
            self.shape_groups_record.append(path_group_record)

        # # 保存路径为 SVG 文件
        # pydiffvg.save_svg("shapes.svg", 800, 800, self.shapes, self.shape_groups)
        self.optimize_flag = [True for i in range(len(self.shapes))]

        # img = self.render_warp()
        # img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (
        #         1 - img[:, :, 3:4])
        # img = img[:, :, :3]
        # # Convert img from HWC to NCHW
        # img = img.unsqueeze(0)
        # img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        # return img

    def init_image(self, inds_record, num_paths):
        num_paths_exists = 0
        self.shapes = []
        self.shape_groups = []

        if self.path_svg != "none":
            self.canvas_width, self.canvas_height, self.shapes_record, self.shape_groups_record = utils.load_svg(self.path_svg)
            # if you want to add more strokes to existing ones and optimize on all of them
            num_paths_exists = len(self.shapes)
        print("num_paths_exists========", num_paths_exists)
        # self.shapes_record = self.shapes
        # self.shape_groups_record = self.shape_groups
        for i in range(num_paths_exists, num_paths):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path = self.get_path_2(inds_record[i])
            # path = self.get_path_1(inds_record)#起点不动
            # self.shapes.append(path)
            self.shapes.append(path)
            self.shapes_record.append(path)

            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes) - 1]),
                                             fill_color=None,
                                             stroke_color=stroke_color)
            path_group_record = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes_record) - 1]),
                                                    fill_color=None,
                                                    stroke_color=stroke_color)
            self.shape_groups.append(path_group)
            self.shape_groups_record.append(path_group_record)

        # # 保存路径为 SVG 文件
        # pydiffvg.save_svg("shapes.svg", 800, 800, self.shapes, self.shape_groups)
        self.optimize_flag = [True for i in range(len(self.shapes))]

        # img = self.render_warp()
        # img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (
        #         1 - img[:, :, 3:4])
        # img = img[:, :, :3]
        # # Convert img from HWC to NCHW
        # img = img.unsqueeze(0)
        # img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        # return img

    def render_warp(self):
        if self.opacity_optim:
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0., 0.)  # to force black stroke
                group.stroke_color.data[-1].clamp_(0., 1.)  # opacity
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply
        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for path in self.shapes:
                    path.points.data.add_(eps * torch.randn_like(path.points))
        # scene_args = pydiffvg.RenderFunction.serialize_scene( \
        #     self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        # scene_args = pydiffvg.RenderFunction.serialize_scene( \
        #     self.canvas_width, self.canvas_height, self.shapes_, self.shape_groups_)#起点不动
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            self.canvas_width, self.canvas_height, self.shapes_record, self.shape_groups_record)
        print("!!!!!2!!!!!!!")
        img = _render(self.canvas_width,  # width
                      self.canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      0,  # seed
                      None,
                      *scene_args)
        return img

    def render_warp_2(self):
        if self.opacity_optim:
            for group in self.shape_groups:
                group.stroke_color.data[:3].clamp_(0., 0.)  # to force black stroke
                group.stroke_color.data[-1].clamp_(0., 1.)  # opacity
                # group.stroke_color.data[-1] = (group.stroke_color.data[-1] >= self.color_vars_threshold).float()
        _render = pydiffvg.RenderFunction.apply
        # uncomment if you want to add random noise
        if self.add_random_noise:
            if random.random() > self.noise_thresh:
                eps = 0.01 * min(self.canvas_width, self.canvas_height)
                for path in self.shapes:
                    path.points.data.add_(eps * torch.randn_like(path.points))
        scene_args = pydiffvg.RenderFunction.serialize_scene( \
            self.canvas_width, self.canvas_height, self.shapes, self.shape_groups)
        # scene_args = pydiffvg.RenderFunction.serialize_scene( \
        #     self.canvas_width, self.canvas_height, self.shapes_, self.shape_groups_)#起点不动
        # scene_args = pydiffvg.RenderFunction.serialize_scene( \
        #     self.canvas_width, self.canvas_height, self.shapes_record, self.shape_groups_record)
        print("!!!!!2!!!!!!!")
        img = _render(self.canvas_width,  # width
                      self.canvas_height,  # height
                      2,  # num_samples_x
                      2,  # num_samples_y
                      0,  # seed
                      None,
                      *scene_args)
        return img


    def get_lineloss(self, inds_num):
        distances = 0

        print("inds_num===", inds_num)
        def distance_(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        for path in self.shapes:

            # 获取起始点和终点
            start_point = path.points[0]
            middle_point1 = path.points[1]
            middle_point2 = path.points[2]
            end_point = path.points[3]
            # 计算起始点和终点距离的均值
            distances += (distance_(start_point, middle_point1) + distance_(middle_point1, middle_point2) + distance_(middle_point2, end_point)) / 3


        distances = distances / (inds_num)
        distances_str = str(distances)
        if '.' in distances_str:
            integer_digits = distances_str.find('.')
        else:
            integer_digits = len(distances_str)
        print("integer_digits",integer_digits)

        return integer_digits, distances

    def get_lineloss_2(self):
        distances = 0
        # ll = 0

        def distance(point1, point2):
            x1, y1 = point1
            x2, y2 = point2
            return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

        for path in self.shapes:
            # 获取起始点和终点
            start_point = path.points[0]
            middle_point1 = path.points[1]
            middle_point2 = path.points[2]
            end_point = path.points[3]
            # 计算相邻点之间距离之和
            distances += distance(start_point, middle_point1)
            distances += distance(middle_point1, middle_point2)
            distances += distance(middle_point2, end_point)

        distances = distances / (self.num_paths)
        # print("ll=====", ll)
        # distances = distances + ll
        return distances
    def parameters(self):
        self.points_vars = []
        # storkes' location optimization
        for i, path in enumerate(self.shapes):
            path.points.requires_grad = True
            self.points_vars.append(path.points)
        # return self.points_vars
        return self.points_vars

    def get_points_parans(self):
        return self.points_vars

    def set_color_parameters(self):
        # for storkes' color optimization (opacity)
        self.color_vars = []
        for i, group in enumerate(self.shape_groups):
            group.stroke_color.requires_grad = True
            self.color_vars.append(group.stroke_color)
        return self.color_vars

    def set_stroke_parameters(self):
        # for storkes' color optimization (opacity)
        self.stroke_color_vars = []
        for i, group in enumerate(self.shape_groups):
            group.stroke_color.requires_grad = True
            # self.stroke_color.data.clamp_(0, 255)
            # Make sure stroke_color is either black or white
            self.stroke_color_vars.append(group.stroke_color)
        return self.stroke_color_vars


    def get_color_parameters(self):
        return self.color_vars
    # def get_stroke_parameters(self):
    #     return self.stroke_var

    # def save_svg(self, output_dir, name):
    #     pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes,
    #                       self.shape_groups)
    def save_svg(self, output_dir, name):
        pydiffvg.save_svg('{}/{}.svg'.format(output_dir, name), self.canvas_width, self.canvas_height, self.shapes_record,
                          self.shape_groups_record)#freeze
    def dino_attn(self):
        patch_size = 8  # dino hyperparameter
        threshold = 0.6

        # for dino model
        mean_imagenet = torch.Tensor([0.485, 0.456, 0.406])[None, :, None, None].to(self.device)
        std_imagenet = torch.Tensor([0.229, 0.224, 0.225])[None, :, None, None].to(self.device)
        totens = transforms.Compose([
            transforms.Resize((self.canvas_height, self.canvas_width)),
            transforms.ToTensor()
        ])

        dino_model = torch.hub.load('facebookresearch/dino:main', 'dino_vits8').eval().to(self.device)

        self.main_im = Image.open(self.target_path).convert("RGB")
        main_im_tensor = totens(self.main_im).to(self.device)
        img = (main_im_tensor.unsqueeze(0) - mean_imagenet) / std_imagenet
        w_featmap = img.shape[-2] // patch_size
        h_featmap = img.shape[-1] // patch_size

        with torch.no_grad():
            attn = dino_model.get_last_selfattention(img).detach().cpu()[0]

        nh = attn.shape[0]
        attn = attn[:, 0, 1:].reshape(nh, -1)
        val, idx = torch.sort(attn)
        val /= torch.sum(val, dim=1, keepdim=True)
        cumval = torch.cumsum(val, dim=1)
        th_attn = cumval > (1 - threshold)
        idx2 = torch.argsort(idx)
        for head in range(nh):
            th_attn[head] = th_attn[head][idx2[head]]
        th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
        th_attn = nn.functional.interpolate(th_attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()

        attn = attn.reshape(nh, w_featmap, h_featmap).float()
        attn = nn.functional.interpolate(attn.unsqueeze(0), scale_factor=patch_size, mode="nearest")[0].cpu()

        return attn

    def define_attention_input(self, target_im):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
            preprocess.transforms[-1],
        ])
        self.image_input_attn_clip = data_transforms(target_im).to(self.device)

    def clip_attn(self):
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        text_input = clip.tokenize([self.text_target]).to(self.device)

        if "RN" in self.saliency_clip_model:
            saliency_layer = "layer4"
            attn_map = gradCAM(
                model.visual,
                self.image_input_attn_clip,
                model.encode_text(text_input).float(),
                getattr(model.visual, saliency_layer)
            )
            attn_map = attn_map.squeeze().detach().cpu().numpy()
            attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())

        else:
            # attn_map = interpret(self.image_input_attn_clip, text_input, model, device=self.device, index=0).astype(np.float32)
            attn_map = interpret(self.image_input_attn_clip, model, device=self.device)

        del model
        return attn_map

    def set_attention_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.dino_attn()
        elif self.saliency_model == "clip":
            return self.clip_attn()

    def softmax(self, x, tau=0.2):
        e_x = np.exp(x / tau)
        return e_x / e_x.sum()

    def set_inds_clip(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (
                    self.attention_map.max() - self.attention_map.min())
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map

        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        k = self.num_stages * self.num_paths
        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False,
                                     p=attn_map_soft.flatten())
        # self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False)
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return attn_map_soft

    def set_inds_dino(self):
        k = max(3, (self.num_stages * self.num_paths) // 6 + 1)  # sample top 3 three points from each attention head
        num_heads = self.attention_map.shape[0]
        self.inds = np.zeros((k * num_heads, 2))
        # "thresh" is used for visualisaiton purposes only
        thresh = torch.zeros(num_heads + 1, self.attention_map.shape[1], self.attention_map.shape[2])
        softmax = nn.Softmax(dim=1)
        for i in range(num_heads):
            # replace "self.attention_map[i]" with "self.attention_map" to get the highest values among
            # all heads.
            topk, indices = np.unique(self.attention_map[i].numpy(), return_index=True)
            topk = topk[::-1][:k]
            cur_attn_map = self.attention_map[i].numpy()
            # prob function for uniform sampling
            prob = cur_attn_map.flatten()
            prob[prob > topk[-1]] = 1
            prob[prob <= topk[-1]] = 0
            prob = prob / prob.sum()
            thresh[i] = torch.Tensor(prob.reshape(cur_attn_map.shape))

            # choose k pixels from each head
            inds = np.random.choice(range(cur_attn_map.flatten().shape[0]), size=k, replace=False, p=prob)
            inds = np.unravel_index(inds, cur_attn_map.shape)
            self.inds[i * k: i * k + k, 0] = inds[0]
            self.inds[i * k: i * k + k, 1] = inds[1]

        # for visualisaiton
        sum_attn = self.attention_map.sum(0).numpy()
        mask = np.zeros(sum_attn.shape)
        mask[thresh[:-1].sum(0) > 0] = 1
        sum_attn = sum_attn * mask
        sum_attn = sum_attn / sum_attn.sum()
        thresh[-1] = torch.Tensor(sum_attn)

        # sample num_paths from the chosen pixels.
        prob_sum = sum_attn[self.inds[:, 0].astype(np.int), self.inds[:, 1].astype(np.int)]
        prob_sum = prob_sum / prob_sum.sum()
        new_inds = []
        for i in range(self.num_stages):
            new_inds.extend(np.random.choice(range(self.inds.shape[0]), size=self.num_paths, replace=False, p=prob_sum))
        self.inds = self.inds[new_inds]
        print("self.inds", self.inds.shape)

        self.inds_normalised = np.zeros(self.inds.shape)
        self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        self.inds_normalised = self.inds_normalised.tolist()
        return thresh

    def set_attention_threshold_map(self):
        assert self.saliency_model in ["dino", "clip"]
        if self.saliency_model == "dino":
            return self.set_inds_dino()
        elif self.saliency_model == "clip":
            return self.set_inds_clip()

    def get_attn(self):
        return self.attention_map

    def get_thresh(self):
        return self.thresh

    def get_inds(self):
        return self.inds

    def get_mask(self):
        return self.mask

    def set_random_noise(self, epoch):
        if epoch % self.args.save_interval == 0:
            self.add_random_noise = False
        else:
            self.add_random_noise = "noise" in self.args.augemntations

    def get_attn_global_points(self,output_dir):
        print("!!!!!!!!!!!!!")
        print("====================global=====================")
        attn_map = (self.attention_map - self.attention_map.min()) / (
                    self.attention_map.max() - self.attention_map.min())
        print("attn_map=========",type(attn_map))
        print("attn_map=========",attn_map.shape)
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map
        print("attn_map2=========", type(attn_map))
        print("attn_map2=========", attn_map.shape)
        cv2.imwrite(r"{}/attn_map.png".format(output_dir), attn_map)
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        # k = int((self.num_stages * self.num_paths)/self.masknum)
        numsize = 64
        # numsize = self.num_hair_paths
        k = self.num_stages * numsize
        print("range(attn_map.flatten().shape[0])======",range(attn_map.flatten().shape[0]))
        print("attn_map_soft.flatten()======", attn_map_soft.flatten())

        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False,
                                     p=attn_map_soft.flatten())
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T
        print("11111111111111")
        print("self.inds==========",self.inds)
        # self.inds_normalised = np.zeros(self.inds.shape)
        # self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        # self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        # self.inds_normalised = self.inds_normalised.tolist()
        # print("self.inds_normalised000000000==========", self.inds_normalised)
        return list(self.inds)



    # def get_hair_points(self):
    #     attn_map = (self.attention_map - self.attention_map.min()) / (
    #             self.attention_map.max() - self.attention_map.min())
    #     print("attn_map=========", type(attn_map))
    #     print("attn_map=========", attn_map.shape)
    #     if self.xdog_intersec:
    #         xdog = XDoG_()
    #         im_xdog = xdog(self.image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=15)
    #         intersec_map = (1 - im_xdog) * attn_map
    #         attn_map = intersec_map
    #     print("attn_map2=========", type(attn_map))
    #     print("attn_map2=========", attn_map.shape)
    #     attn_map_soft = np.copy(attn_map)
    #     attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)
    #     attn_map_soft = np.copy(attn_map)
    #     attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)
    #
    #     hair_inds = np.argwhere(attn_map_soft)
    #     print("hair_inds===", hair_inds)
    #     # hair_inds = np.array(np.unravel_index(attn_map_soft.flatten(), attn_map.shape)).T
    #     hair_inds = utils.fps(hair_inds, self.num_hair_paths)
    #     hair_inds = np.array(hair_inds)
    #     # hair_inds = [(y, x) for x, y in hair_inds]  # 调整坐标
    #     return hair_inds

    def get_hair_points(self):
        attn_map = (self.attention_map - self.attention_map.min()) / (
                self.attention_map.max() - self.attention_map.min())
        print("attn_map2=========", type(attn_map))
        print("attn_map2=========", attn_map.shape)
        gray = cv2.cvtColor(attn_map, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(gray, (4, 4))
        # 使用 Canny 边缘检测器检测线条
        edges = cv2.Canny(blur, 100, 150)
        edge_points = np.argwhere(edges > 0)  # 检测的边缘上的点


        # hair_inds = np.array(np.unravel_index(attn_map_soft.flatten(), attn_map.shape)).T
        hair_inds = utils.fps(edge_points, self.num_hair_paths)
        hair_inds = np.array(hair_inds)
        print("hair_inds", hair_inds)

        # hair_inds = [(y, x) for x, y in hair_inds]  # 调整坐标
        return hair_inds

    def get_face_points(self,face_path, output_dir):
        print("====================face=====================")
        # 定位68点位置
        FACIAL_LANDMARKS_68_IDX = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor/shape_predictor_68_face_landmarks.dat")#创建人脸68点预测器
        img = cv2.imread(face_path)
        img = cv2.resize(img,(self.canvas_width,self.canvas_height),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#得到灰度图
        cv2.imwrite(r"{}/gray_1.png".format(output_dir), gray)
        rects = detector(gray, 1)
        print("rects==========",rects)

        #face
        for (i, rect) in enumerate(rects):
            # 对人脸框进行关键点定位
            # 转换为ndarray
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)  # 得到68个关键点的坐标，(68,2)numpy.ndarray
        shape_ = shape;
        face_inds = utils.fps(shape_, self.num_face_paths)
        # face_inds = utils.fps(np.concatenate([shape[17:26],shape[36:67] ]), self.num_face_paths)
        face_inds = [(y, x) for x, y in face_inds]#调整坐标
        face_inds = np.array(face_inds)

        # 68位定点
        # for (name, (i, j)) in FACIAL_LANDMARKS_68_IDX.items():
        #     clone = img.copy()
        #     cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        #     for (x, y) in shape[i:j]:
        #         cv2.circle(clone, (x, y), 2, (0, 0, 255), -1)
        #     # cv2.imshow("image", img)
        #     cv2.imwrite(r"{}/clone_all.png".format(output_dir), clone)
        clone = img.copy()
        for (name, (i, j)) in FACIAL_LANDMARKS_68_IDX.items():
            # cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 2, (0, 255, 0), -1)
            # cv2.imshow("image", img)
        cv2.imwrite(r"{}/clone_all.png".format(output_dir), clone)

        return face_inds

    def get_points(self,face_path, output_dir):
        print("====================face=====================")
        # 定位68点位置
        FACIAL_LANDMARKS_68_IDX = OrderedDict([
            ("mouth", (48, 68)),
            ("right_eyebrow", (17, 22)),
            ("left_eyebrow", (22, 27)),
            ("right_eye", (36, 42)),
            ("left_eye", (42, 48)),
            ("nose", (27, 36)),
            ("jaw", (0, 17))
        ])
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("./shape_predictor/shape_predictor_68_face_landmarks.dat")#创建人脸68点预测器
        img = cv2.imread(face_path)
        img = cv2.resize(img,(self.canvas_width,self.canvas_height),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#得到灰度图
        cv2.imwrite(r"{}/gray_1.png".format(output_dir), gray)
        rects = detector(gray, 1)
        print("rects==========",rects)

        #face
        for (i, rect) in enumerate(rects):
            # 对人脸框进行关键点定位
            # 转换为ndarray
            shape = predictor(gray, rect)
            shape = shape_to_np(shape)  # 得到68个关键点的坐标，(68,2)numpy.ndarray

        print("face_shape_len====",len(shape))
        print("face_shape====",shape)
        face_inds = utils.fps(shape, self.num_face_paths)
        face_inds = [(y, x) for x, y in face_inds]#调整坐标
        face_inds = np.array(face_inds)
        #hair
        hull_mask, inversemask = utils.get_image_hull_mask(img, shape, ie_polys=None)
        cv2.imwrite(r"{}/hull_mask.png".format(output_dir), hull_mask)
        cv2.imwrite(r"{}/inversemask.png".format(output_dir), inversemask)
        hull_mask_img = cv2.bitwise_and(img, img, mask=hull_mask)
        inversemask_img = cv2.bitwise_and(img, img, mask=inversemask)
        print("inversemask_img===", type(inversemask_img))
        # 遍历图像的每个像素
        for i in range(inversemask_img.shape[0]):  # 图像宽度
            for j in range(inversemask_img.shape[1]):  # 图像高度
                # 如果掩码值为0，则将像素值改为白色（RGB值为255）
                if inversemask[i, j] == 0:
                    inversemask_img[i, j] = (255, 255, 255)
        cv2.imwrite(r"{}/hull_mask_img.png".format(output_dir), hull_mask_img)
        cv2.imwrite(r"{}/inversemask_img.png".format(output_dir), inversemask_img)

        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)
        data_transforms = transforms.Compose([
            preprocess.transforms[-1],
        ])
        inversemask_img = torch.from_numpy(inversemask_img).float()
        inversemask_img= inversemask_img.permute(2,0,1).unsqueeze(0)
        image_input_attn_clip = data_transforms(inversemask_img).to(self.device)
        print("image_input_attn_clip===", image_input_attn_clip.size())
        model, preprocess = clip.load(self.saliency_clip_model, device=self.device, jit=False)
        model.eval().to(self.device)

        attn_map = interpret(image_input_attn_clip, model, device=self.device)

        del model
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min())
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map
        cv2.imwrite(r"{}/attn_map3.png".format(output_dir), attn_map)
        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        hair_inds = np.argwhere(attn_map_soft)
        print("hair_inds===", hair_inds)
        # hair_inds = np.array(np.unravel_index(attn_map_soft.flatten(), attn_map.shape)).T
        hair_inds = utils.fps(hair_inds, self.num_hair_paths)


        #FPS采点
        # masked_gray = cv2.cvtColor(inversemask_img, cv2.COLOR_BGR2GRAY)
        # masked_blur = cv2.blur(masked_gray, (4, 4))
        # # 使用 Canny 边缘检测器检测线条
        # masked_edges = cv2.Canny(masked_blur, 100, 150)
        # masked_points = np.argwhere(masked_edges > 0)  #
        # hair_inds = utils.fps(masked_points, self.num_hair_paths)
        # hair_inds = [(y, x) for x, y in hair_inds]  # 调整坐标
        # for face in rects:
        #     shape = predictor(img, face)  # 寻找人脸的68个标定点
        #     # 遍历所有点，打印出其坐标，并圈出来
        #     print("\n")
        #     i = 1
        #     for pt in shape.parts():
        #         # print(str(i) + "\t" + str(pt))
        #         i = i + 1
        #         pt_pos = (pt.x, pt.y)
        #         cv2.circle(img, pt_pos, 2, (0, 255, 0), -1)
        #     cv2.imwrite(r"{}/clone_all.png".format(output_dir), img)

        # k = int((self.num_stages * self.num_paths)/self.masknum)
        # numsize = self.num_face_paths
        # print("numsize========", numsize)
        # k = self.num_stages * numsize
        # # 从68个点中酌情选择一部分，再选取k个点
        # jaw_weight = 1
        # r_brow_weight = 2
        # l_brow_weight = 2
        # nose_weight = 2
        # r_eye_weight = 4
        # l_eye_weight = 4
        # mouth_weight = 1
        #
        # jaw_weight = 1
        # r_brow_weight = 1
        # l_brow_weight = 1
        # nose_weight = 1
        # r_eye_weight = 1
        # l_eye_weight = 1
        # mouth_weight = 1

        #
        # shape_list =  jaw_weight * list(shape[0:16]) + r_brow_weight * list(shape[16:21]) + l_brow_weight * list(shape[22:26]) + \
        #               nose_weight * list(shape[27:35]) + r_eye_weight * list(shape[36:41]) + l_eye_weight * list(shape[42:47]) + \
        #               mouth_weight * list(shape[48:67])
        # print("shape_list_len",len(shape_list))
        # print("shape_list_type",shape_list)
        # #随机打乱定位顺序
        # # random.shuffle(shape_list)
        # # random.shuffle(shape_list)
        # random.shuffle(shape_list)
        # self.inds = random.sample(shape_list, k)  # [1, 2]
        # # self.inds = random.sample(list(shape[6:10])+list(shape[17:26])+list(shape[31:47])+list(shape[60:64]), k ) # [1, 2]
        # #从68个点中选取k个点
        #
        # # self.inds = random.sample(list(shape), k) # [1, 2]
        # self.inds  = np.array(self.inds)
        # print("=======pick_face_inds==========", self.inds.shape[0])
        # self.inds_normalised = np.zeros(self.inds.shape)
        # self.inds_normalised[:, 0] = self.inds[:, 0] / self.canvas_width
        # self.inds_normalised[:, 1] = self.inds[:, 1] / self.canvas_height
        # self.inds_normalised = self.inds_normalised.tolist()
        # print("self.inds_normalised====face======", self.inds_normalised)
        # 68位定点
        for (name, (i, j)) in FACIAL_LANDMARKS_68_IDX.items():
            clone = img.copy()
            cv2.putText(clone, name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            for (x, y) in shape[i:j]:
                cv2.circle(clone, (x, y), 2, (0, 0, 255), -1)
            # cv2.imshow("image", img)
            cv2.imwrite(r"{}/clone_all.png".format(output_dir), clone)


        return face_inds, hair_inds, attn_map

    def get_image_points(self, roi_pathnum):
        attn_map = (self.attention_map - self.attention_map.min()) / (
                self.attention_map.max() - self.attention_map.min())
        if self.xdog_intersec:
            xdog = XDoG_()
            im_xdog = xdog(self.image_input_attn_clip[0].permute(1, 2, 0).cpu().numpy(), k=10)
            intersec_map = (1 - im_xdog) * attn_map
            attn_map = intersec_map

        attn_map_soft = np.copy(attn_map)
        attn_map_soft[attn_map > 0] = self.softmax(attn_map[attn_map > 0], tau=self.softmax_temp)

        # k = self.num_stages * self.num_paths
        k = 1 * roi_pathnum
        self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False,
                                     p=attn_map_soft.flatten())
        # self.inds = np.random.choice(range(attn_map.flatten().shape[0]), size=k, replace=False)
        self.inds = np.array(np.unravel_index(self.inds, attn_map.shape)).T

        # self.inds_normalised = np.zeros(self.inds.shape)
        # self.inds_normalised[:, 0] = self.inds[:, 1] / self.canvas_width
        # self.inds_normalised[:, 1] = self.inds[:, 0] / self.canvas_height
        # self.inds_normalised = self.inds_normalised.tolist()

        # return self.inds_normalised,list(self.inds)
        return self.inds

    def set_shapes(self,inds_record, record):
        num_paths_exists = 0
        self.shapes = []
        self.shape_groups = []
        #single 加
        print("inds_record=",inds_record)
        # if len(inds_record) != 1:
        #     inds_record = [inds_record]
        # print("inds_record=", inds_record)
        # single 加
        # for i in range(num_paths_exists, (self.num_hair_paths + self.num_face_paths)):
        for i in range(num_paths_exists, 1):
            stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
            path, p0_tensor = self.get_path_1(inds_record)
            # self.shapes.append(path)
            self.shapes.append(path)
            p0_tensor = torch.tensor(inds_record, device=self.device).unsqueeze(0)
            p0_tensor[:, 0] *= self.image_scale
            p0_tensor[:, 1] *= self.image_scale
            path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes)-1]),
                                             fill_color=None,
                                             stroke_color=stroke_color)
            self.shape_groups.append(path_group)

        # # 保存路径为 SVG 文件
        # pydiffvg.save_svg("shapes.svg", 800, 800, self.shapes, self.shape_groups)
        self.optimize_flag = [True for i in range(len(self.shapes))]

        img = self.render_warp()
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device=self.device) * (
                1 - img[:, :, 3:4])
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2).to(self.device)  # NHWC -> NCHW
        return img
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def get_shapes(self, p0_tensor, record, shapes_record):
        self.shapes_ = []
        self.shape_groups_ = []
        #single 加
        for path in self.shapes:
                new_points = torch.cat((p0_tensor, path.points), dim=0)
        print("new_points===", new_points)
        print("self.num_control_points===", self.num_control_points)

        path_ = pydiffvg.Path(num_control_points=self.num_control_points,
                              points=new_points,
                              stroke_width=torch.tensor(self.width),
                              is_closed=False)
        self.shapes_.append(path_)

        # single 加
        stroke_color = torch.tensor([0.0, 0.0, 0.0, 1.0])
        print("len(self.shape=====", len(self.shapes_))
        print("record=====", record)
        # path_group_ = pydiffvg.ShapeGroup(shape_ids=torch.tensor([len(self.shapes_) - 1]),
        path_group_ = pydiffvg.ShapeGroup(shape_ids=torch.tensor([record]),
                                          fill_color=None,
                                          stroke_color=stroke_color)
        self.shape_groups_.append(path_group_)

        return self.shapes_, self.shape_groups_
        # utils.imwrite(img.cpu(), '{}/init.png'.format(args.output_dir), gamma=args.gamma, use_wandb=args.use_wandb, wandb_name="init")

    def get_end(self):
        pydiffvg.save_svg('./final.svg', self.canvas_width, self.canvas_height, self.shapes,
                          self.shape_groups)
        for path in self.shapes:
            end_point = path.points[2]
        return end_point

    def freeze_hair(self):
        self.shapes= []

class PainterOptimizer:
    def __init__(self, args, renderer):
        self.renderer = renderer
        self.points_lr = args.lr
        self.color_lr = args.color_lr
        self.stroke_lr = args.stroke_lr
        self.args = args
        self.optim_color = args.force_sparse


    def init_optimizers(self):
        self.points_optim = torch.optim.Adam(self.renderer.parameters(), lr=self.points_lr)
        if self.optim_color:
            self.color_optim = torch.optim.Adam(self.renderer.set_color_parameters(), lr=self.color_lr)

        # self.color_optim = torch.optim.Adam(self.renderer.set_color_parameters(), lr=self.color_lr)

    def update_lr(self, counter):
        new_lr = utils.get_epoch_lr(counter, self.args)
        print("new_lr====",new_lr)
        for param_group in self.points_optim.param_groups:
            param_group["lr"] = new_lr

    def zero_grad_(self):
        self.points_optim.zero_grad()
        if self.optim_color:
            self.color_optim.zero_grad()
        # self.color_optim.zero_grad()
    def step_(self):
        self.points_optim.step()
        if self.optim_color:
            self.color_optim.step()
        # self.color_optim.step()
    def get_lr(self):
        return self.points_optim.param_groups[0]['lr']


class Hook:
    """Attaches to a module and records its activations and gradients."""

    def __init__(self, module: nn.Module):
        self.data = None
        self.hook = module.register_forward_hook(self.save_grad)

    def save_grad(self, module, input, output):
        self.data = output
        output.requires_grad_(True)
        output.retain_grad()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.hook.remove()

    @property
    def activation(self) -> torch.Tensor:
        return self.data

    @property
    def gradient(self) -> torch.Tensor:
        return self.data.grad


def interpret(image, model, device):
    images = image.repeat(1, 1, 1, 1)
    res = model.encode_image(images)
    model.zero_grad()
    image_attn_blocks = list(dict(model.visual.transformer.resblocks.named_children()).values())
    # target_block = image_attn_blocks[5]
    num_tokens = image_attn_blocks[0].attn_probs.shape[-1]
    R = torch.eye(num_tokens, num_tokens, dtype=image_attn_blocks[0].attn_probs.dtype).to(device)
    R = R.unsqueeze(0).expand(1, num_tokens, num_tokens)
    cams = []  # there are 12 attention blocks
    for i, blk in enumerate(image_attn_blocks):
        cam = blk.attn_probs.detach()  # attn_probs shape is 12, 50, 50
        # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
        cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
        cam = cam.clamp(min=0)
        cam = cam.clamp(min=0).mean(dim=1)  # mean of the 12 something
        cams.append(cam)
        R = R + torch.bmm(cam, R)

    # cam = target_block.attn_probs.detach()  # attn_probs shape is 12, 50, 50
    #     # each patch is 7x7 so we have 49 pixels + 1 for positional encoding
    # cam = cam.reshape(1, -1, cam.shape[-1], cam.shape[-1])
    # cam = cam.clamp(min=0)
    # cam = cam.clamp(min=0).mean(dim=1)  # mean of the 12 something
    # cams.append(cam)
    # R = R + torch.bmm(cam, R)

    cams_avg = torch.cat(cams)  # 12, 50, 50
    cams_avg = cams_avg[:, 0, 1:]  # 12, 1, 49
    image_relevance = cams_avg.mean(dim=0).unsqueeze(0)
    image_relevance = image_relevance.reshape(1, 1, 7, 7)
    image_relevance = torch.nn.functional.interpolate(image_relevance, size=224, mode='bicubic')
    image_relevance = image_relevance.reshape(224, 224).data.cpu().numpy().astype(np.float32)
    image_relevance = (image_relevance - image_relevance.min()) / (image_relevance.max() - image_relevance.min())
    return image_relevance


# Reference: https://arxiv.org/abs/1610.02391
def gradCAM(
        model: nn.Module,
        input: torch.Tensor,
        target: torch.Tensor,
        layer: nn.Module
) -> torch.Tensor:
    # Zero out any gradients at the input.
    if input.grad is not None:
        input.grad.data.zero_()

    # Disable gradient settings.
    requires_grad = {}
    for name, param in model.named_parameters():
        requires_grad[name] = param.requires_grad
        param.requires_grad_(False)

    # Attach a hook to the model at the desired layer.
    assert isinstance(layer, nn.Module)
    with Hook(layer) as hook:
        # Do a forward and backward pass.
        output = model(input)
        output.backward(target)

        grad = hook.gradient.float()
        act = hook.activation.float()

        # Global average pool gradient across spatial dimension
        # to obtain importance weights.
        alpha = grad.mean(dim=(2, 3), keepdim=True)
        # Weighted combination of activation maps over channel
        # dimension.
        gradcam = torch.sum(act * alpha, dim=1, keepdim=True)
        # We only want neurons with positive influence so we
        # clamp any negative ones.
        gradcam = torch.clamp(gradcam, min=0)

    # Resize gradcam to input resolution.
    gradcam = F.interpolate(
        gradcam,
        input.shape[2:],
        mode='bicubic',
        align_corners=False)

    # Restore gradient settings.
    for name, param in model.named_parameters():
        param.requires_grad_(requires_grad[name])

    return gradcam


class XDoG_(object):
    def __init__(self):
        super(XDoG_, self).__init__()
        self.gamma = 0.98
        self.phi = 200
        self.eps = -0.1
        self.sigma = 0.8
        self.binarize = True

    def __call__(self, im, k=10):
        if im.shape[2] == 3:
            im = rgb2gray(im)
        imf1 = gaussian_filter(im, self.sigma)
        imf2 = gaussian_filter(im, self.sigma * k)
        imdiff = imf1 - self.gamma * imf2
        imdiff = (imdiff < self.eps) * 1.0 + (imdiff >= self.eps) * (1.0 + np.tanh(self.phi * imdiff))
        imdiff -= imdiff.min()
        imdiff /= imdiff.max()
        if self.binarize:
            th = threshold_otsu(imdiff)
            imdiff = imdiff >= th
        imdiff = imdiff.astype('float32')
        return imdiff