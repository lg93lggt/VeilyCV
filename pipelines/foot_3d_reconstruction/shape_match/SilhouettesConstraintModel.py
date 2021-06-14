

from pathlib import Path
import matplotlib
import pytorch3d.renderer.cameras  
from pytorch3d.structures import Pointclouds, Meshes
import torch
import torch.nn
from torch.utils.data import Dataset, DataLoader
import pytorch3d.structures
import pytorch3d.renderer
import pytorch3d.io
import pytorch3d.loss
import pytorch3d.ops
import pytorch3d.transforms
import open3d as o3d
import glob
import numpy as np
import pandas as pd
import cv2
from cv2 import aruco
from icecream import ic
from matplotlib import pyplot as plt
from easydict import EasyDict
import os
import sys
import time
from typing import Dict, Union

sys.path.append("../..")
from dataset.PeakFootDataset.PeakFootDataset import PeakFootDataset, SIDE_OF_FOOT, UNITIZED_LENGTH, DATA_TYPE
from src.Camera import PinholeCamera
from src.plugins.image import cvt_fig_from_plt_to_cv2
from src.plugins.camera import cvt_camera_vl_with_trajectory_to_torch3d
from src.plugins.geometry3d import covert_meshes_o3d_to_torch3d
from src.utils.debug import debug_separator
os.environ["NUMEXPR_MAX_THREADS"] = "8"

SCALE = 4
DO_BINARIZE_OBJ = False
DO_BINARIZE_DST = False
THRES_BIN = 0.01
K_BIN = 100
LR_OPT = 0.001
NITERS_OPT = 10000
N_SAMPLED_MODELS = 100
DEBUG = False
LAMDA = EasyDict({"bce": 1, "smooth": 0.00001, "scale_regularize": 0.00001})
sum_lamda = 0
for key in LAMDA.keys():
    sum_lamda += LAMDA[key]
for key in LAMDA.keys():
    LAMDA[key] /= sum_lamda


class SilhouettesConstraintModel(torch.nn.Module):
    def __init__(self, masks, meshes, cameras: pytorch3d.renderer.cameras.PerspectiveCameras, device="cuda:0", **kwargs):
        super().__init__()
        self.device = device
        self.vis = o3d.visualization.Visualizer()


        if "imgs_raw" in kwargs.keys():
            self.imgs_raw = kwargs["imgs_raw"]

        self.masks_dst  = torch.from_numpy(np.asarray(masks) / 255).to(dtype=torch.float32, device=device)
        self.meshes_src = meshes
        self.cameras    = cameras
        
        self.n_models = self.meshes_src.verts_padded().shape[0]
        self.n_points = self.meshes_src.verts_padded().shape[1]
        self.n_images = self.masks_dst.shape[0]
        self.n_cameras = self.cameras._N
        
        if self.n_cameras != self.n_images:
            raise ValueError("Num of cameras should equal to num of imaeges")

        self.R_cam = self.cameras.R.clone()
        self.T_cam = self.cameras.T.clone()

        self.size_image_wh = self.cameras.image_size
        self.size_image_hw = torch.flip(self.cameras.image_size, dims=[1])
        self.meshes_dst = meshes.clone()
        
        # NOTE raster_settings.image_size = (H, W)
        raster_settings = pytorch3d.renderer.RasterizationSettings(
            image_size=(int(self.size_image_hw[0, 0]), int(self.size_image_hw[0, 1])),
            blur_radius=0,
            faces_per_pixel=100,
        )

        self.silhouette_renderer = pytorch3d.renderer.MeshRenderer(
            rasterizer=pytorch3d.renderer.MeshRasterizer(
                cameras=self.cameras,
                raster_settings=raster_settings
            ),
            shader=pytorch3d.renderer.SoftSilhouetteShader()
        )

        """
        coord_axis
        """
        ax_o3d = o3d.geometry.TriangleMesh.create_coordinate_frame(0.18)
        v = torch.from_numpy(np.array(ax_o3d.vertices)) \
            .to(dtype=torch.float32, device=self.device)
        f = torch.from_numpy(np.array(ax_o3d.triangles)) \
            .to(dtype=torch.float32, device=self.device)
        t = pytorch3d.renderer.Textures(verts_rgb=torch.from_numpy(np.asarray(ax_o3d.vertex_colors)).to(
            dtype=torch.float32, device=self.device).unsqueeze(0).repeat(self.n_images, 1, 1))
        self.ax = pytorch3d.structures.Meshes(
            [v]*self.n_images, [f]*self.n_images).to(device) 

        """
        params optim
        """
        self._set_initial_parameters()
        return
    
    def _set_initial_parameters(self, **kwargs: Union[None, float]):
        keys = kwargs.keys()
        if "scale" not in keys:
            self.register_parameter(name="scale", param=torch.nn.Parameter(torch.tensor([0.25], dtype=torch.float32).to(self.device),  requires_grad=True))
        else:
            self.register_parameter(name="scale", param=torch.nn.Parameter(torch.tensor([kwargs["scale"]], dtype=torch.float32).to(self.device), requires_grad=True))
            
        if "theta" not in keys:
            self.register_parameter(name="theta", param=torch.nn.Parameter(torch.tensor([0], dtype=torch.float32).to(self.device), requires_grad=True))
        else:
            self.register_parameter(name="theta", param=torch.nn.Parameter(torch.tensor([kwargs["theta"]], dtype=torch.float32).to(self.device), requires_grad=True))
            
        if "tx" not in keys:
            self.register_parameter(name="tx", param=torch.nn.Parameter(torch.tensor([0.1], dtype=torch.float32).to(self.device), requires_grad=True))
        else:
            self.register_parameter(name="tx", param=torch.nn.Parameter(torch.tensor([kwargs["tx"]], dtype=torch.float32).to(self.device), requires_grad=True))
            
        if "ty" not in keys:
            self.register_parameter(name="ty", param=torch.nn.Parameter(torch.tensor([0.15], dtype=torch.float32).to(self.device), requires_grad=True))
        else:
            self.register_parameter(name="ty", param=torch.nn.Parameter(torch.tensor([kwargs["ty"]], dtype=torch.float32).to(self.device), requires_grad=True))
        return


    def show_image(self, block=False):
        if not block:
            plt.ion()
            
        imgs_out = self.masks_show[..., 3].detach().squeeze().cpu().numpy()
        #(fig, axes)  = plt.subplots(1, self.n_images)
        for i in range(self.n_images):            
            plt.subplot(1, self.n_images, i+1)
            plt.imshow(imgs_out[i] * 255 + self.imgs_raw[i].astype(np.float))
        if not block:
            plt.show()
            plt.pause(1)
        else:
            plt.show()
        I = cvt_fig_from_plt_to_cv2(plt.figure(1))
        return I


    def forward(self):
        self.scale.clamp(min= 0.20, max=0.32,)
        self.theta.clamp(min=-0.30, max=0.30,)
        self.tx.clamp(min=0.00, max=0.20,)
        self.ty.clamp(min=0.03, max=0.23,)

        mesh_world = self.meshes_dst.clone()
        
        Tz = torch.eye(4).repeat((self.n_models, 1, 1)).to(self.device)
        Tz[:, -1, 2] = - mesh_world.get_bounding_boxes()[:, 2, 0]
        trans_Tz = pytorch3d.transforms.Transform3d(matrix=Tz, device=self.device)
        verts_Tz = trans_Tz.transform_points(mesh_world.verts_padded())
        mesh_world = mesh_world.update_padded(verts_Tz)
           
        S = pytorch3d.transforms.Transform3d()
        S = S.scale(self.scale.repeat(self.n_models, 3) / 2,).to(self.device)     
        verts_S = S.transform_points(mesh_world.verts_padded())
        mesh_world = mesh_world.update_padded(verts_S)

        M = torch.eye(4).repeat((self.n_models, 1, 1)).to(self.device)
        RR = pytorch3d.transforms.RotateAxisAngle(angle=self.theta*500, axis="Z", dtype=torch.float32, device=self.device).get_matrix().expand(self.n_models, 4, 4)
        M[:, :3, :3] = RR[:, :3, :3]
        M[:, -1, 0] = self.tx.expand(self.n_models).to(dtype=torch.float32, device=self.device)
        M[:, -1, 1] = self.ty.expand(self.n_models).to(dtype=torch.float32, device=self.device)

        trans_M = pytorch3d.transforms.Transform3d(matrix=M, device=self.device)
        verts_M = trans_M.transform_points(mesh_world.verts_padded())
        mesh_world = mesh_world.update_padded(verts_M)

        masks_ren = self.silhouette_renderer(meshes_world=mesh_world, R=self.R_cam, T=self.T_cam)
        
        bce_loss = torch.nn.BCELoss(weight=torch.tensor([0.999], dtype=torch.float32, device=self.device))
        if DEBUG:
            ic(info(self.cameras.R))
            ic(info(self.cameras.T))
            ic(info(self.cameras.image_size))
            ic(self.meshes_src._N)
            ic(info(masks_ren[..., 3]))
            ic(info(self.masks_dst))
        loss_bce = bce_loss(masks_ren[..., 3], self.masks_dst[...])
        loss = loss_bce

        self.masks_show = 0.2*masks_ren + self.silhouette_renderer(meshes_world=self.ax.clone(), R=self.R_cam, T=self.T_cam)
        return loss, masks_ren
    
    @ debug_separator
    def train(self, n_iters: int=300, debug=False, dir_debug="/media/veily3/data_ligan/匹克脚部数据集/cropped_simplified_1000/debug"):
        img_out = None
        optimizer = torch.optim.Adam([model.scale, model.theta, model.tx, model.ty, ], LR_OPT)      
        loss_min = torch.tensor(1E6, dtype=torch.float32, device=self.device)
        t0 = time.time()
        for i_iter in range(n_iters):
            optimizer.zero_grad()
            (loss, masks_render) = model.forward()
            loss.backward(retain_graph=True)
            optimizer.step()
            
            if loss < loss_min:
                loss_min = loss
                params = self.state_dict()
                masks_render_best = masks_render.clone()
                
            if (i_iter % 100 == 0) or (i_iter == n_iters - 1):
                t1 = time.time()
                dt = t1 - t0
                progress = "{0}/{1}".format(i_iter, n_iters)
                ic(progress, np.round(loss.item(), 4), np.round(dt, 2))
                t0 = time.time()                    
                if debug:
                    img_out = model.show_image(block=False)

            if (loss < 0.01) and (i_iter > 100):
                break
        ic(params, loss_min.item())
        return (loss_min, params, masks_render_best, img_out)


def info(tensor):
    return (tensor.shape, tensor.dtype, tensor.device)


def differentiable_binarization_gpu(img_src, threshold=1e-6, k=1000, device="cuda:0"):
    T = torch.ones_like(img_src).to(device) * threshold
    img_dst = 1 / (1 + torch.exp(-k * (img_src - T)))
    return img_dst


def differentiable_binarization(img_src, threshold=1e-6, k=1000):
    T = np.ones_like(img_src) * threshold
    img_dst = 1 / (1 + np.exp(-k * (img_src - T)))
    return img_dst


def mesh_pytorch3d_to_o3d(mesh):
    verts = mesh.verts_padded().squeeze(0).detach().cpu().numpy()
    faces = mesh.faces_padded().squeeze(0).detach().cpu().numpy().astype(np.int)
    mesh_o3d = o3d.geometry.TriangleMesh(
        vertices=o3d.utility.Vector3dVector(verts),
        triangles=o3d.utility.Vector3iVector(faces),
    )
    return mesh_o3d



if __name__ == "__main__":
    dir_dataset_root = Path("/media/veily3/data_ligan/匹克脚部数据集/cropped_simplified")
    peak_dataset = PeakFootDataset(
        dir_root=dir_dataset_root,
        suffix=".obj",
        flag_side_foot=SIDE_OF_FOOT.RIGHT,
        flag_unitized_length=UNITIZED_LENGTH.IDENTITY,
    )

    device = "cuda:0"
    dir_root = Path("/media/veily3/data_ligan/voxel_carving_data/project/ligan_right_0526-1317")
    dir_mask = Path(dir_root, "mask")
    dir_raw  = Path(dir_root, "raw")
    
    n_cam = 4
    SCALE = 0.5 

    cam = PinholeCamera(640, 480)
    pth_camera_parameters  = Path(dir_root, "data/camera_params.pkl")
    pth_camera_trajcectory = Path(dir_root, "data/trajectory.pkl")
    cam.load_from_file(pth_camera_parameters)
    cam.load_from_file(pth_camera_trajcectory)
    cam.resize(scale=SCALE)
    cameras = cvt_camera_vl_with_trajectory_to_torch3d(camera=cam, idxes=range(n_cam), device=device)
    
    masks        = np.asarray([cv2.resize(cv2.imread(str(dir_mask / name), cv2.IMREAD_GRAYSCALE), (cam.width, cam.height, )) for name in cam.names_traj])
    imgs_raw_bgr = np.asarray([cv2.resize(cv2.imread(str(dir_raw  / name)), (cam.width, cam.height, )) for name in cam.names_traj])
        
    imgs_raw_gray = np.zeros_like(masks)
    for i_img, img_render in enumerate(imgs_raw_bgr):
        cam.apply_trajectory_by_index(i_img)
        img_render = cam.project_axis_on_image(0.18, 2, img_render)
        imgs_raw_gray[i_img] = cv2.cvtColor(img_render, cv2.COLOR_BGR2GRAY)
        
    masks = masks[:n_cam]
    imgs_raw_gray = imgs_raw_gray[:n_cam]
    
    results_all = pd.DataFrame(index=peak_dataset.sr_names.values.tolist(), columns=["scale", "theta", "tx", "ty", "loss"])
    params = None
    dataloader = DataLoader(dataset=peak_dataset, batch_size=1, shuffle=False, drop_last=False)
    for (i_btach, idxes_in_batch) in enumerate(dataloader):
        if i_btach < 1000:
            continue
        for idx in idxes_in_batch:
            [_, names, meshes] = peak_dataset._get_items([int(idx)], flag_data_type=DATA_TYPE.NORMALIZED_X)
            if "ligan" not in names:
                break
            
            meshes = pytorch3d.structures.join_meshes_as_batch([meshes]*n_cam)    
            
            #  Model
            model = SilhouettesConstraintModel(masks=masks, meshes=meshes, cameras=cameras, imgs_raw=imgs_raw_gray)
            model._set_initial_parameters(**params) if params is not None else None
            (loss, params, imgs_render, img_out) = model.train(1000, debug=True)
            imgs_render = imgs_render.detach().cpu().numpy()[..., -1]
            for (i_img, img_render) in enumerate(imgs_render):
                img_render = (img_render * 255).astype(np.uint8)
                img_render = cv2.resize(img_render, (0, 0), fx=1/SCALE, fy=1/SCALE)
                (contours, _) = cv2.findContours(img_render, cv2.RETR_EXTERNAL, cv2.CONTOURS_MATCH_I1,)
                pth_img_show = dir_root / "raw" / cam.names_traj[i_img]
                img_show = cv2.imread(str(pth_img_show))
                #img_show = cv2.resize(img_show, (cam.width, cam.height))
                img_show = cv2.drawContours(image=img_show, contours=contours, contourIdx=-1, color=(0, 0, 255))
                cv2.imwrite(str(peak_dataset.dir_root / "{}.png".format(i_img)), img_show)
                ic(str(peak_dataset.dir_root / "{}.png".format(i_img)))
                print()
                
            
            pth_output_image = peak_dataset.dir_root / "{}.png".format(names[0])
            cv2.imwrite(str(pth_output_image), img_out)
            ic(pth_output_image)
            for key in params.keys():
                value = params[key].cpu().numpy()
                value = np.round(value, decimals=4)
                results_all.loc[names, key] = value[0]
            results_all.loc[names, "loss"] = loss.item()
    #results_all.to_excel(peak_dataset.dir_root / "debug" / "match_results.xlsx")
