
import sys
from typing import Iterable, List, Literal, Sequence, Tuple, Union
import os

from torch.nn.functional import normalize
from pipelines.foot_3d_reconstruction.main_aruco import UNITIZED_LENGTH
import pytorch3d.io
from pytorch3d.structures import Meshes
import pytorch3d.loss
import pytorch3d.transforms
import pytorch3d.ops
import numpy as np
from icecream import ic
from pathlib import Path
from easydict import EasyDict
import docx
import os
import pandas as pd
from enum import Enum, auto, IntEnum, Flag, unique
import torch
from torch._C import device
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler 
import open3d as o3d
import multiprocessing


os.environ["NUMEXPR_MAX_THREADS"] = "8"
sys.path.append("../..")
from src.utils.debug import debug_separator, debug_vis
from src.plugins.geometry3d import covert_meshes_o3d_to_torch3d, convert_meshes_torch3d_to_o3d



class SIDE_OF_FOOT(Enum):
    left  = auto()
    right = auto()
SIDE_LEFT  = SIDE_OF_FOOT.left
SIDE_RIGHT = SIDE_OF_FOOT.right


@ unique
class DATA_TYPE(Flag):
    RAW            = 0
    NORMALIZED_X   = 1
    NORMALIZED_Y   = 2
    NORMALIZED_Z   = 4
    NORMALIZED_XY  = NORMALIZED_X | NORMALIZED_Y
    NORMALIZED_XYZ = NORMALIZED_X | NORMALIZED_Y | NORMALIZED_Z


@ debug_separator
class PeakFootsDataset(Dataset):
    """
    """
    def __init__(self, dir_root: Union[str, Path], flag_side_foot: SIDE_OF_FOOT=SIDE_RIGHT, flag_unitized_length: UNITIZED_LENGTH=UNITIZED_LENGTH.MM, suffix=".ply", device: str="cuda:0"):
        super().__init__()
        self.unitized_length = flag_unitized_length
        self.side_of_foot       = flag_side_foot

        self.suffix = suffix

        self.dir_root = Path(dir_root)
        self.dict_dirs_data = {
            DATA_TYPE.RAW           : Path(self.dir_root, "raw"),
            DATA_TYPE.NORMALIZED_X  : Path(self.dir_root, "normalized", "X"),
            DATA_TYPE.NORMALIZED_Y  : Path(self.dir_root, "normalized", "Y"),
            DATA_TYPE.NORMALIZED_Z  : Path(self.dir_root, "normalized", "Z"),
            DATA_TYPE.NORMALIZED_XY : Path(self.dir_root, "normalized", "XY"),
            DATA_TYPE.NORMALIZED_XYZ: Path(self.dir_root, "normalized", "XYZ"),
        }

        self.pthes_row = sorted(self.dict_dirs_data[DATA_TYPE.RAW].glob("*/"))
        self.sr_names = pd.Series([pth_raw.name for pth_raw in self.pthes_row])

        self.device = device

        ic(PeakFootsDataset, dir_root, flag_side_foot, flag_unitized_length, len(self), self.suffix)
        return


    def __getitem__(self, idx: int) -> int:
        super().__init__()
        return idx
        # else:
        #     raise FileExistsError("The PLY file does not exist.")

    def __len__(self):
        super().__init__()
        return len(self.pthes_row)

    def _get_items_by_names(self, names: List[str], flag_data_type: DATA_TYPE=DATA_TYPE.RAW) -> Union[List[int], Meshes]:
        pthes_mesh  = []
        idxes_exist = []
        for name in names:
            dir_mesh = self.dict_dirs_data[flag_data_type]
            pth_mesh = Path(dir_mesh, name, self.side_of_foot.name + self.suffix)
            if pth_mesh.exists():
                pthes_mesh.append(pth_mesh)
                idxes_exist.append((self.sr_names[self.sr_names == name]).index[0])
            else:
                ic(pth_mesh.parent.stem, "dont exist")
                continue

        meshes      = []
        idxes_valid = []
        if   self.suffix == ".obj":
            meshes = pytorch3d.io.load_objs_as_meshes(pthes_mesh)
            idxes_valid = idxes_exist
        elif self.suffix == ".ply":
            for [idx_ply, pth_mesh] in enumerate(pthes_mesh):
                try:
                    [verts, faces] = pytorch3d.io.load_ply(pth_mesh)
                    mesh = Meshes([verts], [faces]).to(self.device)
                    meshes.append(mesh)
                    idxes_valid.append(int(idxes_exist[idx_ply]))
                except :
                    continue
            meshes = pytorch3d.structures.join_meshes_as_batch(meshes)

        meshes = meshes.to(self.device)
        ic(len(meshes), meshes.verts_padded().shape, meshes.faces_padded().shape)
        return [idxes_valid, meshes]

    def _get_items(self, idxes: List[int] or torch.Tensor, flag_data_type: DATA_TYPE=DATA_TYPE.RAW) -> Tuple[List[int], Meshes]:
        pthes_mesh  = []
        idxes_exist = []
        idxes = idxes.cpu().numpy().tolist() if isinstance(idxes, torch.Tensor) else idxes
        for idx in idxes:
            dir_mesh = self.dict_dirs_data[flag_data_type]
            pth_mesh = Path(dir_mesh, self.sr_names.iloc[idx], self.side_of_foot.name + self.suffix)
            if pth_mesh.exists():
                pthes_mesh.append(pth_mesh)
                idxes_exist.append(idx)
            else:
                ic(pth_mesh.parent.stem, "dont exist")
                continue
            
        if   self.suffix == ".obj":
            meshes = pytorch3d.io.load_objs_as_meshes(pthes_mesh)
            idxes_valid = idxes_exist
        elif self.suffix == ".ply":
            meshes      = []
            idxes_valid = []
            for [idx_ply, pth_mesh] in enumerate(pthes_mesh):
                try:
                    [verts, faces] = pytorch3d.io.load_ply(pth_mesh)
                    mesh = Meshes([verts], [faces]).to(self.device)
                    meshes.append(mesh)
                    idxes_valid.append(int(idxes_exist[idx_ply]))
                except :
                    continue
            meshes = pytorch3d.structures.join_meshes_as_batch(meshes)

        meshes = meshes.to(self.device)
        ic(meshes.verts_padded().shape)
        return [idxes_valid, meshes]

    def icp(self, meshes_src: Meshes, mesh_std: Meshes):
        """
        align meshes_src to mesh_std
        """
        X = meshes_src.verts_padded()
        Y = mesh_std.verts_padded().repeat(len(X), 1, 1)
        (converged, rmse, XT, _, _) = pytorch3d.ops.iterative_closest_point(X, Y)
        ic(converged, rmse)
        meshes_src = meshes_src.update_padded(XT)
        return meshes_src


    def normalize(self, meshes: Meshes, flag_unitized_length: UNITIZED_LENGTH, flag_data_type: DATA_TYPE=DATA_TYPE.NORMALIZED_XYZ) \
            -> Tuple[Meshes, pd.DataFrame]:

        meshes = meshes.to(self.device) if (meshes.device != self.device) else meshes
        n_mesh = len(meshes)
        max_len = meshes.get_bounding_boxes()[:, :, 1] - meshes.get_bounding_boxes()[:, :, 0] 
        center  = (meshes.get_bounding_boxes()[:, :, 1] + meshes.get_bounding_boxes()[:, :, 0]) / 2 

        """
        Translate
        """
        tx = -center[:, 0] 
        ty = -center[:, 1] 
        tz = -center[:, 2] if (flag_data_type != DATA_TYPE.NORMALIZED_Z) else torch.zeros(n_mesh, dtype=torch.float32, device=self.device)
        # 原始为后跟
        # centroids = torch.mean(meshes.verts_padded(), axis=1)
        T = pytorch3d.transforms.Transform3d()
        #T = T.translate(-centroids).to(self.device) # 质心
        T = T.translate(
            x=tx, 
            y=ty,
            z=tz
        ).to(self.device) # 脚底中心
        verts_T = T.transform_points(meshes.verts_padded())
        meshes = meshes.update_padded(verts_T)

        """
        Scale
        """
        [sx, sy, sz] = torch.ones((3, n_mesh), dtype=torch.float32, device=self.device)
        if  (flag_data_type == DATA_TYPE.NORMALIZED_X):
            [sx, sy, sz] = (2/max_len[:, 0]).expand(3, n_mesh)
        elif (flag_data_type == DATA_TYPE.NORMALIZED_Y):
            [sx, sy, sz] = (2/max_len[:, 1]).expand(3, n_mesh)
        elif (flag_data_type == DATA_TYPE.NORMALIZED_Z):
            [sx, sy, sz] = (2/max_len[:, 2]).expand(3, n_mesh)
        elif (flag_data_type == DATA_TYPE.NORMALIZED_XY):
            sx = 2 / max_len[:, 0]
            sy = 2 / max_len[:, 1]
            sz = torch.full([n_mesh, ], fill_value=UNITIZED_LENGTH.DM.value / flag_unitized_length.value).to(self.device)
        elif (flag_data_type == DATA_TYPE.NORMALIZED_XYZ):
            sx = 2 / max_len[:, 0]
            sy = 2 / max_len[:, 1]
            sz = 2 / max_len[:, 2]
            
        #[max_len, idx_argmax] = torch.max(torch.norm(meshes.verts_padded(), dim=2), dim=1)#求这个batch点云的模的最大值
        if (max_len==0).any():
            raise ZeroDivisionError("Max length of mesh is ZERO.")
        else:
            # scale = 2 / max_len_x
            S = pytorch3d.transforms.Transform3d()
            S = S.scale(
                x=sx, 
                y=sy,
                z=sz
            ).to(self.device)
            verts_S = S.transform_points(meshes.verts_padded())
            meshes = meshes.update_padded(verts_S)

        df_factors = pd.DataFrame(
            columns=["tx", "ty", "tz", "sx", "sy", "sz"], 
            data=torch.stack((tx, ty, tz, sx, sy, sz), dim=1).cpu().numpy()
        )
        return [meshes, df_factors]


    @ debug_separator
    def generate_normalized_dataset(self, batch_size: int=128, flag_data_type: DATA_TYPE=DATA_TYPE.NORMALIZED_XYZ):
        dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=False, drop_last=False)
        

        dir_output_parent = self.dict_dirs_data[flag_data_type]
        pth_output_xlsx = dir_output_parent / "normalization_factors.xlsx"
        df_factors_all =  pd.read_excel(pth_output_xlsx, index_col=0) if pth_output_xlsx.exists() else  pd.DataFrame(index=self.sr_names, columns=["tx", "ty", "tz", "sx", "sy", "sz"], )

        for i_batch, idxes in enumerate(dataloader):
            print("*"*64)
            ic(i_batch+1, "/", len(dataloader))
            ic(dir_output_parent)
            ic(self.suffix)
            [idxes_valid, meshes] = self._get_items(idxes, flag_data_type=DATA_TYPE.RAW)
            if len(meshes) == 0:
                continue
            
            [meshes, df_factors] = self.normalize(meshes, flag_data_type=flag_data_type, flag_unitized_length=self.unitized_length)

            df_factors.index = self.sr_names.iloc[idxes_valid]
            df_factors_all.iloc[idxes_valid, :] = df_factors.values

            for [idx_mesh, idx_valid] in enumerate(idxes_valid):
                dir_output = dir_output_parent / self.sr_names.iloc[idx_valid]
                dir_output.mkdir(parents=True, exist_ok=True, )
                pth_output = Path(dir_output, self.side_of_foot.name + self.suffix)
                if   self.suffix == ".ply":
                    pytorch3d.io.save_ply(f=pth_output, verts=meshes.verts_list()[idx_mesh].cpu(), faces=meshes.faces_list()[idx_mesh].cpu())
                elif self.suffix == ".obj":
                    pytorch3d.io.save_obj(f=pth_output, verts=meshes.verts_list()[idx_mesh].cpu(), faces=meshes.faces_list()[idx_mesh].cpu())
                ic(pth_output.parent.name)

                df_factors_all.to_excel(pth_output_xlsx)
                """
                DEBUG
                """
                if False:
                    ms = o3d.io.read_triangle_mesh(str(pth_output))
                    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(1.)
                    sphere =  o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_sphere(1.))
                    o3d.visualization.draw_geometries([ sphere, ms, coord])
        ic(df_factors_all.info())
        ic(pth_output_xlsx)
        return

    @ debug_separator
    def generate_aligned_dataset(self, batch_size: int, flag_data_type: DATA_TYPE):
        dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=False, drop_last=False)

        dir_output_parent = self.dict_dirs_data[flag_data_type]
        (idxes, mesh_std) = self._get_items([0], flag_data_type=flag_data_type)

        for i_batch, idxes in enumerate(dataloader):
            print("*"*64)
            ic(i_batch+1, "/", len(dataloader))
            ic(dir_output_parent)
            ic(self.suffix)

            [idxes_valid, meshes] = self._get_items(idxes, flag_data_type=DATA_TYPE.RAW)
            if len(meshes) == 0:
                continue
            
            meshes = self.icp(meshes, mesh_std)

            for [idx_mesh, idx_valid] in enumerate(idxes_valid):
                dir_output = dir_output_parent / self.sr_names.iloc[idx_valid]
                dir_output.mkdir(parents=True, exist_ok=True, )
                pth_output = Path(dir_output, self.side_of_foot.name + self.suffix)
                if   self.suffix == ".ply":
                    pytorch3d.io.save_ply(f=pth_output, verts=meshes.verts_list()[idx_mesh].cpu(), faces=meshes.faces_list()[idx_mesh].cpu())
                elif self.suffix == ".obj":
                    pytorch3d.io.save_obj(f=pth_output, verts=meshes.verts_list()[idx_mesh].cpu(), faces=meshes.faces_list()[idx_mesh].cpu())
                ic(pth_output.parent.name)

                """
                DEBUG
                """
                if False:
                    m0 = cvt_meshes_torch3d_to_o3d(mesh_std)[0]
                    m1 = o3d.io.read_triangle_mesh(str(pth_output))
                    
                    o3d.visualization.draw_geometries([m0.paint_uniform_color([1,0,0.]), m1.paint_uniform_color([0,0,1.])])
        return

    @ debug_separator
    def match_with_all(self, mesh_src: Meshes, batch_size: int=128, n_samples: int=2000, flag_data_type: DATA_TYPE=DATA_TYPE.NORMALIZED_XY) -> pd.Series:
        mesh_src = mesh_src.to(self.device)
        # [mesh_src, df_factors] = self.normalize(mesh_src, flag_data_type=flag_data_type, flag_unitized_length=UNITIZED_LENGTH.M)

        dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=False, drop_last=False)
        
        sr_losses = pd.Series(index=self.sr_names.values, dtype=float)
        for i_batch, idxes in enumerate(dataloader): 
            ic(i_batch+1, "/", len(dataloader))

            [idxes_valid, meshes_dst] = self._get_items(idxes, flag_data_type=flag_data_type)
            n_meshes_dst = len(meshes_dst) 
            
            points_src = mesh_src.verts_padded().expand((n_meshes_dst, -1, -1))
            points_dst = meshes_dst.verts_padded()
            # points_src = pytorch3d.ops.sample_points_from_meshes(mesh_src, num_samples=n_samples).expand((n_meshes_dst, -1, -1))
            # points_dst = pytorch3d.ops.sample_points_from_meshes(meshes_dst, num_samples=n_samples)

            [losses_verts, _] = pytorch3d.loss.chamfer_distance( points_src, points_dst,batch_reduction=None)
            sr_min = sr_losses[sr_losses==sr_losses[sr_losses.notna()].min()]
            ic(sr_min)
            sr_losses.iloc[idxes_valid] = losses_verts.cpu().numpy()
        return sr_losses

    def get_chamfer_dist_matrix(self, meshes: Meshes) -> torch.Tensor:
        n_meshes = len(meshes)
        loss = torch.eye(n_meshes, device=meshes.device)
        for i in range(n_meshes):
            for j in range(n_meshes):
                # if i == j:
                #     continue
                # else:
                    (loss[i, j], _) = pytorch3d.loss.chamfer_distance(meshes[i].verts_padded(), meshes[j].verts_padded())
        return loss

    def crop_meshes_by_z(self, meshes: Meshes, z: float=7, flag_unitized_length: UNITIZED_LENGTH=UNITIZED_LENGTH.CM) -> Meshes:
        meshes_o3d = convert_meshes_torch3d_to_o3d(meshes)
        for mesh in meshes_o3d:
            bbox = mesh.get_axis_aligned_bounding_box()
            bbox_new = o3d.geometry.AxisAlignedBoundingBox(
                bbox.min_bound, 
                np.array([
                    bbox.max_bound[0], 
                    bbox.max_bound[1], 
                    bbox.min_bound[2] + z * self.unitized_length.value / flag_unitized_length.value
                ])
            )
            meshes_new = [mesh.crop(bbox_new)]
            meshes_crop = covert_meshes_o3d_to_torch3d(meshes=meshes_new, device=self.device)
        return meshes_crop

def change_unitized_length(meshes_src: Meshes, flags_unitized_length_src: UNITIZED_LENGTH, flags_unitized_length_dst: UNITIZED_LENGTH) -> Meshes:
    v = meshes_src.verts_padded()
    v_new = v * flags_unitized_length_dst.value / flags_unitized_length_src.value
    meshes_new = meshes_src.update_padded(v_new)
    return meshes_new
        


if __name__ == '__main__':
    peak_cropped_dataset = PeakFootsDataset(dir_root="/media/veily3/data_ligan/匹克脚部数据集/cropped", flag_side_foot=SIDE_RIGHT, flag_unitized_length=UNITIZED_LENGTH.MM, suffix=".obj")
    
    #peak_cropped_dataset.generate_normalized_dataset(batch_size=5, flag_data_type=DATA_TYPE.NORMALIZED_X)
    # peak_cropped_dataset.generate_aligned_dataset(batch_size=5, flag_data_type=DATA_TYPE.NORMALIZED_X)

    pth_input = Path("/media/veily3/data_ligan/voxel_carving_data/project/ligan_right_0526-1317/result")

    mesh_src = covert_meshes_o3d_to_torch3d([o3d.io.read_triangle_mesh(str(pth_input / "mesh.obj"))])
    mesh_src = change_unitized_length(meshes_src=mesh_src, flags_unitized_length_src=UNITIZED_LENGTH.M, flags_unitized_length_dst=UNITIZED_LENGTH.MM)
    pytorch3d.io.save_obj(pth_input / "right_mm.obj", mesh_src.verts_list()[0], mesh_src.faces_list()[0])

    mesh_src = peak_cropped_dataset.crop_meshes_by_z(mesh_src)
    pytorch3d.io.save_obj(pth_input / "right_crop.obj", mesh_src.verts_list()[0], mesh_src.faces_list()[0])

    [mesh_src, df_factors] = peak_cropped_dataset.normalize(mesh_src, UNITIZED_LENGTH.MM, flag_data_type=DATA_TYPE.NORMALIZED_X)
    pytorch3d.io.save_obj(pth_input / "right_crop_norm.obj", mesh_src.verts_list()[0], mesh_src.faces_list()[0])
    df_factors.to_excel(pth_input / "factors_crop_norm_xyz.xlsx")

    sr_match = peak_cropped_dataset.match_with_all(mesh_src, batch_size=10, flag_data_type=DATA_TYPE.NORMALIZED_X) # 000095 001035 000287
    names_min = sr_match[sr_match==sr_match.min()].index.tolist()
    #names_min = ["000287"]
    ic(names_min) # 000864

    mesh_src = covert_meshes_o3d_to_torch3d([o3d.io.read_triangle_mesh(str(pth_input / "right_crop_norm.obj"))])
    [_, mesh_simi] = peak_cropped_dataset._get_items_by_names(names_min, DATA_TYPE.NORMALIZED_X)
    [_, mesh_gt  ] = peak_cropped_dataset._get_items_by_names(["ligan"], DATA_TYPE.NORMALIZED_X)

    meshes = pytorch3d.structures.join_meshes_as_batch([mesh_src, mesh_simi, mesh_gt])

    sr_factors = pd.read_excel(pth_input / "factors_crop_norm_xyz.xlsx")
    df_factors = pd.read_excel("/media/veily3/data_ligan/匹克脚部数据集/cropped/normalized/XYZ/normalization_factors.xlsx", index_col=0)
    df_factors = df_factors.loc[names_min + ["ligan"], :]
    
    ic(sr_factors)
    S = pytorch3d.transforms.Transform3d()
    S = S.scale(
        x=1/sr_factors.sx, 
        y=1/sr_factors.sy,
        z=1/sr_factors.sz
    ).to(peak_cropped_dataset.device)
    verts_S = S.transform_points(meshes.verts_padded())
    meshes = meshes.update_padded(verts_S)

    T = pytorch3d.transforms.Transform3d()
    T = T.translate(
        x=-sr_factors.tx, 
        y=-sr_factors.ty,
        z=-sr_factors.tz
    ).to(peak_cropped_dataset.device)
    verts_T = T.transform_points(meshes.verts_padded())
    meshes = meshes.update_padded(verts_T)
    ic(meshes.get_bounding_boxes())
    i = 0
    meshes = convert_meshes_torch3d_to_o3d(meshes)
    for m in meshes:
        o3d.io.write_triangle_mesh("test/test_528/{:0>3d}.obj".format(i+1), m, )
        i += 1

    cam_params = pd.read_pickle("/media/veily3/data_ligan/voxel_carving_data/project/ligan_right_0526-1317/data/camera_params.pkl")
    traj = pd.read_pickle("/media/veily3/data_ligan/voxel_carving_data/project/ligan_right_0526-1317/data/trajectory.pkl")
    import cv2

    from src.Camera import PinholeCamera
    cam = PinholeCamera(int(cam_params.height), int(cam_params.width), K=cam_params.intrinsic)
    names_img = traj.index.tolist()
    dir_img = Path(pth_input.parent / "raw")
    pcd_dst = o3d.io.read_point_cloud(str(pth_input / "carving.ply"))
    pcd_dst = pcd_dst.transform(np.diag([1, 1, -1, 1]))
    regis = []
    for [i_mesh, mesh] in enumerate(meshes):
        pcd_src = o3d.geometry.PointCloud(mesh.vertices)
        pcd_src = pcd_src.scale(0.001, np.zeros(3))
        r = o3d.pipelines.registration.registration_icp(
            pcd_src, pcd_dst, 0.02, 
        )
        ic(r.transformation)
        meshes[i_mesh] = mesh.scale(0.001, np.zeros(3)).transform(r.transformation)
        regis.append(r.transformation)

    for name_img in names_img:
        I = cv2.imread(str(dir_img / name_img))
        cam.set_extrinsic_by_rtvec(traj.loc[name_img, "rvec"].astype(float), traj.loc[name_img, "tvec"].astype(float))
        colores = [(0,0,255), (0,255,0), (255,0,0)]
        #I = cam.project_points_on_image(points3d=verts[::5, :], color=(50, 10, 75), radius=1, thickness=-1, img=I)

        J = I.copy()
        for [i_mesh, mesh] in enumerate(meshes):
            pt = np.asarray(mesh.vertices)[::5, :] 
           # l = (pt.max(0)-pt.min(0))[0]
           # pt = pt - np.array([1.5*l, 0, 0])
            #P = cam.project_points(points3d=pt)
            J = cam.project_points_on_image(points3d=pt, color=colores[i_mesh], radius=1, thickness=-1, img=J)
            cv2.namedWindow("", cv2.WINDOW_NORMAL)
        I = cv2.addWeighted(I, 0.8, J, 0.2, 0)
        I = cam.project_axis_on_image(0.1, 10, I)
        cv2.imshow("", I)
        cv2.waitKey(1)
        cv2.imwrite(str(pth_input / name_img), I)
    print()

