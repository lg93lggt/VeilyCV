
import sys
from typing import List, Literal, Sequence, Tuple, Union

from torch.nn.functional import normalize
from piplines.foot_3d_reconstruction.main_aruco import UNITIZED_LENGTH
import pytorch3d.io
import pytorch3d.structures
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


sys.path.append("../..")
from src.utils.debug import debug_separator, debug_vis
from src.plugin.geometry3d import cvt_meshes_torch3d_to_o3d



class SIDE_OF_FOOT(Enum):
    LEFT  = auto()
    RIGHT = auto()
SIDE_LEFT  = SIDE_OF_FOOT.LEFT
SIDE_RIGHT = SIDE_OF_FOOT.RIGHT


@ unique
class DATA_TYPE(Flag):
    RAW            = 0
    NORMALIZED_X   = 1
    NORMALIZED_Y   = 2
    NORMALIZED_Z   = 4
    NORMALIZED_XY  = NORMALIZED_X | NORMALIZED_Y
    NORMALIZED_XYZ = NORMALIZED_X | NORMALIZED_Y | NORMALIZED_Z
    
UNITIZED_LENGTH = UNITIZED_LENGTH


@ debug_separator
class PeakFootDataset(Dataset):
    """
    """
    def __init__(self, dir_root: Union[str, Path], suffix=".ply", flag_side_foot: SIDE_OF_FOOT=SIDE_OF_FOOT.RIGHT, flag_unitized_length: UNITIZED_LENGTH=UNITIZED_LENGTH.MM, device: str="cuda:0"):
        super().__init__()
        self.unitized_length = flag_unitized_length
        self.side_foot = "right" if (flag_side_foot == SIDE_OF_FOOT.RIGHT) else "left"

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

        ic(PeakFootDataset, dir_root, flag_side_foot, flag_unitized_length, len(self))
        return

    def __getitem__(self, idx: int):
        super().__init__()
        return idx
        # else:
        #     raise FileExistsError("The PLY file does not exist.")

    def __len__(self):
        super().__init__()
        return len(self.pthes_row)

    def _get_items_by_name(self, names: List[str], flag_data_type: DATA_TYPE=DATA_TYPE.RAW) -> Union[List[int], pytorch3d.structures.Meshes]:
        pthes_mesh   = []
        idxes_exist = []
        for name in names:
            dir_mesh = self.dict_dirs_data[flag_data_type]
            pth_mesh = Path(dir_mesh, name, self.side_foot + self.suffix)
            if pth_mesh.exists():
                pthes_mesh.append(pth_mesh)
                idxes_exist.append((self.sr_names[self.sr_names == name]).index[0])
            else:
                continue

        meshes = []
        idxes_valid = []
        for [idx_ply, pth_mesh] in enumerate(pthes_mesh):
            try:
                [verts, faces] = pytorch3d.io.load_ply(pth_mesh)
                mesh = pytorch3d.structures.Meshes([verts], [faces]).to(self.device)
                meshes.append(mesh)
                idxes_valid.append(int(idxes_exist[idx_ply]))
            except :
                continue
        meshes = pytorch3d.structures.join_meshes_as_batch(meshes)
        meshes = meshes.to(self.device)
        ic(len(meshes), meshes.verts_padded().shape, meshes.faces_padded().shape)
        return [idxes_valid, meshes]

    @ debug_separator
    def _get_items(self, idxes: List[int] or torch.Tensor, flag_data_type: DATA_TYPE=DATA_TYPE.RAW) -> Union[List[int], pytorch3d.structures.Meshes]:
        pthes_mesh   = []
        idxes_exist = []
        idxes = idxes.cpu().numpy().tolist() if isinstance(idxes, torch.Tensor) else idxes
        for idx in idxes:
            dir_mesh = self.dict_dirs_data[flag_data_type]
            pth_mesh = Path(dir_mesh, self.sr_names.iloc[idx], self.side_foot + self.suffix)
            if pth_mesh.exists():
                pthes_mesh.append(pth_mesh)
                idxes_exist.append(idx)
            else:
                continue
        
        if   self.suffix == ".obj":
            meshes = pytorch3d.io.load_objs_as_meshes(pthes_mesh, device=self.device)
            idxes_valid = idxes_exist
        elif self.suffix == ".ply":  
            meshes = []
            idxes_valid = []
            for [idx_ply, pth_mesh] in enumerate(pthes_mesh):
                try:
                    [verts, faces] = pytorch3d.io.load_ply(pth_mesh)
                    mesh = pytorch3d.structures.Meshes([verts], [faces]).to(self.device)
                    meshes.append(mesh)
                    idxes_valid.append(int(idxes_exist[idx_ply]))
                except :
                    continue
            meshes = pytorch3d.structures.join_meshes_as_batch(meshes)
            meshes = meshes.to(self.device)
        names = self.sr_names.iloc[idxes_valid].values.tolist()
        ic(len(meshes), names, meshes.verts_padded().shape, meshes.faces_padded().shape)
        return [idxes_valid, names, meshes]

    def normalize(self, meshes: pytorch3d.structures.Meshes, flag_unitized_length: UNITIZED_LENGTH, flag_data_type: DATA_TYPE=DATA_TYPE.NORMALIZED_XYZ) \
            -> Tuple[pytorch3d.structures.Meshes, pd.DataFrame]:

        meshes = meshes.to(self.device) if (meshes.device != self.device) else meshes
        n_mesh = len(meshes)
        max_len = meshes.get_bounding_boxes()[:, :, 1] - meshes.get_bounding_boxes()[:, :, 0] 
        center  = (meshes.get_bounding_boxes()[:, :, 1] + meshes.get_bounding_boxes()[:, :, 0]) / 2 

        """
        Translate
        """
        tx = -center[:, 0] if (flag_data_type & DATA_TYPE.NORMALIZED_X) else torch.zeros(n_mesh, dtype=torch.float32, device=self.device)
        ty = -center[:, 1] if (flag_data_type & DATA_TYPE.NORMALIZED_Y) else torch.zeros(n_mesh, dtype=torch.float32, device=self.device)
        tz = -center[:, 2] if (flag_data_type & DATA_TYPE.NORMALIZED_Z) else torch.zeros(n_mesh, dtype=torch.float32, device=self.device)
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
        if pth_output_xlsx.exists():
            df_factors_all = pd.read_excel(pth_output_xlsx)
        else:
            df_factors_all = pd.DataFrame(index=self.sr_names, columns=["tx", "ty", "tz", "sx", "sy", "sz"], )
        ic(dir_output_parent)
        for i_batch, idxes in enumerate(dataloader):
            
            ic(i_batch, "/", len(dataloader))
            [idxes_valid, names, meshes] = self._get_items(idxes, flag_data_type=DATA_TYPE.RAW)
            if names != "000766":
                continue
            
            [meshes, df_factors] = self.normalize(meshes, flag_data_type=flag_data_type, flag_unitized_length=self.unitized_length)

            df_factors.index = self.sr_names[idxes_valid]
            df_factors_all.iloc[idxes_valid, :] = df_factors.values

            for [idx_mesh, idx_valid] in enumerate(idxes_valid):
                dir_output = dir_output_parent / self.sr_names.iloc[idx_valid]
                dir_output.mkdir(parents=True, exist_ok=True, )
                pth_output = Path(dir_output, self.side_foot + self.suffix)
                pytorch3d.io.save_ply(f=pth_output, verts=meshes.verts_list()[idx_mesh].cpu(), faces=meshes.faces_list()[idx_mesh].cpu(), ascii=True)
                ic(pth_output.parent.name)

                """
                DEBUG
                """
                if False:
                    ms = o3d.io.read_triangle_mesh(str(pth_output))
                    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(1.)
                    sphere =  o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_sphere(1.))
                    o3d.visualization.draw_geometries([ sphere, ms, coord])
        df_factors_all.to_excel(pth_output_xlsx)
        ic(df_factors_all.info())
        ic(pth_output_xlsx)
        return

    @ debug_separator
    def match(self, mesh_src: pytorch3d.structures.Meshes, batch_size: int=128, n_samples: int=2000, flag_data_type: DATA_TYPE=DATA_TYPE.NORMALIZED_XY) \
            -> Tuple[torch.Tensor, int]:
        mesh_src = mesh_src.to(self.device)
        # [mesh_src, df_factors] = self.normalize(mesh_src, flag_data_type=flag_data_type, flag_unitized_length=UNITIZED_LENGTH.M)

        dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=False, drop_last=False)
        
        sr_losses = pd.Series(index=self.sr_names.values, dtype=float)
        for i_batch, idxes in enumerate(dataloader): 
            ic(i_batch, "/", len(dataloader))

            [idxes_valid, meshes_dst] = self._get_items(idxes, flag_data_type=flag_data_type)
            n_meshes_dst = len(meshes_dst) 
            
            points_dst = pytorch3d.ops.sample_points_from_meshes(meshes_dst, num_samples=n_samples)
            points_src = pytorch3d.ops.sample_points_from_meshes(mesh_src, num_samples=n_samples).expand((n_meshes_dst, -1, -1))

            [losses_verts, _] = pytorch3d.loss.chamfer_distance(points_dst, points_src, batch_reduction=None)
            [loss_min, idx_min] = torch.min(losses_verts, dim=0)
            ic(sr_losses[sr_losses==sr_losses[sr_losses.notna()].min()])
            sr_losses.iloc[idxes_valid] = losses_verts.cpu().numpy()
        return sr_losses

    def get_chamfer_dist_matrix(self, flag_data_type =DATA_TYPE.NORMALIZED_XY):
        for name in self.sr_names.values:
            mesh = self._get_items_by_name[name]
            [] = self.match(mesh, batch_size=1, n_samples=2000, flag_data_type=flag_data_type)
        return


        
if __name__ == '__main__':
    peak_dataset = PeakFootDataset(dir_root="/media/veily3/data_ligan/匹克脚部数据集/cropped_simplified_1000", suffix=".obj", flag_side_foot=SIDE_RIGHT, flag_unitized_length=UNITIZED_LENGTH.MM, )
    peak_dataset.generate_normalized_dataset(batch_size=1, flag_data_type=DATA_TYPE.NORMALIZED_X)
    
    
    # pth_mesh_src = Path("/media/veily3/data_ligan/匹克脚部数据集/raw/ligan/right.ply")
    # [verts, faces] = pytorch3d.io.load_ply(pth_mesh_src)
    # mesh_src = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
    # [mesh_src_norm, df_factors] = peak_dataset.normalize(mesh_src, flag_data_type=DATA_TYPE.NORMALIZED_X, flag_unitized_length=UNITIZED_LENGTH.MM,)
    # pytorch3d.io.save_ply("/media/veily3/data_ligan/匹克脚部数据集/normalized/X/ligan/right.ply", mesh_src_norm.verts_list()[0].cpu(), mesh_src_norm.faces_list()[0].cpu(), )

    # pth_mesh_src = Path("/media/veily3/data_ligan/voxel_carving_data/result/ligan/normright.ply")
    # [verts, faces] = pytorch3d.io.load_ply(pth_mesh_src)
    # mesh_src_norm = pytorch3d.structures.Meshes(verts=[verts], faces=[faces])
    

    # sr_losses = peak_dataset.match(mesh_src=mesh_src_norm, batch_size=64, flag_data_type=DATA_TYPE.NORMALIZED_XY)   # "000436"

    print()

