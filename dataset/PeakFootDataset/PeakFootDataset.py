
import sys
from typing import List, Iterable, Sequence, Tuple, Union, Optional, Dict, Any

from piplines.foot_3d_reconstruction.main_aruco import UNITIZED_LENGTH
import pytorch3d.io
import pytorch3d.structures
import pytorch3d.loss
import pytorch3d.transforms
import pytorch3d.ops
import numpy as np
from icecream import ic
from pathlib import Path
import docx
import os
import pandas as pd
from enum import Enum, auto, IntEnum, Flag, unique
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SequentialSampler 
import open3d as o3d
import multiprocessing

sys.path.append("../..")
from src.utils.debug import debug_separator, debug_vis
from src.plugins.geometry3d import cvt_meshes_torch3d_to_o3d
from src.vl3d.pytorch3d.iterative_closest_point import iterative_closest_point
from src.vl3d.pytorch3d.crop_meshes import crop_meshes_by_z


@ unique
class SIDE_OF_FOOT(Enum):
    LEFT  = auto()
    RIGHT = auto()

@ unique
class NORMALIZE_TYPE(Flag):
    NONE = 0
    X    = 1
    Y    = 2
    Z    = 4
    XY   = X | Y
    XYZ  = X | Y | Z
    
@ unique
class DATA_STRUCTURE(Flag):
    NONE        = 0
    MESH_LEFT   = 1
    MESH_RIGHT  = 2
    LABEL_LEFT  = 4
    LABEL_RIGHT = 8
    MESH_ALL    =  MESH_LEFT |  MESH_RIGHT
    LABEL_ALL   = LABEL_LEFT | LABEL_RIGHT
    ALL         =  MESH_ALL  | LABEL_ALL
    
@ unique
class SUFFIX(Enum):
    PLY  = ".ply"
    OBJ  = ".obj"
    DOCX = ".docx"

@ debug_separator
class PeakFootDatasetBase(Dataset):
    """
    """
    def __init__(self, dir_root: Union[Path, str], flag_data_structure: Optional[DATA_STRUCTURE]=DATA_STRUCTURE.ALL, device: str="cuda:0") -> None:
        super().__init__()
        self.dir_root         : Path = Path(dir_root)
        self.pth_info         : Path = Path(self.dir_root, "info.xlsx")
        self.pth_poi_template : Path = Path(self.dir_root, "poi_template.xlsx")
        self.dir_data         : Path = Path(self.dir_root, "data")
        self.name: str = self.dir_root.stem
        
        self.unitized_length: UNITIZED_LENGTH = UNITIZED_LENGTH.MM
        self.suffix_mesh    : SUFFIX          = SUFFIX.PLY
        self.data_structure : DATA_STRUCTURE  = flag_data_structure
        
        self.device: str = device
        self.dataloader = self.set_dataloader()

        self.__names_all: List[str] = sorted(p.stem for p in self.dir_data.glob("*/"))
        self.sr_names_valid: pd.Series = self._get_names_valid()
        
        self._get_df_poi_template()

        ic(dir_root, self.name, self.unitized_length, self.suffix_mesh, self.sr_names_valid.__len__())
        return
    
    def __len__(self):
        super().__init__()
        return len(self.sr_names_valid)
    
    def __getitem__(self, idx: int) -> str:
        super().__init__()
        name: str = self.sr_names_valid.iloc[idx]
        return name
    
    def _check_data_valid(self) -> pd.DataFrame:
        """---
        _check_data_valid
        -------
        Check the valid data.

        Returns
        -------
        df_info [pd.DataFrame]
            [description]
        """        
        df_info: pd.DataFrame
        if self.pth_info.exists():
            df_info = pd.read_excel(self.pth_info, index_col=0)  
        else:
            df_info = pd.DataFrame(
                index=self.__names_all, 
                columns=[
                    SIDE_OF_FOOT.LEFT.name.lower(), 
                    SIDE_OF_FOOT.RIGHT.name.lower(), 
                    SUFFIX.DOCX.name.lower()
                ], 
                dtype=np.bool
            )
        ic(df_info.columns.tolist())
        for (idx, name) in enumerate(self.__names_all): 
            process = "{0} / {1}".format(idx, len(self.__names_all))
            ic(process)
            side: SIDE_OF_FOOT
            for side in SIDE_OF_FOOT:
                pth_mesh = Path(self.dir_data, name, side.name.lower() + self.suffix_mesh.value.lower())
                if pth_mesh.exists():
                    df_info.loc[name, side.name.lower()] = True
                else:
                    df_info.loc[name, side.name.lower()] = False
            pth_docx = Path(self.dir_data, name, "report.docx")            
            if pth_docx.exists():
                df_info.loc[name, "docx"] = True
            else:
                df_info.loc[name, "docx"] = False
            ic(name, df_info.loc[name].values.tolist())
        df_info.to_excel(self.pth_info)
        ic(self.pth_info)
        return df_info
        
    def _get_names_valid(self) -> pd.Series:
        df_info: pd.DataFrame
        if self.pth_info.exists():
            df_info = pd.read_excel(self.pth_info, index_col=0)
        else:
            df_info = self._check_data_valid()
        sr_valid_names: pd.Series = pd.Series(df_info.index[df_info.all(axis=1)].tolist())
        return sr_valid_names

    def set_data_structure(flag_data_structure: DATA_STRUCTURE):
        self.data_structure: DATA_STRUCTURE = flag_data_structure
        return

    def set_dataloader(self, **kwargs: Optional[Any]) -> None:
        self.dataloader: PeakFootDataLoader = PeakFootDataLoader(self, **kwargs)
        return

    def get_pth_mesh_by_name(self, name: str, flag_side_of_foot: SIDE_OF_FOOT) -> Path:
        pth_mesh: Path = Path(self.dir_data, name, flag_side_of_foot.name.lower() + self.suffix_mesh.value)
        return pth_mesh
    
    def get_pth_label_by_name(self, name: str) -> Path:
        pth_poi: Path = Path(self.dir_data, name, "report" + SUFFIX.DOCX.value)
        return pth_poi

    def _getitems_by_names(self, names: List[str]) -> Dict:
        batch_data = {"names": names}
        
        if (DATA_STRUCTURE.LABEL_LEFT & self.data_structure):
            batch_data[DATA_STRUCTURE.LABEL_LEFT] : pd.DataFrame = pd.DataFrame(index=self.df_poi_template.index, columns=names)
                
        if (DATA_STRUCTURE.LABEL_RIGHT & self.data_structure):
            batch_data[DATA_STRUCTURE.LABEL_RIGHT]: pd.DataFrame = pd.DataFrame(index=self.df_poi_template.index, columns=names)
            
        if self.suffix_mesh is SUFFIX.PLY:
            list_meshes_l: List = []
            list_meshes_r: List = []
        
        pthes_mesh_l: List[path] = []
        pthes_mesh_r: List[path] = []
        for name in names:
            """
            # load labels
            """
            pth_poi = Path(self.dir_data, name, "report.docx")
            df_poi = self.load_poi_from_docx(pth_poi)
            if (DATA_STRUCTURE.LABEL_LEFT & self.data_structure):
                batch_data[DATA_STRUCTURE.LABEL_LEFT ].loc[:, name] = df_poi.iloc[:, 0]
                
            if (DATA_STRUCTURE.LABEL_RIGHT & self.data_structure):
                batch_data[DATA_STRUCTURE.LABEL_RIGHT].loc[:, name] = df_poi.iloc[:, 1]
                
            """
            # load ply mesh
            """
            if self.suffix_mesh is SUFFIX.PLY:
                if (DATA_STRUCTURE.MESH_LEFT & self.data_structure):                    
                    (verts, faces) = pytorch3d.io.load_ply(self.get_pth_mesh_by_name(name=name, flag_side_of_foot=SIDE_OF_FOOT.LEFT))
                    mesh_l = pytorch3d.structures.Meshes(
                        verts=verts.to(dtype=torch.float32, device=self.device).unsqueeze(0), 
                        faces=faces.to(dtype=torch.float32, device=self.device).unsqueeze(0),
                    )
                    list_meshes_l.append(mesh_l)
                
                if (DATA_STRUCTURE.MESH_RIGHT & self.data_structure):
                    (verts, faces) = pytorch3d.io.load_ply(self.get_pth_mesh_by_name(name=name, flag_side_of_foot=SIDE_OF_FOOT.RIGHT))
                    mesh_r = pytorch3d.structures.Meshes(
                        verts=verts.to(dtype=torch.float32, device=self.device).unsqueeze(0), 
                        faces=faces.to(dtype=torch.float32, device=self.device).unsqueeze(0),
                    )
                    list_meshes_r.append(mesh_r)
            elif self.suffix_mesh is SUFFIX.OBJ:
                if (DATA_STRUCTURE.MESH_LEFT  & self.data_structure):
                    pthes_mesh_l.append(self.get_pth_mesh_by_name(name=name, flag_side_of_foot=SIDE_OF_FOOT.LEFT))
                if (DATA_STRUCTURE.MESH_RIGHT & self.data_structure):
                    pthes_mesh_r.append(self.get_pth_mesh_by_name(name=name, flag_side_of_foot=SIDE_OF_FOOT.RIGHT))

        """
        # load ply meshes as batch
        """
        if self.suffix_mesh is SUFFIX.PLY:
            if (DATA_STRUCTURE.MESH_LEFT & self.data_structure):
                batch_data[DATA_STRUCTURE.MESH_LEFT] = pytorch3d.structures.join_meshes_as_batch(list_meshes_l)
            
            if (DATA_STRUCTURE.MESH_RIGHT & self.data_structure):
                batch_data[DATA_STRUCTURE.MESH_RIGHT] = pytorch3d.structures.join_meshes_as_batch(list_meshes_r)
        elif self.suffix_mesh is SUFFIX.OBJ:
            if (DATA_STRUCTURE.MESH_LEFT & self.data_structure):
                batch_data[DATA_STRUCTURE.MESH_LEFT] = pytorch3d.io.load_objs_as_meshes(pthes_mesh_l, device=self.device)
            
            if (DATA_STRUCTURE.MESH_RIGHT & self.data_structure):
                batch_data[DATA_STRUCTURE.MESH_RIGHT] = pytorch3d.io.load_objs_as_meshes(pthes_mesh_r, device=self.device)
        return batch_data

    def _get_df_poi_template(self) -> pd.DataFrame:
        assert self.pth_poi_template.exists(), "File df_poi_template.xlsx in {} does not exist.".format(str(self.pth_poi_template))
        self.df_poi_template: pd.DataFrame = pd.read_excel(self.pth_poi_template, index_col=0, header=0)
        return self.df_poi_template
    
    def load_poi_from_docx(self, pth_poi: Union[Path, str]) -> pd.DataFrame:
        doc = docx.Document(pth_poi)
        df_poi : pd.DataFrame = self.df_poi_template.copy(deep=True)
        
        for table in doc.tables:
            idxes_cell = [3, 6]
            for i_cell in idxes_cell:
                name_side: str = table.rows[4].cells[i_cell].text
                assert (name_side in df_poi.columns)
                for (_, row) in enumerate(table.rows[5:20]):
                    name_param: str = row.cells[1].text
                    if name_param in df_poi.index: 
                        df_poi.loc[name_param, name_side] = row.cells[i_cell].text
        
            idxes_row = [-3, -2]
            for i_row in idxes_row:
                for (i_cell, _) in enumerate(table.rows[i_row].cells):
                    name_side: str = table.rows[i_row].cells[i_cell].text[:-1]
                    if (name_side in df_poi.columns):
                        text_tmp: str = ""
                        for (i_col, _) in enumerate(table.rows[-4].cells):
                            name_param: str = table.rows[-4].cells[i_col].text
                            if "足弓评" in name_param:
                                text_tmp = table.rows[i_row].cells[i_col].text
                                if "型" in text_tmp:
                                    df_poi.loc["足弓分类", name_side] = text_tmp.split()[-1][0]
                                    break
                        break
        return df_poi
    
    @ debug_separator
    def collate_fn(self, batch_names: Iterable[str]):
        batch_data = self._getitems_by_names(batch_names)
        batch_size = len(batch_data["names"])
        ic(batch_data["names"], batch_size)
        if (DATA_STRUCTURE.MESH_LEFT & self.data_structure):
            ic(batch_data[DATA_STRUCTURE.MESH_LEFT ].faces_padded().shape)
        if (DATA_STRUCTURE.MESH_RIGHT & self.data_structure):
            ic(batch_data[DATA_STRUCTURE.MESH_RIGHT].faces_padded().shape)
        if (DATA_STRUCTURE.LABEL_LEFT & self.data_structure):
            ic(batch_data[DATA_STRUCTURE.LABEL_LEFT ].info())
        if (DATA_STRUCTURE.LABEL_RIGHT & self.data_structure):
            ic(batch_data[DATA_STRUCTURE.LABEL_RIGHT].info())
        return batch_data
    
    @ staticmethod
    def normalize_mesh(
        meshes: pytorch3d.structures.Meshes,  
        flag_normalize_type: NORMALIZE_TYPE=NORMALIZE_TYPE.X, 
        device: Optional[str]="cuda:0"
    ) -> Tuple[pytorch3d.structures.Meshes, pd.DataFrame]:

        meshes = meshes.to(device) if (meshes.device != device) else meshes
        n_mesh = len(meshes)
        max_len = meshes.get_bounding_boxes()[:, :, 1] - meshes.get_bounding_boxes()[:, :, 0] 
        center  = (meshes.get_bounding_boxes()[:, :, 1] + meshes.get_bounding_boxes()[:, :, 0]) / 2 

        # Translate
        tx = -center[:, 0] if (flag_normalize_type & NORMALIZE_TYPE.X) else torch.zeros(n_mesh, dtype=torch.float32, device=device)
        ty = -center[:, 1] if (flag_normalize_type & NORMALIZE_TYPE.Y) else torch.zeros(n_mesh, dtype=torch.float32, device=device)
        tz = -center[:, 2] if (flag_normalize_type & NORMALIZE_TYPE.Z) else torch.zeros(n_mesh, dtype=torch.float32, device=device)
        
        T = pytorch3d.transforms.Transform3d()
        
        T = T.translate(
            x=tx, 
            y=ty,
            z=tz
        ).to(device) # 脚底中心
        verts_T = T.transform_points(meshes.verts_padded())
        meshes = meshes.update_padded(verts_T)

        # Scale
        [sx, sy, sz] = torch.ones((3, n_mesh), dtype=torch.float32, device=device)
        if  (flag_normalize_type == NORMALIZE_TYPE.X):
            [sx, sy, sz] = (1/max_len[:, 0]).expand(3, n_mesh)
        elif (flag_normalize_type == NORMALIZE_TYPE.Y):
            [sx, sy, sz] = (1/max_len[:, 1]).expand(3, n_mesh)
        elif (flag_normalize_type == NORMALIZE_TYPE.Z):
            [sx, sy, sz] = (1/max_len[:, 2]).expand(3, n_mesh)
        elif (flag_normalize_type == NORMALIZE_TYPE.XY):
            sx = 2 / max_len[:, 0]
            sy = 2 / max_len[:, 1]
            sz = 1
        elif (flag_normalize_type == NORMALIZE_TYPE.XYZ):
            sx = 2 / max_len[:, 0]
            sy = 2 / max_len[:, 1]
            sz = 2 / max_len[:, 2]
            
        if (max_len==0).any():
            raise ZeroDivisionError("Max length of mesh is ZERO.")
        else:
            S = pytorch3d.transforms.Transform3d()
            S = S.scale(
                x=sx, 
                y=sy,
                z=sz
            ).to(device)
            verts_S = S.transform_points(meshes.verts_padded())
            meshes = meshes.update_padded(verts_S)

        df_factors = pd.DataFrame(
            columns=["tx", "ty", "tz", "sx", "sy", "sz"], 
            data=torch.stack((tx, ty, tz, sx, sy, sz), dim=1).cpu().numpy()
        )
        return [meshes, df_factors]

    def generate_cropped_dataset(self, dir_output_dataset: Optional[Union[Path, str, None]]=None, batch_size: Optional[int]=1, flags_output_suffix: SUFFIX=SUFFIX.OBJ):
        dir_output_data = Path(dir_output_dataset) if (dir_output_dataset is not None) else Path(self.dir_root.parent, "cropped_z", "data")
        ic(dir_output_data)
        self.set_dataloader(batch_size=batch_size,)
        for batch_data in self.dataloader:
            meshes_cropped_l = crop_meshes_by_z(meshes=batch_data[DATA_STRUCTURE.MESH_LEFT ], z=70.)
            meshes_cropped_r = crop_meshes_by_z(meshes=batch_data[DATA_STRUCTURE.MESH_RIGHT], z=70.)
            for (idx_mesh, name) in enumerate(batch_data["names"]):
                pth_output_l = Path(dir_output_data, name, SIDE_OF_FOOT.LEFT .name.lower() + flags_output_suffix.value)
                pth_output_r = Path(dir_output_data, name, SIDE_OF_FOOT.RIGHT.name.lower() + flags_output_suffix.value)
                pth_output_l.parent.mkdir(exist_ok=True, parents=True)
                pth_output_r.parent.mkdir(exist_ok=True, parents=True)
                
                if   flags_output_suffix is SUFFIX.OBJ:
                    pytorch3d.io.save_obj(f=pth_output_l, verts=meshes_cropped_l.verts_list()[idx_mesh].cpu(), faces=meshes_cropped_l.faces_list()[idx_mesh].cpu())
                    pytorch3d.io.save_obj(f=pth_output_r, verts=meshes_cropped_r.verts_list()[idx_mesh].cpu(), faces=meshes_cropped_r.faces_list()[idx_mesh].cpu())
                elif flags_output_suffix is SUFFIX.PLY:
                    pytorch3d.io.save_ply(f=pth_output_l, verts=meshes_cropped_l.verts_list()[idx_mesh].cpu(), faces=meshes_cropped_l.faces_list()[idx_mesh].cpu())
                    pytorch3d.io.save_ply(f=pth_output_r, verts=meshes_cropped_r.verts_list()[idx_mesh].cpu(), faces=meshes_cropped_r.faces_list()[idx_mesh].cpu())
                else:
                    raise TypeError("Unsupported output suffix {}".format(flags_output_suffix))
                ic(name)
                ic(pth_output_l.name)
                ic(pth_output_r.name)
        return 

    @ debug_separator
    def generate_normalized_dataset(
        self, 
        dir_output_dataset: Optional[Union[Path, str, None]]=None, 
        batch_size: Optional[int]=2, 
        flag_normalize_type: Optional[NORMALIZE_TYPE]=NORMALIZE_TYPE.X,
        flags_output_suffix: Optional[SUFFIX]=SUFFIX.OBJ,
    ):
        if (dir_output_dataset is not None):
            dir_output_data = Path(dir_output_dataset)  
        else:
            dir_output_data = Path(self.dir_root.parent, "normalized_" + flag_normalize_type.name.lower(), "data")
        ic(dir_output_data)
        df_all_l = pd.DataFrame()
        df_all_r = pd.DataFrame()
        self.set_dataloader(batch_size=batch_size)
        for batch_data in self.dataloader:
            (meshes_trans_l, df_l) = self.normalize_mesh(meshes=batch_data[DATA_STRUCTURE.MESH_LEFT ], device=self.device)
            (meshes_trans_r, df_r) = self.normalize_mesh(meshes=batch_data[DATA_STRUCTURE.MESH_RIGHT], device=self.device)
            for (idx_mesh, name) in enumerate(batch_data["names"]):
                pth_output_l = Path(dir_output_data, name, SIDE_OF_FOOT.LEFT .name.lower() + flags_output_suffix.value)
                pth_output_r = Path(dir_output_data, name, SIDE_OF_FOOT.RIGHT.name.lower() + flags_output_suffix.value)
                pth_output_l.parent.mkdir(exist_ok=True, parents=True)
                pth_output_r.parent.mkdir(exist_ok=True, parents=True)
                
                if   flags_output_suffix is SUFFIX.OBJ:
                    pytorch3d.io.save_obj(f=pth_output_l, verts=meshes_trans_l.verts_list()[idx_mesh].cpu(), faces=meshes_trans_l.faces_list()[idx_mesh].cpu())
                    pytorch3d.io.save_obj(f=pth_output_r, verts=meshes_trans_r.verts_list()[idx_mesh].cpu(), faces=meshes_trans_r.faces_list()[idx_mesh].cpu())
                elif flags_output_suffix is SUFFIX.PLY:
                    pytorch3d.io.save_ply(f=pth_output_l, verts=meshes_trans_l.verts_list()[idx_mesh].cpu(), faces=meshes_trans_l.faces_list()[idx_mesh].cpu())
                    pytorch3d.io.save_ply(f=pth_output_r, verts=meshes_trans_r.verts_list()[idx_mesh].cpu(), faces=meshes_trans_r.faces_list()[idx_mesh].cpu())
                else:
                    raise TypeError("Unsupported output suffix {}".format(flags_output_suffix))
                ic(name)
                ic(pth_output_l.name)
                ic(pth_output_r.name)
            df_l.index = [batch_data["names"]]
            df_r.index = [batch_data["names"]]
            df_all_l = df_all_l.append(df_l)
            df_all_r = df_all_r.append(df_r)
            ic(df_l)
            ic(df_r) 
            with pd.ExcelWriter(dir_output_data.parent / "normalization_paramters.xlsx") as writer:
                df_all_l.to_excel(writer, sheet_name="left")
                df_all_r.to_excel(writer, sheet_name="right")
        return
    
    @ debug_separator
    def match(self, mesh_src: pytorch3d.structures.Meshes, batch_size: int=128, n_samples: int=2000, flag_data_type: NORMALIZE_TYPE=NORMALIZE_TYPE.XY) \
            -> Tuple[torch.Tensor, int]:
        mesh_src = mesh_src.to(self.device)
        # [mesh_src, df_factors] = self.normalize(mesh_src, flag_data_type=flag_data_type, flag_unitized_length=UNITIZED_LENGTH.M)

        dataloader = DataLoader(dataset=self, batch_size=batch_size, shuffle=False, drop_last=False)
        
        sr_losses = pd.Series(index=self.sr_names.values, dtype=float)
        for i_batch, idxes in enumerate(dataloader): 
            ic(i_batch, "/", len(dataloader))

            [idxes_valid, names, meshes]=  self._get_items(idxes, flag_data_type=flag_data_type)
            n_meshes_dst = len(meshes_dst) 
            
            points_dst = pytorch3d.ops.sample_points_from_meshes(meshes_dst, num_samples=n_samples)
            points_src = pytorch3d.ops.sample_points_from_meshes(mesh_src, num_samples=n_samples).expand((n_meshes_dst, -1, -1))

            [losses_verts, _] = pytorch3d.loss.chamfer_distance(points_dst, points_src, batch_reduction=None)
            [loss_min, idx_min] = torch.min(losses_verts, dim=0)
            ic(sr_losses[sr_losses==sr_losses[sr_losses.notna()].min()])
            sr_losses.iloc[idxes_valid] = losses_verts.cpu().numpy()
        return sr_losses

    def calc_chamfer_dist_matrix(self, flag_data_type =NORMALIZE_TYPE.XY):
        for name in self.sr_names.values:
            mesh = self._get_items_by_name[name]
            [] = self.match(mesh, batch_size=1, n_samples=2000, flag_data_type=flag_data_type)
    
 
class PeakFootDataLoader(DataLoader):
    def __init__(self, dataset: PeakFootDatasetBase, **kwargs: Any):
        super().__init__(dataset=dataset, **kwargs, collate_fn=dataset.collate_fn)
        return 
      
        
if __name__ == '__main__':
    peak_dataset = PeakFootDatasetBase(dir_root="/media/veily3/data_ligan/匹克脚部数据集/simplify_2000", flag_data_structure=DATA_STRUCTURE.ALL)
    peak_dataset.suffix_mesh = SUFFIX.OBJ
    from scripts.MeshLab.apply_meshlab_script import apply_meshlab_mlx_script       
    # peak_dataset.generate_normalized_dataset()
    dir_output_dataset = Path("/media/veily3/data_ligan/匹克脚部数据集/normalized_x_filled/data")
    for name in peak_dataset.sr_names_valid:
        apply_meshlab_mlx_script(Path("/media/veily3/data_ligan/匹克脚部数据集/normalized_x/data", name, "left.obj"), Path(dir_output_dataset, name, "left.obj"), "scripts/MeshLab/fill_holes.mlx")
        apply_meshlab_mlx_script(Path("/media/veily3/data_ligan/匹克脚部数据集/normalized_x/data", name, "right.obj"), Path(dir_output_dataset, name, "right.obj"),"scripts/MeshLab/fill_holes.mlx")
    print()

