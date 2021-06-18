
from pathlib import Path
from typing import Union
import os
from icecream import  ic


def apply_meshlab_mlx_script(pth_input_mesh: Union[Path, str], pth_output_mesh: Union[Path, str], pth_mlx_script: Union[Path, str]):
    meshlabserver = "meshlabserver"
    pth_output_mesh: Path = Path(pth_output_mesh)
    pth_output_mesh.parent.mkdir(parents=True, exist_ok=True)
    scriptname = " -s {}".format(str(pth_mlx_script))
    cmd = meshlabserver + " -i " + str(pth_input_mesh) + " -o " + str(pth_output_mesh) + " -m vc vn"  + scriptname
    ic(cmd)
    os.system(cmd)
    return
    
