
import pytorch3d.structures
import pytorch3d.io
import pytorch3d.renderer
import torch

shader = pytorch3d.renderer.SoftSilhouetteShader()
camera = pytorch3d.renderer.PerspectiveCameras()