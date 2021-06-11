import  cv2
import  os
from icecream import ic
from  pathlib import Path

def imwrite_with_zn(pth_image_with_zn, img):
    cv2.imencode('.png', img)[1].tofile(pth_image_with_zn)
    return 

def viedo2images(pth_video, dir_output=None, downsample=1):
    pth_video    = Path(pth_video)
    dir_viedeo   = pth_video.parents
    prefix_video = pth_video.name

    dir_output = Path(dir_viedeo, prefix_video) if dir_output is None else Path(dir_output)
    dir_output.mkdir(parents=True, exist_ok=True)
    ic(dir_output)

    video = cv2.VideoCapture(str(pth_video))
    cnt = 0
    strat_reading =True
    while strat_reading:
        [strat_reading, img] = video.read()
        if strat_reading:
            if cnt % downsample == 0:
                pth_img = Path(dir_output, "{:0>6d}.png".format(cnt + 1))
                imwrite_with_zn(str(pth_img), img)
                ic(pth_img.name)
            cnt += 1
    video.release()
    return


if __name__ == "__main__":
    pth_video  = Path("/media/veily3/data_ligan/voxel_carving_data/video/ligan_right_0526-1848/afe534bb07dd73aac035b42ff88b759d.mp4")
    dir_output = pth_video.parents[2] / "image" / pth_video.parent.name
    viedo2images(
        pth_video=pth_video, 
        dir_output=dir_output, 
        downsample=1,
    )
    