

import  cv2
import numpy as np
import  glob
import  os

if __name__ == "__main__":
    
    dir_images_1 = "test_422/test1/out"
    dir_images_2 = "test_422/test1/"
    pths_imagse_1 = sorted(glob.glob(os.path.join(dir_images_1, "*.jpg")))
    pths_imagse_2 = sorted(glob.glob(os.path.join(dir_images_2, "*.jpg")))

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img_1 = cv2.imread(pths_imagse_1[0])
    [H, W] = img_1.shape[:2]

    video = cv2.VideoWriter('test_422/test1/carve.avi', fourcc, 1.0, (W * 2, H),True)

    for idx in range(len(pths_imagse_1)):
        print("{:0>6d}".format(idx))
        img_for_video = np.zeros((H, W * 2, 3)).astype(np.uint8)
        img_1 = cv2.imread(pths_imagse_1[idx])
        img_2 = cv2.imread(pths_imagse_2[idx])
        img_for_video[:, :W, :] = img_1
        img_for_video[:, W:, :] = img_2
        cv2.imwrite("test_422/test1/duo/{:0>4d}.jpg".format(idx), img_for_video)
        video.write(img_for_video)
    video.release()