

import  cv2
import numpy as np
from pathlib import Path
import  glob
import  os

if __name__ == "__main__":
    name_output = "7df3343d5906ac944e8b75c45ccbad0b.mp4"
    dir_images = Path("/home/veily3/LIGAN/VeilyCV/test/test_517/")
    pths_imagse = sorted(Path(dir_images, "*.png").glob())

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    img_1 = cv2.imread(pths_imagse[0])
    [H, W] = img_1.shape[:2]

    video = cv2.VideoWriter(Path(dir_images, name_output), fourcc, 1, (W, H),True)

    for idx in range(len(pths_imagse)):
        print("{:0>6d}".format(idx))
        img_for_video = np.zeros((H, W, 3)).astype(np.uint8)
        img_1 = cv2.imread(pths_imagse[idx])
        img_for_video[:, :W, :] = img_1
        video.write(img_for_video)
    video.release()
    #     if ret==True:

    #         cv2.imshow('frame',frame)
    #         out.write(frame)

    #         if cv2.waitKey(10) & 0xFF == ord('q'):
    #             break
    #     else:
    #         break

    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()