
import cv2
import numpy as np

try:
    from .. import nlmeans
    from .. import noise
    cwd = "."
except:
    import sys
    cwd = "./image_denoise"
    sys.path.append(cwd)
    import nlmeans
    import noise




if __name__ == "__main__":
    img = cv2.imread(cwd + "/test/lena.png")
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img_with_noise = noise.add_gaussian_noise_multi_channels(img, 0, 20)
    img_with_noise3 = noise.add_gaussian_noise(img, 0, 40)
    cv2.imshow("a", img_with_noise)
    # cv2.waitKey()
    cv2.imshow("b", img_with_noise3)
    cv2.imwrite(cwd + "/test/lena_with_noise.png", img_with_noise3, )
    
    img3 = np.zeros(img_with_noise3.shape, dtype=np.uint8)
    img3 = cv2.fastNlMeansDenoisingColored(img_with_noise3, 10, 7, 21, 21)
    cv2.imshow("c", img3)
    img = np.zeros(img_with_noise.shape, dtype=np.uint8)
    img = cv2.fastNlMeansDenoising(img_with_noise, 10, 10, 7, 21)
    cv2.imshow("d", img)
    cv2.waitKey()
    print()