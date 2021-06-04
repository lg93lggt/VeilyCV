


import numpy as np
from scipy import ndimage, signal
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2

# open the input images
I = cv2.imread('/home/veily3/LIGAN/VeilyCV/test_409/1.jpg', flags=cv2.IMREAD_GRAYSCALE).astype(float)
rois = np.loadtxt("/home/veily3/LIGAN/VeilyCV/test_409/rois.txt")
print(rois)

Is_roi = []
for roi in rois:
    [x, y, w, h] = roi
    Is_roi.append(I[int(y):int(y + h), int(x):int(x + w)])

def corner(I, w, t, w_nms):
    # inputs: 
    # I is the input image (may be mxn for BW or mxnx3 for RGB)
    # w is the size of the window used to compute the gradient matrix N
    # t is the minor eigenvalue threshold
    # w_nms is the size of the window used for nonmaximal supression
    # outputs:
    # J0 is the mxn image of minor eigenvalues of N before thresholding
    # J1 is the mxn image of minor eigenvalues of N after thresholding
    # J2 is the mxn image of minor eigenvalues of N after nonmaximal supression
    # pts0 is the 2xk list of coordinates of (pixel accurate) corners
    #     (ie. coordinates of nonzero values of J2)
    # pts is the 2xk list of coordinates of subpixel accurate corners
    #     found using the Forstner detector
    
    """your code here"""
    m,n = I.shape[:2]
    k = [-1.0,8.0,0.0,-8.0,1.0]
    fx = np.zeros((1,5))
    fx[:] = k
    fx = 1.0/12.0*fx
    fy = fx.T
    
    ## Gradient computation
    Ix = ndimage.filters.convolve(I,fx)
    Iy = ndimage.filters.convolve(I,fy)
    Ixx = Ix**2
    Ixy = Ix*Iy
    Iyy = Iy**2
    
    ## Sum computation
    f = np.ones((w,w))
    Sxx = ndimage.filters.convolve(Ixx,f)
    Sxy = ndimage.filters.convolve(Ixy,f)
    Syy = ndimage.filters.convolve(Iyy,f)
    
    ## Forstner computation
    X, Y = np.meshgrid(range(n),range(m))
    Ixxf = Ixx*X
    Ixyf = Ixy*X
    Iyxf = Ixy*Y
    Iyyf = Iyy*Y
    Sxxf = ndimage.filters.convolve(Ixxf,f)
    Sxyf = ndimage.filters.convolve(Ixyf,f)
    Syxf = ndimage.filters.convolve(Iyxf,f)
    Syyf = ndimage.filters.convolve(Iyyf,f)

    ## Minimum eigen value extraction
    J1 = np.zeros((m,n))
    J0 = np.zeros((m,n))
    subp = np.zeros((m,n,2))
    s = (w-1)//2
    
    for r in range(s, m-s):
        for c in range(s,n-s):
            N = np.zeros((2,2))
            b = np.zeros((2,1))
            N = np.asarray([[Sxx[r,c], Sxy[r,c]],[Sxy[r,c], Syy[r,c]]])
            det = np.linalg.det(N)
            tr = np.trace(N)
            J0[r,c] = (tr - (tr**2-4*det)**0.5)/2
            
            ## Forstner subpixel computation
            b = np.asarray([Sxxf[r,c] + Syxf[r,c],Sxyf[r,c] + Syyf[r,c]])
            subp[r,c] = np.linalg.lstsq(N,b)[0].reshape(2,)
            
            ## Thresholding of values
            J1[r,c] = J0[r,c] if J0[r,c]>t*w**2 else 0.0
                    
    ## Non maximal suppression
    J2 = np.zeros((m,n))
    s_nms = (w_nms-1)//2
    for r in range(s_nms,m-s_nms):
        for c in range(s_nms,n-s_nms):
            J2[r,c] = 0.0 if J1[r,c]<np.max(J1[r-s_nms:r+s_nms,c-s_nms:c+s_nms]) else J1[r,c]
        
    y, x = np.nonzero(J2)
    pts0 = np.vstack((x,y))
    pts = np.zeros(pts0.shape)
    for i in range(pts.shape[1]):
        pts[0,i] = subp[pts0[1,i],pts0[0,i]][0]
        pts[1,i] = subp[pts0[1,i],pts0[0,i]][1] 
    
    return J0, J1, J2, pts0, pts

# parameters to tune
w=3
t=0.0002
w_nms=7

n_rois = len(Is_roi)
# extract corners
for [idx, I_roi] in enumerate(Is_roi):
    J_0, J_1, J_2, pts_0, pts_1 = corner(I_roi, w, t, w_nms)
    plt.subplot(n_rois, 2, idx*2+1)
    plt.imshow(I_roi, cmap="gray")
    ax = plt.subplot(n_rois, 2, idx*2+2)
    plt.imshow(I_roi)
    for i in range(pts_0.shape[1]):
        x,y = pts_0[:,i]
        ax.add_patch(patches.Rectangle((x-w/2,y-w/2),w,w, fill=False))
    plt.plot(pts_0[0,:], pts_0[1,:], '.r') # display pixel accurate corners
    plt.plot(pts_1[0,:], pts_1[1,:], '.g') # display subpixel corners

plt.show()
