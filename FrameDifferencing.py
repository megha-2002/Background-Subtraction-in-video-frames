import cv2
import numpy as np
import glob
import math

from PIL import Image
from skimage import img_as_float
from skimage.metrics import mean_squared_error as mse
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

def average(lst):
    return sum(lst) / len(lst)

imdir = 'C:\\Users\\DELL\\OneDrive\\Desktop\\megha\\MATH S3\\project\\Intersection\\'
ext1 = ['png', 'jpg', 'gif'] # Add image formats here

#read file
files = []
[files.extend(glob.glob(imdir + '*.' + e)) for e in ext1] 
images = [cv2.imread(file) for file in files]

# Ground Truth,nimage used to compare all the images we produce.
ext2 = ['bmp'] # Ground Truth format.
files2 = []
[files2.extend(glob.glob(imdir + '*.' + e)) for e in ext2]
gtimages = [cv2.imread(file) for file in files2]


first_frame=images[0]
first_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
first_gray = cv2.GaussianBlur(first_gray, (5, 5), 0)

gtindex=0
gtval=500
mselst=[]
psnrlst=[]
ssimlst=[]

for i in range(1,len(images)):
    frame=images[i]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #convert the color profile
    gray_frame = cv2.GaussianBlur(gray_frame, (5, 5), 0)
    difference = cv2.absdiff(first_gray, gray_frame)
    _, difference = cv2.threshold(difference, 60, 255, cv2.THRESH_BINARY)
    
    #show
    cv2.imshow("First frame", first_frame)
    cv2.imshow("Frame", frame)
    cv2.imshow("difference", difference)
    
    #display a certain frame for some time
    key = cv2.waitKey(30) #pause
    if key == 27:
        break
    
    if(i==gtval):

        gt=gtimages[gtindex]

        img1=cv2.cvtColor(gt,cv2.COLOR_BGR2RGB)
        img2=cv2.cvtColor(difference,cv2.COLOR_BGR2RGB)
        gt_pil=Image.fromarray(img1)
        mask_pil=Image.fromarray(img2)

        mselst.append(mse(img_as_float(mask_pil),img_as_float(gt_pil)))
        psnrlst.append(psnr(img_as_float(mask_pil),img_as_float(gt_pil)))
        ssimlst.append(ssim(img_as_float(mask_pil),img_as_float(gt_pil),multichannel=True,gaussian_weights=True))

        gtindex+=1
        gtval+=15

avgmse=average(mselst)
avgpsnr=average(psnrlst)
avgssim=average(ssimlst)

print("Average MSE: ",avgmse)
print("Average PSNR: ",avgpsnr)
print("Average SSIM: ",avgssim)
cv2.destroyAllWindows()
