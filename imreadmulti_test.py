import cv2 as cv
from PIL import Image

# Tiff画像を読み込む
image = Image.open('C:\\Users\\ochi\\temp\\sample2.tif')
#ページ数を求める
fLength =image.n_frames
#１ページずつ抜き出して処理
for i in range(0,fLength+1):
    image.seek(1)
    sImg =image.copy()    
    # 以下で何らかの処理
#path for the data
path ='inputs/flybrain.tif'
#read the colored-stack_image
Multi_img = []
ret, Multi_img = cv.imreadmulti(mats=Multi_img, filename=path, flags=cv.IMREAD_UNCHANGED)

#convert from tuple to list
Multi_img = list(Multi_img)

cv.imshow('flyBrain (Viewer)', Multi_img[29])
cv.waitKey(0)
cv.destroyAllWindows()

#loop for grayscale process
for i in range(len(Multi_img)):
    Multi_img[i] = cv.cvtColor(Multi_img[i], cv.COLOR_BGR2GRAY)
    #write the single image files separately
    cv.imwrite('outputs/separated_img/flybrain_No.'+str(i)+'.tif', Multi_img[i])

print('all separated images were saved')
print('separated frame counts: ', len(Multi_img))