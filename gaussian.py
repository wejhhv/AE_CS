import cv2
#ファイルパス
imgpath="cat.jpg"

#! ガウシアン 
img = cv2.imread(imgpath)
#引数は２つ目がカーネル、３つ目が標準偏差
dst = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imwrite('Gaussian'+imgpath, dst)

