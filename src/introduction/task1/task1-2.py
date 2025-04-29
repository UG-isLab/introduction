import cv2

imagePath = '/home/ug-islab/isLab/introduction/image'
savePath = '/home/ug-islab/isLab/introduction/result'
if not savePath:
    print('resultフォルダがありません。')
    exit()

img = cv2.imread(f'{imagePath}/sakura-hiru.jpg')

#画像の表示
cv2.imshow('sakura', img)
cv2.waitKey(0)

#画像の縮小
resized_img = cv2.resize(img, (100, 300))
# cv2.imshow('small-sakura', resized_img)
# cv2.waitKey(0)
cv2.imwrite(f'{savePath}/small-sakura.jpg', resized_img)

#画像の拡大
resized_img = cv2.resize(img, None, fx=1.5, fy=2)
cv2.imshow('big-sakura', resized_img)
cv2.waitKey(0)
cv2.imwrite(f'{savePath}/big-sakura.jpg', resized_img)

#画像の回転
height, width = img.shape[:2]
center = (width // 2, height // 2)
matrix = cv2.getRotationMatrix2D(center, 45, 1.0)
rotated_img = cv2.warpAffine(img, matrix, (width, height))
# cv2.imshow('rotated-sakura', rotated_img)
# cv2.waitKey(0)
cv2.imwrite(f'{savePath}/rotated-sakura.jpg', rotated_img)


#画像の二値化
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, binary_img = cv2.threshold(gray_img, 128, 255, cv2.THRESH_BINARY)
# cv2.imshow('gray-sakura', binary_img)
# cv2.waitKey(0)
cv2.imwrite(f'{savePath}/gray-sakura.jpg', binary_img)