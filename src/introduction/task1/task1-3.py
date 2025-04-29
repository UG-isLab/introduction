import cv2

imagePath = '/home/ug-islab/isLab/introduction/image'
savePath = '/home/ug-islab/isLab/introduction/result'

img1 = cv2.imread(f'{imagePath}/sakura-hiru.jpg')
img2 = cv2.imread(f'{imagePath}/sakura-yoru.jpg')

gray_day = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray_night = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

diff_img = cv2.absdiff(gray_day, gray_night)

# _, binary_diff = cv2.threshold(diff_img, 30, 255, cv2.THRESH_BINARY)
cv2.imshow("Difference", diff_img)
cv2.waitKey(0)

cv2.imwrite(f'{savePath}/diff-sakura.jpg', diff_img)