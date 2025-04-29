import cv2
import numpy as np
import matplotlib.pyplot as plt

imagePath = '/home/ug-islab/isLab/introduction/image'
savePath = '/home/ug-islab/isLab/introduction/result'

img1 = cv2.imread(f'{imagePath}/sakura-hiru.jpg')
img2 = cv2.imread(f'{imagePath}/sakura-yoru.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# ヒストグラム
def show_histogram(img, title):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        hist = cv2.calcHist([img], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.savefig(f"{savePath}/{title}.png")
    plt.show()

show_histogram(img1, "sakura-hiru-histogram")
show_histogram(img2, "sakura-yoru-histogram")

# ORB特徴量抽出
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

img1_kp = cv2.drawKeypoints(img1, kp1, None, color=(0, 255, 0), flags=0)
img2_kp = cv2.drawKeypoints(img2, kp2, None, color=(0, 255, 0), flags=0)

plt.imshow(cv2.cvtColor(img1_kp, cv2.COLOR_BGR2RGB))
plt.title("昼の桜 - 特徴点")
plt.axis("off")
plt.savefig(f"{savePath}/sakra-hiru-kp.png")
plt.show()

plt.imshow(cv2.cvtColor(img2_kp, cv2.COLOR_BGR2RGB))
plt.title("夜の桜 - 特徴点")
plt.axis("off")
plt.savefig(f"{savePath}/sakra-yoru-kp.png")
plt.show()

# 特徴点マッチング
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
plt.figure(figsize=(12, 6))
plt.imshow(cv2.cvtColor(img_match, cv2.COLOR_BGR2RGB))
plt.title("Top 20 ORB Feature Matches")
plt.axis("off")
plt.savefig(f"{savePath}/sakra-match.png")
plt.show()

# SIFT特徴量抽出

sift = cv2.SIFT_create()

keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors1, descriptors2, k=2)

good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

img_matches = cv2.drawMatches(img1, keypoints1, img2, keypoints2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imwrite(f"{savePath}/sakura-sift.png", img_matches)
cv2.imshow("Matches", img_matches)
cv2.waitKey(0)