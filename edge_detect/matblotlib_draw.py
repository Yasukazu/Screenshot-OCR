import cv2
import matplotlib.pyplot as plt

# 画像を読み込む
image = cv2.imread('data/sample.png')

# BGRからRGBに変換して表示
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.show()
