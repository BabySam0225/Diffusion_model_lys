import cv2
import numpy as np

# 读取图像
image = cv2.imread('im_1051_.png', cv2.IMREAD_GRAYSCALE)

# 使用Canny算法进行边缘检测
edges = cv2.Canny(image, threshold1=100, threshold2=200)

# 显示边缘检测结果
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()