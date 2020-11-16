# import cv2
# import numpy as np
# import sys
# import copy
#
# def show(name, src):
#     """
#         展示图片 show
#         :param name:图片
#         :param src:宽度
#         :return:展示的图片
#     """
#     cv2.imshow(name, src)
#     cv2.waitKey(0)
#
#
# def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
#     """
#     使用插值方法对图片 resize
#     :param image:图片
#     :param width:宽度
#     :param height:高度
#     :param inter:插值方法
#     :return:调整大小后的图片
#     """
#     dim = None
#     (h, w) = image.shape[:2]  #
#     if width is None and height is None:
#         return image
#     if width is None:
#         r = height / float(h)
#         dim = (int(w * r), height)
#     else:
#         r = width / float(w)
#         dim = (width, int(h * r))
#     # print(dim)
#     resized = cv2.resize(image, dim, interpolation=inter)  # interpo;ation为插值方法，这里选用的是
#     return resized
#
#
# def adaptive_equalization(src):
#     """
#         对原图机型直方图自适应均衡化 adaptive_equalization
#         :param src:图片
#         :return:自适应均衡化后的图片
#     """
#     # 创建CLAHE对象
#     # 拆分每个通道
#     img_b, img_g, img_r = cv2.split(src)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     clahe_B = clahe.apply(img_b)
#     clahe_G = clahe.apply(img_g)
#     clahe_R = clahe.apply(img_r)
#     clahe_test = cv2.merge((clahe_B, clahe_G, clahe_R))
#     show('clahe_test', clahe_test)
#     return clahe_test
#
#
# def image_processing(image):
#     img1 = image.copy()
#     img2 = image.copy()
#     # 二值化
#     thresh = cv2.threshold(image.copy(), 90, 230, cv2.THRESH_BINARY)[1]
#     show("threshold", thresh)
#     # 膨胀
#     kernel = np.ones((5, 5), np.uint8)
#     erosion = cv2.erode(thresh, kernel, iterations=2)
#     show("erosion", erosion)
#     canny = cv2.Canny(erosion, 70, 230)
#     contours = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
#     cv2.drawContours(img1, contours, -1, (0, 0, 255), 1)
#     show("contours", img1)
#     for cnt in contours:
#         # 外接矩形框，没有方向角
#         x, y, w, h = cv2.boundingRect(cnt)
#         cv2.rectangle(img2, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     show("words", img2)
#
#     return image
#
#
# if __name__ == "__main__":
#     src = cv2.imread("E:/HandwritingCode/ocrcn_tf2-master/assets/Pending_pictures/2.jpg", -1)
#     src = resize(src, 400, 0)
#     show("test", src)
#     adaptive_img = adaptive_equalization(src)
#     img = image_processing(adaptive_img)
