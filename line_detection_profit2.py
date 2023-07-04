import numpy as np
import cv2 as cv


import dataset
from shapesimilarity import shape_similarity
import difflib

# 进行透视变换
def warpImage(image, src_points, dst_points):
    image_size = (image.shape[1], image.shape[0])
    M = cv.getPerspectiveTransform(src_points, dst_points)  # 参数一:源图像中待测矩形的四点坐标  参数二:目标图像  返回值:由源图像中矩形到目标图像矩形变换的矩阵
    Minv = cv.getPerspectiveTransform(dst_points, src_points)
    warped_image = cv.warpPerspective(image, M, image_size, flags = cv.INTER_LINEAR)  # 变换后的图像
    return warped_image, M, Minv


# 灰度图转换
def grayscale(image):
    return cv.cvtColor(image, cv.COLOR_RGB2GRAY)

# Canny边缘检测 双阈值
def canny(image, low_threshold, high_threshold):
    return cv.Canny(image, low_threshold, high_threshold)

# 高斯滤波
def gaussian_blur(image, kernel_size):
    return cv.GaussianBlur(image, (kernel_size, kernel_size), 0)

# 生成感兴趣区域即Mask掩模
def region_of_interest(image, vertices):

    mask = np.zeros_like(image)  # 生成图像大小一致的zeros矩

    # 填充顶点vertices中间区域
    if len(image.shape) > 2:
        channel_count = image.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    # 填充函数
    cv.fillPoly(mask, vertices, ignore_mask_color)
    masked_image = cv.bitwise_and(image, mask)#与运算 在感兴趣区域mask保留image图片的信息
    # cv.imshow("mask", mask)
    return masked_image

# 霍夫变换检测直线
def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):

    # rho：线段以像素为单位的距离精度
    # theta : 像素以弧度为单位的角度精度(np.pi/180较为合适)
    # threshold : 霍夫平面累加的阈值
    # minLineLength : 线段最小长度(像素级)
    # maxLineGap : 最大允许断裂长度
    lines = cv.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    return lines

# 画霍夫变换直线
def draw_houghline(image, lines, slope_min, slope_max, color=[0,0,255], thickness=2):
    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)  # 拟合成直线 一阶
            slope = fit[0]  # 斜率

            if slope_min < np.absolute(slope) <= slope_max:
                cv.line(image, (x1, y1), (x2, y2), color, thickness)

# 画车道线
def draw_lines(image, lines, midx=720, slope_min = 0.5, slope_max = 20, color=[255,0,0], thickness=2):

    right_y_set = []
    right_x_set = []
    right_slope_set = []

    left_y_set = []
    left_x_set = []
    left_slope_set = []

    # slope_min = 0.5  # 斜率低阈值 .35
    # slope_max = 20  # 斜率高阈值 .85
    max_y = image.shape[0]  # 最大y坐标

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)    # 拟合成直线 一阶
            slope = fit[0]  # 斜率

            if slope_min < np.absolute(slope) <= slope_max:
                # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
                if slope > 0 and x1 > midx and x2 > midx:
                    right_y_set.append(y1)
                    right_y_set.append(y2)
                    right_x_set.append(x1)
                    right_x_set.append(x2)
                    right_slope_set.append(slope)

                # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
                elif slope < 0 and x1 < midx and x2 < midx:
                    left_y_set.append(y1)
                    left_y_set.append(y2)
                    left_x_set.append(x1)
                    left_x_set.append(x2)
                    left_slope_set.append(slope)

    # 绘制左车道线
    if left_y_set:
        lindex = left_y_set.index(min(left_y_set))  # 从列表中找出最高点的第一个匹配项的索引位置
        left_x_top = left_x_set[lindex]
        left_y_top = left_y_set[lindex]
        lslope = np.median(left_slope_set)   # 计算斜率的平均值

    # 根据斜率计算车道线与图片下方交点作为起点
    left_x_bottom = int(left_x_top + (max_y - left_y_top) / lslope)

    # 绘制线段
    cv.line(image, (left_x_bottom, max_y), (left_x_top, left_y_top), color, thickness)

    # 绘制右车道线
    if right_y_set:
        rindex = right_y_set.index(min(right_y_set))  # 最高点
        right_x_top = right_x_set[rindex]
        right_y_top = right_y_set[rindex]
        rslope = np.median(right_slope_set)

    # 根据斜率计算车道线与图片下方交点作为起点
    right_x_bottom = int(right_x_top + (max_y - right_y_top) / rslope)

    # 绘制线段
    cv.line(image, (right_x_top, right_y_top), (right_x_bottom, max_y), color, thickness)
    # 左右车道线信息
    Curve = [[left_x_bottom, max_y, left_x_top, left_y_top]]
    Curve.append([right_x_top, right_y_top, right_x_bottom, max_y])
    # C1 = [(left_x_bottom, max_y), (left_x_top, left_y_top)]
    # C2 = [(right_x_top, right_y_top), (right_x_bottom, max_y)]
    return Curve

# hough变换后的车道线拟合（曲线）

# def fit_poly(img, lines, h_samples, lanes, midx = 640 , slope_min = 0, slope_max = 12):
#     # slope_min = 0  # 斜率低阈值 .35
#     # slope_max = 30  # 斜率高阈值 .85
#     # middle_x =  # 图像中线x坐标
#     # max_y = image.shape[0]  # 最大y坐标
#
#     leftx = []
#     lefty = []
#     rightx = []
#     righty = []
#
#     for line in lines:
#         for x1, y1, x2, y2 in line:
#             fit = np.polyfit((x1, x2), (y1, y2), 1)    # 拟合成直线 一阶
#             slope = fit[0]  # 斜率
#
#             if slope_min < np.absolute(slope) <= slope_max:
#
#                 # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
#                 if slope > 0 and x1 > midx and x2 > midx:
#                     righty.append(y1)
#                     righty.append(y2)
#                     rightx.append(x1)
#                     rightx.append(x2)
#                 # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
#                 elif slope < 0 and x1 < midx and x2 < midx:
#                     lefty.append(y1)
#                     lefty.append(y2)
#                     leftx.append(x1)
#                     leftx.append(x2)
#
#     # 用np.polyfit()拟合一个二阶多项式
#     # left_fit = np.polyfit(lefty, leftx, 2)
#     # right_fit = np.polyfit(righty, rightx, 2)
#     left_fit = np.polyfit(lefty, leftx, 3)
#     right_fit = np.polyfit(righty, rightx, 3)
#
#     # 为绘图生成x和y值
#     ploty = np.linspace(int(4 * img.shape[0] / 10), img.shape[0] - 1, int(img.shape[0] / 10))
#     ploty = (ploty).astype(np.int)
#     # 使用ploty, left_fit和right_fit计算两个多项式
#     # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
#     # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
#     left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
#     right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
#
#     draw(img, (left_fitx).astype(np.int), ploty, color=[0, 0, 200], thickness=10)
#     draw(img, (right_fitx).astype(np.int), ploty, color=[0, 0, 200], thickness=10)


# hough变换后的车道线拟合（曲线）
def fit_poly(img, lines, h_samples, lanes, midx=640, slope_min=0, slope_max=12):
    # 斜率低阈值 .35
    # 斜率高阈值 .85
    # middle_x =  # 图像中线x坐标
    # max_y = image.shape[0]  # 最大y坐标

    leftx = []
    lefty = []
    rightx = []
    righty = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)    # 拟合成直线 一阶
            slope = fit[0]  # 斜率

            if slope_min < np.absolute(slope) <= slope_max:

                # 将斜率大于0且线段X坐标在图像中线右边的点存为右边车道线
                if slope > 0 and x1 > midx and x2 > midx:
                    righty.append(y1)
                    righty.append(y2)
                    rightx.append(x1)
                    rightx.append(x2)
                # 将斜率小于0且线段X坐标在图像中线左边的点存为左边车道线
                elif slope < 0 and x1 < midx and x2 < midx:
                    lefty.append(y1)
                    lefty.append(y2)
                    leftx.append(x1)
                    leftx.append(x2)

    # 用np.polyfit()拟合一个二阶多项式
    # left_fit = np.polyfit(lefty, leftx, 1)
    # right_fit = np.polyfit(righty, rightx, 1)
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    # left_fit = np.polyfit(lefty, leftx, 3)  # 拟合函数
    # right_fit = np.polyfit(righty, rightx, 3)

    # 为绘图生成x和y值
    # ploty = np.linspace(int(4 * img.shape[0] / 10), img.shape[0] - 1, int(img.shape[0]))
    # ploty = (ploty).astype(np.int)
    # 使用ploty, left_fit和right_fit计算两个多项式
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    # left_fitx = left_fit[0] * ploty ** 3 + left_fit[1] * ploty ** 2 + left_fit[2] * ploty + left_fit[3]
    # right_fitx = right_fit[0] * ploty ** 3 + right_fit[1] * ploty ** 2 + right_fit[2] * ploty + right_fit[3]
    # draw(img, (left_fitx).astype(np.int), ploty, color=[0, 0, 255], thickness=10)
    # draw(img, (right_fitx).astype(np.int), ploty, color=[0, 0, 255], thickness=10)

    X = []
    Y = []
    for lane in lanes:
        for i in range(len(lane)):
            if lane[i] > 0:
                X.append(lane[i])
                Y.append(h_samples[i])


        # X_leftfit = np.poly1d([left_fit[0], left_fit[1]])
        X_leftfit = np.poly1d([left_fit[0], left_fit[1], left_fit[2]])
        # X_leftfit = np.poly1d([left_fit[0], left_fit[1], left_fit[2], left_fit[3]])  # 三阶
        X_left = X_leftfit(Y)
        draw(img, (X_left).astype(np.int), Y, color=[255, 0, 0], thickness=10)  # 画测试的左车道线
        R2(X_left, X)  # 计算拟合度

        # X_rightfit = np.poly1d([right_fit[0], right_fit[1]])
        X_rightfit = np.poly1d([right_fit[0], right_fit[1], right_fit[2]])
        # X_rightfit = np.poly1d([right_fit[0], right_fit[1], right_fit[2], right_fit[3]])  # 三阶
        X_right = X_rightfit(Y)
        draw(img, (X_right).astype(np.int), Y, color=[255, 0, ], thickness=10)
        R2(X_right, X)

        X = []
        Y = []


    # return left_fitx, right_fitx, ploty, left_fit, right_fit

# 原图像与车道线图像按照a:b比例融合
def weighted_img(initial_img, img, a=0.8, b=1.0, c=0.):
    return cv.addWeighted(initial_img, a, img, b, c)

# 两个一维数组画线
def draw(image, X, Y, color=[0,0,255], thickness=2):
    # X = [567, 532, 496, 461, 425, 390, 355, 319, 284, 248, 213, 177, 142, 106, 71, 35]
    # Y = [270, 280, 290, 300, 310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420]
    # a = len(X)
    X = X[::-1]
    Y = Y[::-1]

    for i in range(len(X)-5):
        # print(i)
        cv.line(image, (X[i], Y[i]), (X[i+1], Y[i+1]), color, thickness)
    # cv.line(image, (X, Y), color, thickness)

# 画出测试标签里的车道线
def draw_correctline(image, lanes, h_samples):

    # for yi in h_samples:
    #     for line in lanes:
    #         for xi in line:
    #             if xi > 0:
    #                 cv.line(image, (xi, yi), color, thickness)

    X = []
    Y = []
    for lane in lanes:
        for i in range(len(lane)):
            if lane[i] > 0:
                X.append(lane[i])
                Y.append(h_samples[i])
        draw(image, X, Y)
        X = []
        Y = []

# 指标 线（两点）:标签、实验
# 指标1 曲线拟合程度
def value_similarity(X, X_real, Y):

    # C1 = []
    # C2 = []
    #
    # for i in range(len(Y)):
    #     C1.append((Y[i], X[i]))
    #     C2.append((Y[i], int(X_real[i])))
    # shape1 = np.column_stack(C1)  # 按列合并为一个矩阵
    # shape2 = np.column_stack(C2)
    shape1 = np.column_stack((Y, X))  # 按列合并为一个矩阵
    shape2 = np.column_stack((Y, X_real))
    # # 调用库计算相似度
    # similarity = shape_similarity(shape1, shape2)
    # # 相似度输出
    # print("similarity:".format(similarity))


# 2.计算两曲线之间的欧式距离/拟合程度
def eucliDist(Curve, lanes, h_samples):
    X = []
    Y = []
    for lane in lanes:
        # 得到有效的车道线标签信息（X,Y）
        for i in range(len(lane)):
            if lane[i] > 0:
                X.append(lane[i])
                Y.append(h_samples[i])
        # 处理检测出的车道线信息
        for x1, y1, x2, y2 in Curve:
            # fit = np.polyfit((x1, x2), (y1, y2), 1)    # 拟合成直线 一阶
            # slope = fit[0]  # 斜率

            # x = sympy.symbols("x")  # 申明未知数"x"
            # # X_real = sympy.solve([np.plot1d(np.polyfit((x1, y1), (x2, y2), 1))], [Y])
            # # fit = np.polyfit((x1, y1), (x2, y2), 1)
            # fit = np.poly1d(fit, True, varibale='x')
            # # print(fit)
            # # print(np.poly1d(fit))
            # X_real = sympy.solve([np.poly1d(fit)], [Y])

            fit = np.polyfit((y1, x1), (y2, x2), 1)
            X_test = np.poly1d([fit[0], fit[1]])
            X_test = X_test(Y)  # 将x值带入方程求y



            # # Frechet距离
            # value_similarity(X, X_test, Y)
            # # 拟合优度 针对回归曲线
            # rr = goodness_of_fit(X_real, X)
            # # 文本相似度 所有数据都有偏差
            # sm = difflib.SequenceMatcher(None, X, X_real)
            # print('文本相似度:', sm.ratio())

            ## 利用范数实现拟合程度的回归评价指标 X_real测试值 X真实值
            ## MSE--相当于y-y_hat的二阶范数的平方/n
            MSE = np.linalg.norm(X - X_test, ord=2) ** 2 / len(X)
            print("MSE:", MSE)
            ## RMSE--相当于y-y_hat的二阶范数/根号n
            RMSE = np.linalg.norm(X - X_test, ord=2) / len(X) ** 0.5
            print("RMSE:", RMSE)
            ## MAE--相当于y-y_hat的一阶范数/n
            MAE = np.linalg.norm(X - X_test, ord=1) / len(X)
            print("MAE:", MAE)
            R2 = 1 - MSE / np.var(X)
            print("R2:", R2)

        # 初始化
        X = []
        Y = []

def R2(X_test, X):
    # X为真实值 X_test测试值（真实条件下的）
    # 计算俩车道线与左右车道标签的距离
    # print("欧式距离:", np.sqrt(sum(np.power((X - X_test), 2))))
    # 处理检测出的车道线信息
    ## 利用范数实现拟合程度的回归评价指标 X_real测试值 X真实值
    ## MSE--相当于y-y_hat的二阶范数的平方/n
    MSE = np.linalg.norm(X - X_test, ord=2) ** 2 / len(X)
    print("MSE:", MSE)
    ## RMSE--相当于y-y_hat的二阶范数/根号n
    RMSE = np.linalg.norm(X - X_test, ord=2) / len(X) ** 0.5
    print("RMSE:", RMSE)
    ## MAE--相当于y-y_hat的一阶范数/n
    MAE = np.linalg.norm(X - X_test, ord=1) / len(X)
    print("MAE:", MAE)
    R2 = 1 - MSE / np.var(X)
    print("R2:", R2)


# 3.拟合优度
def __sst(y_no_fitting):
    """
    计算SST(total sum of squares) 总平方和
    :param y_no_predicted: List[int] or array[int] 待拟合的y
    :return: 总平方和SST
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_no_fitting]
    sst = sum(s_list)
    return sst

# def __sse(y_fitting, y_no_fitting):
#     """
#     计算SST(total sum of squares) 总平方和
#     :param y_no_predicted: List[int] or array[int] 待拟合的y
#     :return: 总平方和SST
#     """
#     y_mean = sum(y_no_fitting) / len(y_no_fitting)
#     s_list =[(y - y_mean)**2 for y in y_fitting]
#     sse = sum(s_list)
#     return sse

def __ssr(y_fitting, y_no_fitting):
    """
    计算SSR(regression sum of squares) 回归平方和
    :param y_fitting: List[int] or array[int]  拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 回归平方和SSR
    """
    y_mean = sum(y_no_fitting) / len(y_no_fitting)
    s_list =[(y - y_mean)**2 for y in y_fitting]
    ssr = sum(s_list)
    return ssr

def goodness_of_fit(y_fitting, y_no_fitting):
    """
    计算拟合优度R^2
    :param y_fitting: List[int] or array[int] 拟合好的y值
    :param y_no_fitting: List[int] or array[int] 待拟合y值
    :return: 拟合优度R^2
    """
    SSR = __ssr(y_fitting, y_no_fitting)
    SST = __sst(y_no_fitting)
    rr = SSR / SST
    print("拟合优度为:", rr)
    return rr

# 亮度划分函数   l通道
def hlsLSelect(img, thresh=(200,255)):# 220,255
    hls = cv.cvtColor(img, cv.COLOR_BGR2HLS) # 720*1280*3
    # hue:色相 lightness:亮度 saturation:饱和度
    l_channel = hls[:, :, 1]  # 亮度矩阵 720*1280
    l_channel = l_channel*(255/np.max(l_channel))
    # 创建一个空矩阵，黑图片
    binary_output = np.zeros_like(l_channel)
    # 梯度在阈值范围内，图片点亮
    binary_output[(l_channel > thresh[0]) & (l_channel <= thresh[1])] = 255
    # cv.imshow("binary_hls", binary_output)
    return binary_output

# lab蓝黄通道划分函数
def labBSelect(img,thresh=(215,255)):  # 215,255
    lab = cv.cvtColor(img, cv.COLOR_BGR2Lab)
    lab_b = lab[:, :, 2]
    # 如果图像中没有阈值就不进行归一化
    if np.max(lab_b) > 100:
        lab_b = lab_b*(255/np.max(lab_b))
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0]) & (lab_b <= thresh[1]))] = 255
    # cv.imshow("binary_lab", binary_output)
    return binary_output


def process_image(image):
    # cv.imshow("image", image)

    # #1.1透视变换结果，得到俯视图
    # src = np.float32([[500, 200], [100, 800], [850, 200], [1215, 800]])
    # dst = np.float32([[300, 0], [470, 800], [500, 0], [700, 800]])
    # cv2.line(test_distort_image, (565, 450), (390, 620), (255, 0, 0), 8)
    # cv2.line(test_distort_image, (650, 450), (1015, 620), (255, 0, 0), 8)

    # perspective_image, M, Minv = warpImage(image, src, dst)
    # cv.imshow('perspective', perspective_image)
    #
    # imperspective_image, M, Minv = warpImage(perspective_image, dst, src)
    # cv.imshow('im_perspective', imperspective_image)

    # 2.1预处理
    # hlsL_binary = hlsLSelect(image)
    # cv.imshow("hlsL_binary", hlsL_binary)
    # labB_binary = labBSelect(image)
    # cv.imshow("labB_binary", labB_binary)
    # combined_binary = np.zeros_like(image)
    # combined_binary[(hlsL_binary == 255) | (labB_binary == 255)] = 255
    # cv.imshow('img_binary', combined_binary)

    # 1.灰度图转换
    gray = grayscale(image)
    # cv.imshow('gray', gray)

    # # # 设置高斯噪声
    # # noise = np.random.normal(10, 10, gray.shape)
    # # noisy = gray + noise
    # # noisy = noisy/255
    # # cv.imshow('image1', noisy)
    #
    # # 2.高斯滤波
    # # 设置参数
    # kernel_size = 5  # 高斯滤波器大小size
    # blur_gray = gaussian_blur(gray, kernel_size)
    # # cv.imshow('blur_gray', blur_gray)

    # 3.Canny边缘检测 双阈值技术
    canny_low_threshold = 75  # canny边缘检测低阈值
    canny_high_threshold = canny_low_threshold * 3  # canny边缘检测高阈值
    edge_image = canny(gray, canny_low_threshold, canny_high_threshold)
    cv.imshow('edge_image', edge_image)
    # ret, img = cv.threshold(gray, 190, 255, cv.THRESH_BINARY)  # 单阈值二值化
    # cv.imshow("img", img)

    # # 4.生成Mask掩模
    # 掩膜范围设置
    # vertices = np.array([[(0, image.shape[0]), (0, 5 * image.shape[0] / 12), (image.shape[1], 5 * image.shape[0] / 12), (image.shape[1], image.shape[0])]], dtype=np.int32)
    vertices = np.array([[(1 * image.shape[1] / 20, image.shape[0]), (8 * image.shape[1] / 20, 7 * image.shape[0] / 18),
                (12 * image.shape[1] / 20, 7 * image.shape[0] / 18), (image.shape[1], 15 * image.shape[0] / 17), (image.shape[1], image.shape[0])]], dtype=np.int32)
    # vertices = np.array([[(1 * image.shape[1] / 20, image.shape[0]), (1 * image.shape[1] / 20, 7 * image.shape[0] / 18),
    #             (19 * image.shape[1] / 20, 7 * image.shape[0] / 18),  (19 * image.shape[1] / 20, image.shape[0])]], dtype=np.int32)
    masked_edges = region_of_interest(edge_image, vertices)
    cv.imshow("masked_edges", masked_edges)

    # 5.基于霍夫变换的直线检测
    rho = 1  # 霍夫像素单位
    theta = np.pi / 180  # 霍夫角度移动步长
    hof_threshold = 25  # 霍夫平面累加阈值threshold
    min_line_len = 5  # 线段最小长度
    max_line_gap = 50  # 最大允许断裂长度

    lines = hough_lines(masked_edges, rho, theta, hof_threshold, min_line_len, max_line_gap)
    line_image = np.zeros_like(image)  # 保持车道线信息

    # # 测试
    # 参数设置
    midx = int(7 * image.shape[1] / 13)  # 画中线(参考线)的标准参数
    slope_min = 0.5  # houghlines 低斜率阈值
    slope_max = 20  # houghlines 高斜率阈值

    # draw_houghline(image, lines, slope_min, slope_max)  # 二值图中的ROI上，画满足条件的HOUGH变换的直线
    # cv.line(image, (midx, int(image.shape[0])), (midx, 0), color=[0, 255, 0], thickness=1)  # 画分界线

    draw_correctline(image, dataset.lanes, dataset.h_samples)  # 标签车道线
    # cv.imshow('test_image', image)
    # fit_poly(perspective_image, lines, midx)  # 在透视变换的图中画车道线 用于观察拟合效果
    # cv.imshow("perspective_image", perspective_image)

    # 6.绘制车道线线段
    # Curve = draw_lines(line_image, lines, midx, slope_min, slope_max, thickness=10)
    fit_poly(line_image, lines, dataset.h_samples, dataset.lanes, midx, slope_min, slope_max)  # 只画车道线 用于透视逆变换 图像融合
    # cv.imshow('line_image', line_image)

    # 7.图像融合
    alpha = 0.8   # 原图像权重
    beta = 1.     # 车道线图像权重
    gamma = 0.

    # test, M, Minv = warpImage(line_image, dst, src)
    # lines_edges = weighted_img(image, test, alpha, beta, gamma)  # dst = alpha * image + beta * line_image + gamma
    lines_edges = weighted_img(image, line_image, alpha, beta, gamma)  # dst = alpha * image + beta * line_image + gamma

    cv.imshow('lines_edges', lines_edges)  # 融合后结果图
    # cv.imshow("image_result", image)
    # 8.评价指标:拟合程度 R方（归一化）
    # eucliDist(Curve, dataset.lanes, dataset.h_samples)


if __name__ == '__main__':
    # 图片的车道线检测
    # image = cv.imread('1.jpg')
    image = cv.imread('D:/Pycharm/coding/Dataset/5left.jpg')

    # 主程序
    process_image(image)
    cv.waitKey(0)

    # #视频流的车道线检测
    # cap = cv.VideoCapture("video_1.mp4")
    # while (cap.isOpened()):
    #     ret, frame = cap.read()     # ret读取是否成功的布尔值 frame读取到的图像帧
    #     processed = process_image(frame)
    #     # cv.imshow("image", processed)
    #     cv.waitKey(1)

