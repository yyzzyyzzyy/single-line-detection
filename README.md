# single-line-detection
单车道检测，使用二阶曲线拟合和透视变换拟合车道线，并加入量化指标进行评价

数据集是在图森数据集下挑选的图片和标签
line_detection_profit2.py是利用二阶曲线拟合对5left.jpg进行车道检测
line_detection_curve.py是利用透视图像对7right.jpg进行车道检测
