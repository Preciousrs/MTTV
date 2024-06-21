# import pandas as pd
# import matplotlib.pyplot as plt

# # 读取 Excel 文件
# df = pd.read_excel('flex.xlsx')

# # 提取需要的数据
# g_values =[ 1,2,3,4,5,6,7]
# x_index = [1, 3, 5, 9, 16, 25, 36]
# r_values = [1, 3, 5, 10, 15, 20]

# # 创建一个空的 DataFrame 来存储绘图数据
# plot_data = pd.DataFrame(index=g_values)

# # 遍历 r 值, 将对应的 acc 值添加到 plot_data 中
# for r in r_values:
#     acc_values = df.loc[df['r'] == r, 'acc'].tolist()
    
#     # 确保每个 r 的 acc_values 长度与 g_values 一致
#     if len(acc_values) != len(g_values):
#         acc_values.extend([None] * (len(g_values) - len(acc_values)))
    
#     # 保留三位小数
#     acc_values = [round(acc, 3) if acc is not None else None for acc in acc_values]
    
#     plot_data[f'r={r}'] = acc_values

# # 绘制折线图
# plt.figure(figsize=(8, 6))
# for r in r_values:
#     plt.plot(g_values, plot_data[f'r={r}'], linewidth=2, label=f'r={r}')

# # 设置轴标签和标题
# plt.xlabel('parameter g', fontsize=16)
# plt.ylabel('accuracy', fontsize=16)


# # 设置轴范围和刻度
# plt.ylim(0.78, 0.90)
# # plt.xticks(g_values)
# x = range(1,len(g_values)+1)
# plt.xticks(x,x_index )
# plt.yticks([0.790,0.800,0.810,0.820,0.830, 0.840, 0.850 ,0.860, 0.870, 0.880 , 0.890, 0.900])

# # 添加图例
# plt.legend(loc='upper center', ncol=3)

# # 显示网格
# plt.grid(True)

# # 保存图像
# plt.savefig('acc_vs_g.png', dpi=300)
# plt.show()


#绘制N的图
import matplotlib.pyplot as plt

# 数据
parameters = [1, 3, 6, 9, 12]
accuracy = [0.90641, 0.90182, 0.91648, 0.91418, 0.90835]
precision = [0.90258, 0.90124, 0.9057, 0.9151, 0.9099]
recall = [0.8813, 0.89542, 0.8906, 0.90541, 0.8912]
f1_score = [0.9042, 0.90021, 0.9161, 0.9142, 0.9084]

# 绘制图表
plt.plot(parameters, accuracy, label='accuracy', color='blue', linestyle='-')
plt.plot(parameters, precision, label='precision', color='orange', linestyle='-.')
plt.plot(parameters, recall, label='recall', color='green', linestyle=':')
plt.plot(parameters, f1_score, label='F1 score', color='red', linestyle='--')


# 添加图例
plt.legend(loc='lower right', ncol=2)

# 添加标签和标题
plt.xlabel('parameter N',fontsize=16)
plt.ylabel('metrics',fontsize=16)


# 设置y轴范围
plt.xticks([1,3,6,9,12])
plt.ylim(0.87, 0.92)



# 显示网格
plt.grid(True)

# 保存图像
plt.savefig('acc_vs_N.png', dpi=300)
plt.show()

# #删除无法打开的文件
# import os
# from PIL import Image
# from PIL import UnidentifiedImageError
# import PIL
# Image.MAX_IMAGE_PIXELS = 1000000000
# image_folder = '/data/rensisi/HMCAN/Fakeddit/images/'  # 图像文件夹路径
# file_list = os.listdir(image_folder)
# file_count = len(file_list)
# print(file_count)
# # # 获取文件夹中的所有文件
# file_list = os.listdir(image_folder)
# i =0
# for file_name in file_list:
#     file_path = os.path.join(image_folder, file_name)

#     try:
#         with Image.open(file_path) as image:
#             # 在这里添加对图像的处理操作
#             # image.show()  # 显示图像
#             print('yes')
#     except :
#         print("无法打开图像文件：", file_path)
#         os.remove(file_path)
#         i+=1
#         print("已删除图像文件：", file_path)
# print("无法打开{}个".format(i))