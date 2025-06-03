# ######1、找出格式不正确的npz，并转移
# import os
# import numpy as np
# import shutil

# # 原始目录和失败文件目标目录
# directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/npz'
# failed_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/failnpz'

# # 确保失败文件目录存在，如果不存在则创建
# if not os.path.exists(failed_directory):
#     os.makedirs(failed_directory)

# # 初始化计数器
# error_count = 0
# success_count = 0
# total_files = 0

# # 列出目录下所有文件
# for filename in os.listdir(directory):
#     if filename.endswith('.npz'):  # 只处理 .npz 文件
#         total_files += 1
#         file_path = os.path.join(directory, filename)  # 拼接完整路径
        
#         try:
#             label = np.load(file_path, allow_pickle=True)
#             success_count += 1
#         except Exception as e:
#             print(f"Error loading {filename}: {e}")
#             error_count += 1
#             # 移动加载失败的文件到目标目录
#             shutil.move(file_path, os.path.join(failed_directory, filename))
#             print(f"Moved {filename} to {failed_directory}")

# # 输出加载失败和成功的文件总数
# print(f"Total files: {total_files}")
# print(f"Total successfully loaded files: {success_count}")
# print(f"Total failed files: {error_count}")



# ############2、对应1中npz，找出fasta不用
# import os
# import shutil

# # 目录路径
# npz_failed_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/failnpz'
# fasta_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/fasta'
# fasta_failed_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/fail_fasta'

# # 确保目标目录存在
# if not os.path.exists(fasta_failed_directory):
#     os.makedirs(fasta_failed_directory)

# # 获取加载失败的 .npz 文件名（去掉扩展名）
# failed_npz_files = [f.split('.')[0] for f in os.listdir(npz_failed_directory) if f.endswith('.npz')]

# # 初始化计数器
# moved_count = 0

# # 列出fasta目录下的所有文件并检查是否有对应的失败文件
# for filename in os.listdir(fasta_directory):
#     if filename.endswith('.fasta'):  # 只处理 .fasta 文件
#         fasta_name = filename.split('.')[0]
        
#         # 如果.fasta文件对应的文件在加载失败的.npz文件列表中，则移动
#         if fasta_name in failed_npz_files:
#             source_path = os.path.join(fasta_directory, filename)
#             dest_path = os.path.join(fasta_failed_directory, filename)
#             shutil.move(source_path, dest_path)
#             print(f"Moved {filename} to {fasta_failed_directory}")
#             moved_count += 1

# # 输出移动文件的总数
# print(f"Total files moved: {moved_count}")



# #####3、检查fasta和npz目录下的文件对应关系，多余的移出
# import os
# import shutil

# # 定义目录路径
# fasta_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/fasta'
# npz_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/npz'
# target_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/'

# # 列出fasta和npz目录下所有文件（去掉扩展名）
# fasta_files = {os.path.splitext(filename)[0] for filename in os.listdir(fasta_directory) if filename.endswith('.fasta')}
# npz_files = {os.path.splitext(filename)[0] for filename in os.listdir(npz_directory) if filename.endswith('.npz')}

# # 找出fasta中比npz多的文件
# extra_fasta_files = fasta_files - npz_files

# # 将多余的文件移动到目标目录
# for extra_file in extra_fasta_files:
#     fasta_file_path = os.path.join(fasta_directory, extra_file + '.fasta')
#     shutil.move(fasta_file_path, target_directory)
#     print(f"Moved: {fasta_file_path}")

# # 输出多余的文件数量
# print(f"Total extra fasta files moved: {len(extra_fasta_files)}")



# #######4、representation.py将超过1022的序列晒出生成npy,需要根据npy筛选fasta和npz
# import os
# import shutil

# # 定义目录路径
# rep1_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/rep1'    #14945
# fasta_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/fasta'  #14988
# npz_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/npz'      #14988
# target_fasta_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/1022+/fasta/'
# target_npz_directory = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/1022+/npz/'

# # 创建目标目录（如果不存在）
# os.makedirs(target_fasta_directory, exist_ok=True)
# os.makedirs(target_npz_directory, exist_ok=True)

# # 列出rep1目录下的文件（去掉扩展名）
# rep1_files = {os.path.splitext(filename)[0] for filename in os.listdir(rep1_directory)}

# # 列出fasta和npz目录下的文件（去掉扩展名）
# fasta_files = {os.path.splitext(filename)[0] for filename in os.listdir(fasta_directory) if filename.endswith('.fasta')}
# npz_files = {os.path.splitext(filename)[0] for filename in os.listdir(npz_directory) if filename.endswith('.npz')}

# # 找出不在rep1目录中的多余fasta和npz文件
# extra_fasta_files = fasta_files - rep1_files
# extra_npz_files = npz_files - rep1_files

# # 移动多余的fasta文件
# for extra_file in extra_fasta_files:
#     fasta_file_path = os.path.join(fasta_directory, extra_file + '.fasta')
#     shutil.move(fasta_file_path, target_fasta_directory)
#     print(f"Moved fasta file: {fasta_file_path}")

# # 移动多余的npz文件
# for extra_file in extra_npz_files:
#     npz_file_path = os.path.join(npz_directory, extra_file + '.npz')
#     shutil.move(npz_file_path, target_npz_directory)
#     print(f"Moved npz file: {npz_file_path}")

# # 输出多余文件数量
# print(f"Total extra fasta files moved: {len(extra_fasta_files)}")
# print(f"Total extra npz files moved: {len(extra_npz_files)}")



# ##########5、训练中出现loss：nan情况，先检查原始数据有没有问题
# import os
# import numpy as np

# npz_dir = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/npz'
# npy_dir = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/rep1'

# def check_nan_in_npz(npz_dir):
#     error_files = []
#     for filename in os.listdir(npz_dir):
#         if filename.endswith('.npz'):
#             file_path = os.path.join(npz_dir, filename)
#             try:
#                 data = np.load(file_path, allow_pickle=True)
#                 for key in data.files:
#                     if np.isnan(data[key]).any():
#                         error_files.append(filename)
#                         print(f"NaN found in {filename}")
#                         break
#             except Exception as e:
#                 print(f"Error loading {filename}: {e}")
#     return error_files

# def check_nan_in_npy(npy_dir):
#     error_files = []
#     for filename in os.listdir(npy_dir):
#         if filename.endswith('.npy'):
#             file_path = os.path.join(npy_dir, filename)
#             try:
#                 data = np.load(file_path, allow_pickle=True)
#                 if np.isnan(data).any():
#                     error_files.append(filename)
#                     print(f"NaN found in {filename}")
#             except Exception as e:
#                 print(f"Error loading {filename}: {e}")
#     return error_files

# npz_errors = check_nan_in_npz(npz_dir)
# npy_errors = check_nan_in_npy(npy_dir)

# print(f"Total .npz files with NaN: {len(npz_errors)}")
# print(f"Total .npy files with NaN: {len(npy_errors)}")

# """
# 检查发现原始数据没问题
# Total .npz files with NaN: 0
# Total .npy files with NaN: 0
# """


# ##########随意筛选1000条序列，找到对应的npy和npz进行训练
# import os
# import random
# import shutil

# # 定义源目录和目标目录
# source_dir = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all'
# target_dir = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/random'
# dirs = ['npz', 'fasta', 'rep1']

# # 确保目标目录下的三个子目录存在
# for d in dirs:
#     os.makedirs(os.path.join(target_dir, d), exist_ok=True)

# # 获取 npz 目录下的文件名列表，并保证其他两个目录中的文件名对应
# npz_files = os.listdir(os.path.join(source_dir, 'npz'))
# npz_files = [f for f in npz_files if f.endswith('.npz')]

# # 随机选择1000个文件名（不包括文件后缀）
# selected_files = random.sample(npz_files, 1000)
# selected_files = [os.path.splitext(f)[0] for f in selected_files]

# # 复制文件到目标目录，确保对应
# for file_name in selected_files:
#     # 处理 npz 和 fasta 文件，保持原格式
#     for d in ['npz', 'fasta']:
#         source_file = os.path.join(source_dir, d, f'{file_name}.{d}')
#         target_file = os.path.join(target_dir, d, f'{file_name}.{d}')
#         if os.path.exists(source_file):
#             shutil.copy(source_file, target_file)
#         else:
#             print(f"Warning: {source_file} does not exist!")

#     # 处理 rep1 文件，使用 .npy 扩展名
#     source_file = os.path.join(source_dir, 'rep1', f'{file_name}.npy')
#     target_file = os.path.join(target_dir, 'rep1', f'{file_name}.npy')
#     if os.path.exists(source_file):
#         shutil.copy(source_file, target_file)
#     else:
#         print(f"Warning: {source_file} does not exist!")

# print("复制完成！")

import numpy as np

# 文件路径
file_path = '/data/datasets/zhangyufei/freeprotmap/yang_datasets/all/loss_nan/2yo3_1_C.npy'

# 加载文件
try:
    data = np.load(file_path)

    # 打印文件内容的相关信息
    print(f"文件形状: {data.shape}")
    print(f"数据类型: {data.dtype}")
    print(f"数据值:\n{data}")
    print(f"最大值: {np.max(data)}")
    print(f"最小值: {np.min(data)}")
    print(f"是否包含 NaN: {np.isnan(data).any()}")
    
except Exception as e:
    print(f"读取文件时出错: {e}")
