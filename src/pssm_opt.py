import numpy as np
import os


def readPSSM(path_in):
    '''
    从'/D8244数据集/D8244_pssmMatrix/XXX.txt'中读取 PSSM
    :param path_in: 读取的文件路径
    :return: 提取的PSSM数组
    '''
    pssm_arr = []

    file_arr = os.listdir(path_in)
    file_arr.sort()

    for file in file_arr:
        with open(os.path.join(path_in, file)) as afile:
            pssm_tmp = []
            afile.readlines(3)

            for line in afile.readlines():
                line_tmp = line.split()
                if line_tmp.__len__() == 44:
                    pssm_tmp.append(np.array(line_tmp[2:22]))

            pssm_arr.append(np.array(pssm_tmp))

    return pssm_arr


def writePSSM(pssm_arr, path_out, path_in):
    '''
    将pssm_arr写入对应目录文件下
    :param pssm_arr: PSSM数组
    :param path_out: 写入的文件路径
    :param path_in: 读取的文件路径
    :return:
    '''
    file_arr = []
    for filename in os.listdir(path_in):
        file_tmp = os.path.join(path_out,filename)
        file_arr.append(file_tmp)

    i = 0
    for file in file_arr:
        with open(file, "w") as afile:
            str = pssm_arr[i].__str__()
            afile.write(str)
            i += 1

    return

# 防止矩阵不能输出完整
np.set_printoptions(threshold=np.inf)

path_in = '/Users/yuanchao/Downloads/D8244数据集/D8244_pssmMatrix'
pssm_arr = readPSSM(path_in)

# 在当前文件路径下创建
abspath = os.path.abspath('.')
path_out = os.path.join(abspath, 'D8244PSSM')
try:
    os.makedirs(path_out)
    print("创建目录成功")
except:
    # 如果已经存在该文件，则会显示创建失败
    print("创建目录失败失败")

writePSSM(pssm_arr,path_out,path_in)
