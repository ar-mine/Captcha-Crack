import os

# ----------------公共参数-------------------
# 数据集根目录
DATASET_ROOT = "F:/Dataset/Captcha/"
# 数据集图片与数据集标签路径
IMAGE_DIR = os.path.join(DATASET_ROOT, 'img')
LABEL_DIR = os.path.join(DATASET_ROOT, 'rec')
# 生成标签文本名称
LABEL_NAME = "label.txt"
# 默认字体文件位置
DEFAULT_FONTS = [os.path.join('./asset/font', 'simhei.ttf')]
# 生成图片数量，大小
GEN_NUM, WIDTH, HEIGHT = 25000, 160, 80
# 字符类别，长度
NUM_CLASSES, C_LEN = 10, 4
# batch大小
BATCH_SIZE = 128
# 字符字典集
DICTSET = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
           'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
           'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u',
           'v', 'w', 'x', 'y', 'z', 'A', 'B', 'D', 'E', 'F',
           'G', 'H', 'L', 'M', 'N', 'Q', 'R', 'T', 'Y']


# ----------------超参数-------------------



