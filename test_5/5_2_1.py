import numpy as np
import cv2
import math

# 用于构造哈夫曼树的节点类
class Node:
    def __init__(self, freq):
        self.left = None  # 左子节点
        self.right = None  # 右子节点
        self.father = None  # 父节点
        self.freq = freq  # 当前节点的频率

    # 判断当前节点是否为其父节点的左子节点
    def isLeft(self):
        return self.father.left == self

# 创建叶子节点的函数，接收频率列表
def createNodes(freqs):
    return [Node(freq) for freq in freqs]

# 构造哈夫曼树的函数，传入叶子节点
def createHuffmanTree(nodes):
    queue = nodes[:]  # 创建节点队列
    while len(queue) > 1:  # 当队列中只有一个节点时，停止合并
        queue.sort(key=lambda item: item.freq)  # 按频率升序排序
        node_left = queue.pop(0)  # 取出频率最小的节点作为左子节点
        node_right = queue.pop(0)  # 取出频率次小的节点作为右子节点
        # 创建父节点，其频率为左右子节点频率之和
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left  # 左子节点
        node_father.right = node_right  # 右子节点
        node_left.father = node_father  # 设置左子节点的父节点
        node_right.father = node_father  # 设置右子节点的父节点
        queue.append(node_father)  # 将父节点加入队列
    queue[0].father = None  # 根节点没有父节点
    return queue[0]  # 返回哈夫曼树的根节点

# 哈夫曼编码的函数
def huffmanEncoding(nodes, root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]  # 从叶子节点开始
        while node_tmp != root:  # 遍历到根节点
            if node_tmp.isLeft():  # 如果是左子节点，编码为0
                codes[i] = '0' + codes[i]
            else:  # 如果是右子节点，编码为1
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father  # 向上移动到父节点
    return codes  # 返回所有节点的编码

# 读取图片
img = cv2.imread('C:/Users/Lenovo/Desktop/image_1.png')
# 转换为灰度图
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 获取图片的行数和列数
line = grayimg.shape[0]
row = grayimg.shape[1]

# 初始化频率数组、熵数组和概率数组
s = np.zeros(256)  # 每个灰度值出现的频数
n = np.zeros(256)  # 每个灰度值的熵
t = np.zeros(256)  # 每个灰度值的概率

# 遍历灰度图，统计每个灰度值的出现次数
for i in range(line):
    for j in range(row):
        s[grayimg[i][j]] += 1.0

# 计算每个灰度值的概率和熵
for m in range(256):
    if s[m] == 0:  # 跳过频数为0的灰度值
        continue
    t[m] = s[m] / (line * row)  # 计算概率
    n[m] = -t[m] * (math.log2(t[m]))  # 计算熵

# 将灰度值转换为字符串
s = range(0, 256)
s = [str(i) for i in s]
t1 = t * (10 ** 8)  # 将概率放大，方便计算

# 计算总熵
H = sum(n)

# 将字符与频率配对
chars_freqs = zip(s, t1)
chars_freqs = list(chars_freqs)

# 创建哈夫曼树
nodes = createNodes([item[1] for item in chars_freqs])  # 创建叶子节点
root = createHuffmanTree(nodes)  # 构造哈夫曼树
codes = huffmanEncoding(nodes, root)  # 生成哈夫曼编码

# 计算未编码前的平均码长
tt = np.zeros(255)
for i in range(255):
    tt[i] = len(codes[i])  # 编码长度
    tt[i] *= t[i]  # 加权平均

# 打印每个字符的频率和对应编码
for item in zip(chars_freqs, codes):
    print('字符:%s freq:%-2d   编码: %s %d' % (item[0][0], item[0][1], item[1], len(item[1])))

# 打印未编码前的平均码长
print('未编码前平均码长:%f' % (sum(tt)))

# 打印哈夫曼编码后的熵（平均码长）
print('哈夫曼编码后熵（平均码长）:%f' % (H))

# 计算哈夫曼编码效率
eta = H / (sum(tt))
print('哈夫曼编码效率:%f' % (eta))

cv2.waitKey(0)