import numpy as np
import cv2
import math

# 节点类
class Node:
    def __init__(self, freq):
        self.left = None
        self.right = None
        self.father = None
        self.freq = freq

    def isLeft(self):
        return self.father.left == self

# 创建叶子节点的函数
def createNodes(freqs):
    return [Node(freq) for freq in freqs]

# 传入节点构造哈夫曼树的函数
def createHuffmanTree(nodes):
    queue = nodes[:]
    while len(queue) > 1:
        queue.sort(key=lambda item: item.freq)
        node_left = queue.pop(0)
        node_right = queue.pop(0)
        node_father = Node(node_left.freq + node_right.freq)
        node_father.left = node_left
        node_father.right = node_right
        node_left.father = node_father
        node_right.father = node_father
        queue.append(node_father)
    queue[0].father = None
    return queue[0]

# 进行哈夫曼编码的函数
def huffmanEncoding(nodes, root):
    codes = [''] * len(nodes)
    for i in range(len(nodes)):
        node_tmp = nodes[i]
        while node_tmp != root:
            if node_tmp.isLeft():
                codes[i] = '0' + codes[i]
            else:
                codes[i] = '1' + codes[i]
            node_tmp = node_tmp.father
    return codes

img = cv2.imread('C:/Users/Lenovo/Desktop/image_1.png')
grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

line = grayimg.shape[0]
row = grayimg.shape[1]
s = np.zeros(256)
n = np.zeros(256)
t = np.zeros(256)
for i in range(line):
    for j in range(row):
        s[grayimg[i][j]] += 1.0
for m in range(256):
    if s[m] == 0:
        continue
    t[m] = s[m] / (line * row)
    n[m] = -t[m] * (math.log2(t[m]))
s = range(0, 256)
s = [str(i) for i in s]
t1 = t * (10 ** 8)
H = sum(n)
chars_freqs = zip(s, t1)
chars_freqs = list(chars_freqs)
nodes = createNodes([item[1] for item in chars_freqs])
root = createHuffmanTree(nodes)
codes = huffmanEncoding(nodes, root)
tt = np.zeros(255)
for i in range(255):
    tt[i] = len(codes[i])
    tt[i] *= t[i]
for item in zip(chars_freqs, codes):
    print('字符:%s freq:%-2d   编码: %s %d' % (item[0][0], item[0][1], item[1], len(item[1])))
# 未编码前平均码长
print('未编码前平均码长:%f' % (sum(tt)))
# 打印哈夫曼编码后平均码长
print('哈夫曼编码后熵（平均码长）:%f' % (H))
eta = H / (sum(tt))
print('哈夫曼编码效率:%f' % (eta))
cv2.waitKey(0)