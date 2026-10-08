import numpy as np


#定义一个函数 edge2mat，用于将边列表（edge list）转换为邻接矩阵（adjacency matrix）
# link表示边列表，num_node表示节点数量
def edge2mat(link, num_node):
    #初始化一个全零矩阵A，用于存放邻接矩阵
    A = np.zeros((num_node, num_node))
    #遍历link里的每一条边
    for i, j in link:
        #把第j行第i列的位置设置为1
        A[j, i] = 1
    return A


#对邻接矩阵做有向归一化
def normalize_digraph(A):
    #计算每一列的和
    Dl = np.sum(A, 0)
    h, w = A.shape
    #初始化一个全零矩阵Dn，用于存放对角矩阵
    Dn = np.zeros((w, w))
    #构建Dl的逆
    for i in range(w):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    #用Dl矩阵的逆去归一化邻接矩阵
    AD = np.dot(A, Dn)
    return AD

#构建空间骨架图的 3 个“分区邻接矩阵 自环（I）+ 向心边（In）+ 离心边（Out）
def get_spatial_graph(num_node, self_link, inward, outward):
    #自环：每个节点都和自己相连的一条边，邻接矩阵：只有对角线是 1，其它是 0
    I = edge2mat(self_link, num_node)
    #沿着骨架树，从四肢向“身体中心”方向的有向边，
    In = normalize_digraph(edge2mat(inward, num_node))
    #与向心边相反方向：从“身体中心”向四肢末端扩散的有向边。
    Out = normalize_digraph(edge2mat(outward, num_node))
    #把三个二维矩阵 I、In、Out 沿着一个新维度“堆叠”在一起，形成一个三维数组 A
    A = np.stack((I, In, Out))
    return A


def k_adjacency(A, k, with_self=False, self_factor=1):
    #确保A是numpy数组，
    assert isinstance(A, np.ndarray)
    #构建单位矩阵
    I = np.eye(len(A), dtype=A.dtype)
    #如果k等于0直接返回单位矩阵
    if k == 0:
        return I
    #“k 步可达”矩阵 − “k−1 步可达”矩阵 = “恰好 k 步可达”矩阵
    Ak = np.minimum(np.linalg.matrix_power(A + I, k), 1) \
       - np.minimum(np.linalg.matrix_power(A + I, k - 1), 1)
    #是否在k-hop邻接矩阵里加入“自环”
    if with_self:
        Ak += (self_factor * I)
    return Ak


#把邻接矩阵归一化
def normalize_adjacency_matrix(A):
    node_degrees = A.sum(-1)
    degs_inv_sqrt = np.power(node_degrees, -0.5)
    norm_degs_matrix = np.eye(len(node_degrees)) * degs_inv_sqrt
    return (norm_degs_matrix @ A @ norm_degs_matrix).astype(np.float32)


#边列表转换为邻接矩阵
def get_adjacency_matrix(edges, num_nodes):
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    for edge in edges:
        A[edge] = 1.
    return A

import numpy as np

