import torch
# 1d绝对sin_cos编码
def create_1d_absolute_sin_cos_embedding(pos_len, dim):
    assert dim % 2 == 0, "wrong dimension!"
    position_emb = torch.zeros(pos_len, dim, dtype=torch.float)
    # i矩阵
    i_matrix = torch.arange(dim//2, dtype=torch.float)
    i_matrix /= dim / 2
    i_matrix = torch.pow(10000, i_matrix)
    i_matrix = 1 / i_matrix
    i_matrix = i_matrix.to(torch.long)
    # pos矩阵
    pos_vec = torch.arange(pos_len).to(torch.long)
    # 矩阵相乘，pos变成列向量，i_matrix变成行向量
    out = pos_vec[:, None] @ i_matrix[None, :]
    # 奇/偶数列
    emb_cos = torch.cos(out)
    emb_sin = torch.sin(out)
    # 赋值
    position_emb[:, 0::2] = emb_sin
    position_emb[:, 1::2] = emb_cos
    return position_emb