import numpy as np
import torch
from utils import data_utils


def my_correlation(data, threshold=1e-4):
    B, J, T = data.shape

    # 计算每个 B 和 J 的 T 维度上的方差
    var_T = data.var(dim=-1)

    # 根据阈值筛选出需要设置为0的维度
    valid_mask = var_T >= threshold
    valid_mask = valid_mask.unsqueeze(-1) & valid_mask.unsqueeze(-2)  # 生成 J x J 的掩码

    # 计算斯皮尔曼相关系数矩阵
    spearman_corr = spearman(data) + 1

    # 将无效维度的相关系数设置为0
    matrix = spearman_corr.masked_fill(~valid_mask, float('0'))
    indices = torch.arange(J)
    matrix[:, indices, indices] = 1
    matrix = torch.softmax(matrix, dim=-1)
    return matrix


def kinematics_keypoint(n):
    human36 = [0] + [1, 2] + [6, 7] + [13] + [17, 18] + [25, 26]
    cmu_cap = [0] + [2, 3] + [8, 9] + [16] + [21, 22] + [30, 31]

    nodes = []
    if n == 32:
        nodes = human36
    if n == 33:
        nodes = [0] + [i + 1 for i in human36]
    if n == 38:
        nodes = cmu_cap
    if n == 39:
        nodes = [0] + [i + 1 for i in cmu_cap]

    matrix = torch.eye(n, dtype=torch.bool)
    for i in nodes:
        for j in nodes:
            matrix[i, j] = True
            matrix[j, i] = True
    return matrix


def kinematics_tree(n):
    human36 = [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 12,
               16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30]

    cmu_cap = [-1, 0, 1, 2, 3, 4, 5, 0, 7, 8, 9, 10, 11, 0, 13, 14, 15, 16, 17, 18,
               15, 20, 21, 22, 23, 24, 25, 23, 27, 15, 29, 30, 31, 32, 33, 34, 32, 36]

    nodes = []
    if n == 32:
        nodes = human36
    if n == 33:
        nodes = [-1, -1] + [i + 1 for i in human36[1:]]
    if n == 38:
        nodes = cmu_cap
    if n == 39:
        nodes = [-1, -1] + [i + 1 for i in cmu_cap[1:]]

    return nodes


def spearman(x):
    # 对数据进行排序，并计算秩
    rank = torch.argsort(torch.argsort(x, dim=-1, stable=True), dim=-1, stable=True)
    rank = rank.float() + 1  # 秩从1开始

    # 计算秩的差值的平方
    rank_diff = (rank.unsqueeze(-2) - rank.unsqueeze(-3)).pow(2)

    # 计算斯皮尔曼相关系数
    spearman_corr = 1 - (6 * rank_diff.sum(dim=-1)) / (rank.size(-1) * (rank.size(-1) ** 2 - 1))

    return spearman_corr


def pearson(x, eps=1e-8):
    # 中心化
    x_centered = x - x.mean(dim=-1, keepdim=True)  # [B, J, T]

    # 协方差（分子）
    cov = torch.einsum('bjt,bkt->bjk', x_centered, x_centered)  # [B, J, J]

    # 标准差（分母）
    std = x.std(dim=-1, keepdim=False) + eps  # [B, J]
    denom = torch.einsum('bj,bk->bjk', std, std)  # [B, J, J]

    # Pearson 相关系数
    pearson_corr = cov / denom

    return pearson_corr


def _some_variables():
    """
    borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100

    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],  # 5
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],  # 10
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],  # 15
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],  # 21
              [59, 60, 58],
              [],  # 23
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],  # 29
              [77, 78, 76],
              []]  # 31

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd


def _some_variables_cmu():
    """
    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 38-long vector with parent-child relationships in the kinematic tree
      offset: 114-long vector with bone lenghts
      rotInd: 38-long list with indices into angles
      expmapInd: 38-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16,
                       21, 22, 23, 24, 25, 26, 24, 28, 16, 30, 31, 32, 33, 34, 35, 33, 37]) - 1

    offset = 70 * np.array(
        [0, 0, 0, 0, 0, 0, 1.65674000000000, -1.80282000000000, 0.624770000000000, 2.59720000000000, -7.13576000000000,
         0, 2.49236000000000, -6.84770000000000, 0, 0.197040000000000, -0.541360000000000, 2.14581000000000, 0, 0,
         1.11249000000000, 0, 0, 0, -1.61070000000000, -1.80282000000000, 0.624760000000000, -2.59502000000000,
         -7.12977000000000, 0, -2.46780000000000, -6.78024000000000, 0, -0.230240000000000, -0.632580000000000,
         2.13368000000000, 0, 0, 1.11569000000000, 0, 0, 0, 0.0196100000000000, 2.05450000000000, -0.141120000000000,
         0.0102100000000000, 2.06436000000000, -0.0592100000000000, 0, 0, 0, 0.00713000000000000, 1.56711000000000,
         0.149680000000000, 0.0342900000000000, 1.56041000000000, -0.100060000000000, 0.0130500000000000,
         1.62560000000000, -0.0526500000000000, 0, 0, 0, 3.54205000000000, 0.904360000000000, -0.173640000000000,
         4.86513000000000, 0, 0, 3.35554000000000, 0, 0, 0, 0, 0, 0.661170000000000, 0, 0, 0.533060000000000, 0, 0, 0,
         0, 0, 0.541200000000000, 0, 0.541200000000000, 0, 0, 0, -3.49802000000000, 0.759940000000000,
         -0.326160000000000, -5.02649000000000, 0, 0, -3.36431000000000, 0, 0, 0, 0, 0, -0.730410000000000, 0, 0,
         -0.588870000000000, 0, 0, 0, 0, 0, -0.597860000000000, 0, 0.597860000000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[6, 5, 4],
              [9, 8, 7],
              [12, 11, 10],
              [15, 14, 13],
              [18, 17, 16],
              [21, 20, 19],
              [],
              [24, 23, 22],
              [27, 26, 25],
              [30, 29, 28],
              [33, 32, 31],
              [36, 35, 34],
              [],
              [39, 38, 37],
              [42, 41, 40],
              [45, 44, 43],
              [48, 47, 46],
              [51, 50, 49],
              [54, 53, 52],
              [],
              [57, 56, 55],
              [60, 59, 58],
              [63, 62, 61],
              [66, 65, 64],
              [69, 68, 67],
              [72, 71, 70],
              [],
              [75, 74, 73],
              [],
              [78, 77, 76],
              [81, 80, 79],
              [84, 83, 82],
              [87, 86, 85],
              [90, 89, 88],
              [93, 92, 91],
              [],
              [96, 95, 94],
              []]
    posInd = []
    for index in np.arange(38):
        if index == 0:
            posInd.append([1, 2, 3])
        else:
            posInd.append([])

    expmapInd = np.split(np.arange(4, 118) - 1, 38)

    return parent, offset, posInd, expmapInd


def fkl(angles, parent, offset, rotInd, expmapInd):
    """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 99-long vector with 3d position and 3d joint angles in expmap format
    parent: 32-long vector with parent-child relationships in the kinematic tree
    offset: 96-long vector with bone lenghts
    rotInd: 32-long list with indices into angles
    expmapInd: 32-long list with indices into expmap angles
  Returns
    xyz: 32x3 3d points that represent a person in 3d space
  """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        if not rotInd[i]:  # If the list is empty
            xangle, yangle, zangle = 0, 0, 0
        else:
            xangle = angles[rotInd[i][0] - 1]
            yangle = angles[rotInd[i][1] - 1]
            zangle = angles[rotInd[i][2] - 1]

        r = angles[expmapInd[i]]

        thisRotation = data_utils.expmap2rotmat(r)
        thisPosition = np.array([xangle, yangle, zangle])

        if parent[i] == -1:  # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i, :] + thisPosition).dot(xyzStruct[parent[i]]['rotation']) + \
                                  xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    return xyz


def fkl_cmu(angles, parent, offset, rotInd, expmapInd):
    """
  Convert joint angles and bone lenghts into the 3d points of a person.
  Based on expmap2xyz.m, available at
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m

  Args
    angles: 117-long vector with 3d position and 3d joint angles in expmap format
    parent: 38-long vector with parent-child relationships in the kinematic tree
    offset: 114-long vector with bone lenghts
    rotInd: 38-long list with indices into angles
    expmapInd: 39-long list with indices into expmap angles
  Returns
    xyz: 38x3 3d points that represent a person in 3d space
  """

    assert len(angles) == 117

    # Structure that indicates parents for each joint
    njoints = 38
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        if not rotInd[i]:  # If the list is empty
            xangle, yangle, zangle = 0, 0, 0
        else:
            xangle = angles[rotInd[i][0] - 1]
            yangle = angles[rotInd[i][1] - 1]
            zangle = angles[rotInd[i][2] - 1]

        r = angles[expmapInd[i]]

        thisRotation = data_utils.expmap2rotmat(r)
        thisPosition = np.array([xangle, yangle, zangle])

        if parent[i] == -1:  # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i, :] + thisPosition).dot(xyzStruct[parent[i]]['rotation']) + \
                                  xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    return xyz


def fkl_angle_torch(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.

    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset: 32*3
    :return: N*joint_n*3
    """
    n = angles.shape[0]
    j_n = offset.shape[0]
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = data_utils.expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :])

    # expmap = matrix_to_axis_angle(R)

    return None


def fkl_expmap_torch(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.

    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset: 32*3
    :return: N*joint_n*3
    """
    n = angles.shape[0]
    j_n = offset.shape[0]
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = data_utils.expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :])

    # expmap = matrix_to_axis_angle(R)

    return None


def fkl_torch(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.

    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset: 32*3
    :return: N*joint_n*3
    """
    n = angles.shape[0]
    j_n = offset.shape[0]
    p3d = torch.tensor(offset).float().cuda().unsqueeze(0).repeat(n, 1, 1)
    global_pos = angles[:, :3]
    p3d[:, 0, :] += global_pos
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = data_utils.expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        p3d[:, i, :] = torch.matmul(p3d[:, i, :].unsqueeze(1), R[:, parent[i], :, :]).squeeze(1) + p3d[:, parent[i], :]
        R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :])
    return p3d


def fkl_torch_cmu(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.

    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*117
    :param parent:
    :param offset: 38*3
    :return: N*joint_n*3
    """
    n = angles.shape[0]
    j_n = offset.shape[0]
    p3d = torch.tensor(offset).float().cuda().unsqueeze(0).repeat(n, 1, 1)
    global_pos = angles[:, :3]
    p3d[:, 0, :] += global_pos
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = data_utils.expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        p3d[:, i, :] = torch.matmul(p3d[:, i, :].unsqueeze(1), R[:, parent[i], :, :]).squeeze(1) + p3d[:, parent[i], :]
        R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :])
    return p3d


def revert_coordinate_space(channels, R0, T0):
    """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m
  Args
    channels: N*99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
    n, d = channels.shape

    channels_rec = channels.copy()
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for index in range(n):
        R_diff = data_utils.expmap2rotmat(channels[index, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[index, rootRotInd] = data_utils.rotmat2expmap(R)
        T = T_prev + (R_prev.T.dot(np.reshape(channels[index, :3], [3, 1]))).reshape(-1)
        channels_rec[index, :3] = T
        T_prev = T
        R_prev = R

    return channels_rec


def revert_coordinate_space_cmu(channels, R0, T0):
    """
  Bring a series of poses to a canonical form so they are facing the camera when they start.
  Adapted from
  https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/dataParser/Utils/revertCoordinateSpace.m

  Args
    channels: n-by-99 matrix of poses
    R0: 3x3 rotation for the first frame
    T0: 1x3 position for the first frame
  Returns
    channels_rec: The passed poses, but the first has T0 and R0, and the
                  rest of the sequence is modified accordingly.
  """
    n, d = channels.shape

    channels_rec = channels.copy()
    R_prev = R0
    T_prev = T0
    rootRotInd = np.arange(3, 6)

    # Loop through the passed posses
    for index in range(n):
        R_diff = data_utils.expmap2rotmat(channels[index, rootRotInd])
        R = R_diff.dot(R_prev)

        channels_rec[index, rootRotInd] = data_utils.rotmat2expmap(R)
        T = T_prev + (R_prev.T.dot(np.reshape(channels[index, :3], [3, 1]))).reshape(-1)
        channels_rec[index, :3] = T
        # T_prev = T
        # R_prev = R

    return channels_rec


if __name__ == '__main__':
    # Load all the data
    parent, offset, rotInd, expmapInd = _some_variables()

    expmap_gt = np.array([
        0.0000000, 0.0000000, 0.0000000, 0.0000000, 0.0000000, -0.0000000, 0.1163030, -0.0050304, -0.1667500,
        -0.1836191, -0.0000000, -0.0000000, -0.0396493, 0.2748329, -0.1478505, 0.7091483, -0.0000000, -0.0000000,
        -0.0000000, -0.0000000, -0.0000000, 0.1105347, 0.0589145, -0.0312566, -0.1162424, -0.0000000, -0.0000000,
        -0.0374403, -0.2012415, 0.2250142, 0.5077686, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
        0.1687420, -0.0004372, -0.0246601, 0.1179001, -0.0021100, -0.0946258, -1.0612402, 0.0001206, 0.1402206,
        1.5653780, -0.0063802, -0.3188200, -0.0000000, -0.0000000, -0.0000000, 0.2287144, -0.0240626, 1.9653362,
        -0.2543816, 0.4214011, -0.1498438, -0.1686470, -0.0000000, -0.0000000, 0.2558731, -0.3894913, -0.3560411,
        -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
        -0.0000000, -0.0000000, -0.0000000, 0.2035016, 0.0572155, -1.9129609, -0.1114831, -0.8148697, 0.1134339,
        -0.3157842, -0.0000000, -0.0000000, 0.4648643, 0.7236753, 0.3099173, -0.0000000, -0.0000000, -0.0000000,
        -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000
    ])

    xyz_fkl = fkl(expmap_gt, parent, offset, rotInd, expmapInd)

    exp1 = torch.tensor(np.stack([expmap_gt], axis=0)).float().cuda()
    xyz = fkl_torch(exp1, parent, offset, rotInd, expmapInd)
    xyz_fkl_torch = xyz.cpu().data.numpy()

    print("xyz_fkl", xyz_fkl)
    print("xyz_fkl_torch", xyz_fkl_torch)
