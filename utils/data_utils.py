#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.distance import pdist
import os
import random
from utils import kinematics


def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

    Args
      R: a 3x3 rotation matrix
    Returns
      eul: a 3x1 Euler angle representation of R
    """
    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2]);

        if R[0, 2] == -1:
            E2 = np.pi / 2;
            E1 = E3 + dlta;
        else:
            E2 = -np.pi / 2;
            E1 = -E3 + dlta;

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3]);
    return eul


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Args
      R: 3x3 rotation matrix
    Returns
      q: 1x4 quaternion
    """
    rotdiff = R - R.T

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps)

    costheta = (np.trace(R) - 1) / 2

    theta = np.arctan2(sintheta, costheta)

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R))


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Args
      q: 1x4 quaternion
    Returns
      r: 1x3 exponential map
    Raises
      ValueError if the l2 norm of the quaternion is not close to 1
    """
    if np.abs(np.linalg.norm(q) - 1) > 1e-1:
        print()
        raise Exception("quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps))
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

    Args
      normalizedData: nxd matrix with normalized data
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      origData: data originally used to
    """
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if one_hot:
        origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
    else:
        origData[:, dimensions_to_use] = normalizedData

    # potentially ineficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
    """
    Converts the output of the neural network to a format that is more easy to
    manipulate for, e.g. conversion to other format or visualization

    Args
      poses: The output from the TF model. A list with (seq_length) entries,
      each with a (batch_size, dim) output
    Returns
      poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
      batch is an n-by-d sequence of poses.
    """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in range(poses_out.shape[0]):
        poses_out_list.append(
            unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list


def read_series(filename):
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        if "," in line:
            line = line.strip().split(',')
        else:
            line = line.strip().split(' ')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray



def normalize_data(data, data_mean, data_std, dim_to_use, actions, one_hot):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation

    Args
      data: nx99 matrix with data to normalize
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dim_to_use: vector with dimensions used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[key] = np.divide((data[key][:, 0:99] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:, -nactions:]))

    return data_out


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above

    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = torch.zeros(n, 3).float().cuda()
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = torch.zeros(len(idx_spec1), 3).float().cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = torch.zeros(len(idx_spec2), 3).float().cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = torch.zeros(len(idx_remain), 3).float().cuda()
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul


def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N * 3 * 3
    :return: N * 4
    """
    rotdiff = R - R.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = R[:, 0, 0]
    t2 = R[:, 1, 1]
    t3 = R[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = torch.zeros(R.shape[0], 4).float().cuda()
    q[:, 0] = torch.cos(theta / 2)
    q[:, 1:] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q


def expmap2quat_torch(exp):
    """
    Converts expmap to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N*3
    :return: N*4
    """
    theta = torch.norm(exp, p=2, dim=1).unsqueeze(1)
    v = torch.div(exp, theta.repeat(1, 3) + 1e-8)
    sinhalf = torch.sin(theta / 2)
    coshalf = torch.cos(theta / 2)
    q1 = torch.mul(v, sinhalf.repeat(1, 3))
    q = torch.cat((coshalf, q1), dim=1)
    return q


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 1e-12)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = torch.eye(3, 3).repeat(n, 1, 1).float().cuda() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def expmap2xyz(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = kinematics._some_variables()
    xyzs = []
    for i in range(expmap.shape[0]):
        xyz = kinematics.fkl(expmap[i], parent, offset, rotInd, expmapInd)
        xyzs.append(xyz)
    xyz = np.vstack(xyzs)
    return xyz


def expmap2angle_torch(expmap):
    if not isinstance(expmap, torch.Tensor):
        expmap = torch.tensor(expmap).float()
        if torch.cuda.is_available():
            expmap = expmap.cuda()

    parent, offset, rotInd, expmapInd = kinematics._some_variables()
    angle = kinematics.fkl_angle_torch(expmap, parent, offset, rotInd, expmapInd)
    return angle


def expmap2xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    if not isinstance(expmap, torch.Tensor):
        expmap = torch.tensor(expmap).float()
        if torch.cuda.is_available():
            expmap = expmap.cuda()

    parent, offset, rotInd, expmapInd = kinematics._some_variables()
    xyz = kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def expmap2xyz_torch_cmu(expmap):
    if not isinstance(expmap, torch.Tensor):
        expmap = torch.tensor(expmap).float()
        if torch.cuda.is_available():
            expmap = expmap.cuda()

    parent, offset, rotInd, expmapInd = kinematics._some_variables_cmu()
    xyz = kinematics.fkl_torch_cmu(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def tri_xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*96
    """
    # Put them together and revert the coordinate space
    expmap_all = kinematics.revert_coordinate_space(expmap, np.eye(3), np.zeros(3))
    expmap_gt = expmap_all.reshape(len(expmap), -1)
    xyz = expmap2xyz_torch(expmap_gt).reshape(len(expmap), -1)
    return xyz


def tri_xyz_torch_cmu(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*117
    :return: N*114
    """
    # Put them together and revert the coordinate space
    expmap_all = kinematics.revert_coordinate_space_cmu(expmap, np.eye(3), np.zeros(3))
    expmap_gt = expmap_all.reshape(len(expmap), -1)
    xyz = expmap2xyz_torch_cmu(expmap_gt).reshape(len(expmap), -1)
    # print(xyz.shape)  # check if 38*3
    return xyz


def tri_vel_torch(expmap):
    """
    convert expmap to joint velocity
    :param expmap: N*99
    :return: N*96
    """
    xyz = tri_xyz_torch(expmap)
    mid_vel = xyz[1:] - xyz[:-1]
    vel = torch.zeros(xyz.shape, dtype=xyz.dtype)
    vel[1:-1] = (mid_vel[:-1] + mid_vel[1:]) / 2
    return vel


def tri_vel_torch_cmu(expmap):
    """
    convert expmap to joint velocity
    :param expmap: N*117
    :return: N*114
    """
    xyz = tri_xyz_torch_cmu(expmap)
    mid_vel = xyz[1:] - xyz[:-1]
    vel = torch.zeros(xyz.shape, dtype=xyz.dtype)
    vel[1:-1] = (mid_vel[:-1] + mid_vel[1:]) / 2
    return vel


def vel_torch(expmap):
    """
    convert expmap to joint velocity
    :param expmap: N*99
    :return: N*96
    """
    xyz = expmap2xyz_torch(expmap)
    mid_vel = xyz[1:] - xyz[:-1]
    vel = torch.zeros(xyz.shape, dtype=xyz.dtype)
    vel[:-1] = mid_vel
    return vel


def vel_torch_cmu(expmap):
    """
    convert expmap to joint velocity
    :param expmap: N*117
    :return: N*114
    """
    xyz = expmap2xyz_torch_cmu(expmap)
    mid_vel = xyz[1:] - xyz[:-1]
    vel = torch.zeros(xyz.shape, dtype=xyz.dtype)
    vel[:-1] = mid_vel
    return vel


def wel(expmap):
    """
    convert expmap to joint velocity
    :param expmap: N*99
    :return: N*99
    """
    # xyz = expmap[:, 3:]
    xyz = expmap.copy()
    mid_vel = xyz[1:] - xyz[:-1]
    vel = np.zeros(xyz.shape, dtype=xyz.dtype)
    vel[1:] = mid_vel
    return vel


def wel_cmu(expmap):
    """
    convert expmap to joint velocity
    :param expmap: N*117
    :return: N*117
    """
    # xyz = expmap[:, 3:]
    xyz = expmap.copy()
    mid_vel = xyz[1:] - xyz[:-1]
    vel = np.zeros(xyz.shape, dtype=xyz.dtype)
    vel[1:] = mid_vel
    return vel


def trans_torch(expmap):
    """
    convert expmap to joint velocity
    :param expmap: N*99
    :return: N*33*3
    """
    xyz = expmap2xyz_torch(expmap)
    xyz[:, 1:] = xyz[:, 1:] - xyz[:, :-1]
    return xyz


def trans_torch_cmu(expmap):
    """
    convert expmap to joint velocity
    :param expmap: N*117
    :return: N*39*3
    """
    xyz = expmap2xyz_torch_cmu(expmap)
    xyz[:, 1:] = xyz[:, 1:] - xyz[:, :-1]
    return xyz


def axis_angle(v):
    # 计算旋转角度
    B, T, C = v.shape
    v = v.view(B, T, -1, 3)
    theta = torch.norm(v, dim=-1, keepdim=True)  # 保持最后一维的形状不变
    axis = v / (theta + 1e-10)
    axis = axis.flatten(-2)
    theta = theta.flatten(-2)
    return axis, theta

def angle_separate(e):

    assert e.size(-1) == 3
    theta = torch.cos(torch.norm(e, p=2, dim=-1, keepdim=True))
    expmap = torch.sin(e)
    out = torch.cat([theta, expmap], dim=-1)
    return out

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.size(-1) == 3
    theta = torch.norm(e, p=2, dim=-1, keepdim=True)
    w = torch.cos(0.5 * theta)
    xyz = torch.sin(0.5 * theta) / (theta + 1e-8) * e
    q = torch.cat([w, xyz], dim=-1)
    return q

def quaternion_to_expmap(q):
    """
      Converts an exponential map angle to a rotation matrix
      Matlab port to python for evaluation purposes
      I believe this is also called Rodrigues' formula
      https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m
    q is (*, 4)
    return (*, 3)
    examples:
        e = torch.rand(1, 3, 3)
        q = expmap_to_quaternion(e)
        e2 = quaternion_to_expmap(q)
    """
    sinhalftheta = torch.index_select(q, dim=-1, index=torch.tensor([1, 2, 3]).to(q.device))
    coshalftheta = torch.index_select(q, dim=-1, index=torch.tensor([0]).to(q.device))

    norm_sin = torch.norm(sinhalftheta, p=2, dim=-1, keepdim=True)
    r0 = torch.div(sinhalftheta, norm_sin)

    theta = 2 * torch.atan2(norm_sin, coshalftheta)
    theta = torch.fmod(theta + 2 * np.pi, 2 * np.pi)

    theta = torch.where(theta > np.pi, 2 * np.pi - theta, theta)
    r0 = torch.where(theta > np.pi, -r0, r0)
    r = r0 * theta
    return r


def qeuler(q, order='zyx', epsilon=1e-8):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    q0 = torch.index_select(q, dim=-1, index=torch.tensor([0]).to(q.device))
    q1 = torch.index_select(q, dim=-1, index=torch.tensor([1]).to(q.device))
    q2 = torch.index_select(q, dim=-1, index=torch.tensor([2]).to(q.device))
    q3 = torch.index_select(q, dim=-1, index=torch.tensor([3]).to(q.device))

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise ('not defined')

    return torch.cat([x, y, z], dim=-1)


def expmap_to_euler_torch(data):
    # data is (*, feature_dim), feature_dim is multiple of 3
    ori_shp = data.size()
    eul = qeuler(expmap_to_quaternion(data.contiguous().view(-1, 3)))
    return eul.view(ori_shp)


def dct_matrix(N, device='cuda'):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    dct_m = torch.tensor(dct_m, dtype=torch.float32, device=device)
    idct_m = torch.tensor(idct_m, dtype=torch.float32, device=device)
    return dct_m, idct_m

if __name__ == "__main__":
    r = np.random.rand(2, 3) * 10
    # r = np.array([[0.4, 1.5, -0.0], [0, 0, 1.4]])
    r1 = r[0]
    R1 = expmap2rotmat(r1)
    q1 = rotmat2quat(R1)
    # R1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    e1 = rotmat2euler(R1)

    r2 = r[1]
    R2 = expmap2rotmat(r2)
    q2 = rotmat2quat(R2)
    # R2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    e2 = rotmat2euler(R2)

    r = torch.from_numpy(r).cuda().float()
    # q = expmap2quat_torch(r)
    R = expmap2rotmat_torch(r)
    q = rotmat2quat_torch(R)
    # R = Variable(torch.from_numpy(
    #     np.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]]))).cuda().float()
    eul = rotmat2euler_torch(R)
    eul = eul.cpu().data.numpy()
    R = R.cpu().data.numpy()
    q = q.cpu().data.numpy()

    if np.max(np.abs(eul[0] - e1)) < 0.000001:
        print('e1 clear')
    else:
        print('e1 error {}'.format(np.max(np.abs(eul[0] - e1))))
    if np.max(np.abs(eul[1] - e2)) < 0.000001:
        print('e2 clear')
    else:
        print('e2 error {}'.format(np.max(np.abs(eul[1] - e2))))

    if np.max(np.abs(R[0] - R1)) < 0.000001:
        print('R1 clear')
    else:
        print('R1 error {}'.format(np.max(np.abs(R[0] - R1))))

    if np.max(np.abs(R[1] - R2)) < 0.000001:
        print('R2 clear')
    else:
        print('R2 error {}'.format(np.max(np.abs(R[1] - R2))))

    if np.max(np.abs(q[0] - q1)) < 0.000001:
        print('q1 clear')
    else:
        print('q1 error {}'.format(np.max(np.abs(q[0] - q1))))
