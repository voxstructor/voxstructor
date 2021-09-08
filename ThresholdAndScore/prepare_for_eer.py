#!/usr/bin/env python3
#
# Copyright 2015   David Snyder
# Apache 2.0.
#
# Copied from egs/sre10/v1/local/prepare_for_eer.py (commit 9cb4c4c2fb0223ee90c38d98af11305074eb7ef8)
#
# Given a trials and scores file, this script
# prepares input for the binary compute-eer.
import numpy as np
import matplotlib.pyplot as plt
import sys
# trials = open(sys.argv[1], 'r').readlines()
# scores = open(sys.argv[2], 'r').readlines()



#计算eer


#  #通过率
# epoch=0
# for item in score_in:
#     if item>=-2.0:
#         epoch=epoch+1
#
# ASR = epoch / len(score_in)
# print('ASR:',ASR)
# print('通过数:',epoch)
# print('长度:',len(score_in))
# print('num:',num)

# user_id_length = len(data.values[0, 1:])  # 要识别的数量
# model_id_length = len(data.values[1:, 0])  # 计算出模型ID数量
#
# for i in range(user_id_length):
#   for j in range(model_id_length):
#     # 需要识别的用户id和模型id一样，就认为是类内测试，否则是类间测试
#     if data.values[i + 1][0] == data.values[0][j + 1]:
#       class_in.append(np.float(data.values[i + 1][j + 1]))
#     else:
#       class_each.append(np.float(data.values[i + 1][j + 1]))

