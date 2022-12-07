'''
Author: zengyong 2595650269@qq.com
Date: 2022-12-07 23:00:34
LastEditors: zengyong 2595650269@qq.com
LastEditTime: 2022-12-07 23:02:01
FilePath: \gender-classification\test.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
a = [torch.tensor([1,2]), torch.tensor([2,3]), torch.tensor([3,4])]
print(torch.cat(a, dim=-1))
