# # # 1
# # import torch
# # import matplotlib.pyplot as plt
# # def create_linear_data(nums_data,if_plot=False):
# #     x=torch.linspace(0,1,nums_data)
# #     print(x)
# #     x=torch.unsqueeze(x,dim=1)
# #     print(x)
# #     print(x.size())
# #     k=2
# #     y=k*x+torch.rand(x.size())
# #
# #     if if_plot:
# #         plt.scatter(x.numpy(),y.numpy(),c=x.numpy())
# #         plt.show()
# #     data={"x":x,"y":y}
# #     return data
# # data=create_linear_data(300,if_plot=False)
# # print(data["x"].size())
# # print(data["y"].size())
#
# # #2
# # import torch
# # m=torch.nn.Linear(20,30)
# # print(m)
# # input=torch.randn(128,20)
# # output=m(input)
# # print(output)
# #
# # print(output.size())
#
# #3
# import torch
# import matplotlib.pyplot as plt
# # a = torch.zeros(3, 2)
# # print(a)
# # b = torch.ones(12)
# # print(b)
# # n_data = torch.ones(100,2)
# # x0=torch.normal(2*n_data,1)
# # y0 = torch.zeros(100)
# # x1=torch.normal(-2*n_data,1)
# # y1 = torch.ones(100)
# # x=torch.cat((x0,x1),0).type(torch.FloatTensor)
# # print(x)
# # print(x[:,0])
# # y=torch.cat((y0,y1),0).type(torch.FloatTensor)
# # print(x.data.numpy())
# #
# # print(x.data.numpy()[:0])
# # print(x.data.numpy()[2:3])
#
# x=torch.randn(4,4)
# print(x.view(2,8))
# print(x.view(2,8).size())
# y=x.view(-1,4)
# print(y)
# print(y.size())
# import torch
# from torch.autograd import Variable
# a=Variable(torch.Tensor([2,3]),requires_grad=True)
# print(a)
# b=a+3
# print(b)
# c=b*3
# print(c)
# out=c.mean()
#
# print(out)
# out.backward()
# print(a.grad)

import torch
from torch.autograd import Variable
a=Variable(torch.FloatTensor([[2.,4.]]),requires_grad=True)
b=torch.zeros(1,2)
b[0,0]=a[0,0]**2+a[0,1]
b[0,1]=a[0,1]**3+a[0,0]
out=2*b
print(out)
out.backward(torch.FloatTensor([[1.,0.]]))
print(a)
print(b)
print(out)
print(a.grad)