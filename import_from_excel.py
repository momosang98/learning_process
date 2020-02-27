import pandas as pd

#csv和xlsx不同调用
data=pd.read_excel(r'D:/steam/2016.xlsx')
print(data)
f=open('D:/steam/2017.csv')
data1=pd.read_csv(f)
print(data1)
#打开excel的每个sheet
data2 = pd.read_excel(r'D:/研究生/算法数据测试/测试数据.xlsx',None)#读取数据,设置None可以生成一个字典，字典中的key值即为sheet名字，此时不用使用DataFram，会报错
print(data2.keys())#查看sheet的名字
for sh_name in data2.keys():
    print('sheet_name的名字是：',sh_name)
    sh_data = pd.DataFrame(pd.read_excel(r'D:/研究生/算法数据测试/测试数据.xlsx',sh_name))#获得每一个sheet中的内容
    print(sh_data)



