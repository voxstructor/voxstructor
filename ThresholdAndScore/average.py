#取向量的均值作为平均，读取数据
import numpy as np
vector=[0.,0.,0.,0.]
lines=open(r'Dvector\vector\p-dvector.txt').readlines()
file=open(r'Dvector\vector\spk-p-dvector2.txt',mode='a')
filename='id'
j=0
for i,line in enumerate(lines):
    data=line.strip().replace('[','').replace(']','').split()
    if filename==data[0][:7]:
        continue
    else:
        j = j + 1
        filename = data[0][:7]
        print(filename)
        vector[0]=np.array(list(map(float, data[1:])))
        vector[1]=np.array(list(map(float, lines[i+1].strip().replace('[','').replace(']','').split()[1:])))
        vector[2]=np.array(list(map(float, lines[i+2].strip().replace('[','').replace(']','').split()[1:])))
        vector[3]=np.array(list(map(float, lines[i+3].strip().replace('[','').replace(']','').split()[1:])))
        vmean=(vector[0]+vector[1]+vector[2]+vector[3])/4
        # print(vector[0])
        # print(vector[1])
        # print(vector[2])
        # print(vector[3])
        # print(vmean)
        vmean = list(np.around(vmean, 7))
        a = 'id10270-GWXujl-xAVM-000'+str(j).zfill(2) + '  ' + str(vmean).replace(',', '').replace('[', '[ ').replace(']', ' ]') + '\n'
        file.write(a)
        # if i==1:
        #     break
file.close()

