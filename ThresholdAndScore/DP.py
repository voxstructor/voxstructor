#隐私度量
#N(0,0.5)
import numpy as np
def NV(vectorpath,dppath):
    vectors=open(vectorpath).readlines()
    file=open(dppath,mode='a')
    for line in vectors:
        data=line.strip().replace('[','').replace(']','').split()
        filename=data[0]
        vector=np.array(list(map(float, data[1:])))
        rp=np.random.normal(0,5,512)
        dp_vector=vector+rp
        dp_vector= list(np.around(dp_vector, 7))
        a=data[0]+'  '+str(dp_vector).replace(',','').replace('[','[ ').replace(']',' ]')+'\n'
        file.write(a)
    file.close()

#产生正交矩阵
def OM(a,b,dim=3):
     random_state = np.random
     H = np.eye(dim)
     D = np.ones((dim,))
     for n in range(1, dim):
         x = random_state.normal(a,b,size=(dim-n+1,))
         D[n-1] = np.sign(x[0])
         x[0] -= D[n-1]*np.sqrt((x*x).sum())
         # Householder transformation
         Hx = (np.eye(dim-n+1) - 2.*np.outer(x, x)/(x*x).sum())
         mat = np.eye(dim)
         mat[n-1:, n-1:] = Hx
         H = np.dot(H, mat)
         # Fix the last sign such that the determinant is 1
     D[-1] = (-1)**(1-(dim % 2))*D.prod()
     # Equivalent to np.dot(np.diag(D), H) but faster, apparently
     H = (D*H.T).T
     return H
#加上随机矩阵的隐私保护方法
def NM(vectorpath,dppath):
    vectors = open(vectorpath).readlines()
    file = open(dppath, mode='a')
    for line in vectors:
        data = line.strip().replace('[', '').replace(']', '').split()
        filename = data[0]
        vector = np.array(list(map(float, data[1:])))
        rp = OM(0,5,512)
        dp_vector = np.dot(rp,vector)
        dp_vector = list(np.around(dp_vector, 7))
        a = data[0] + '  ' + str(dp_vector).replace(',', '').replace('[', '[ ').replace(']', ' ]') + '\n'
        file.write(a)
    file.close()

def NVM(vectorpath,dppath):
    vectors = open(vectorpath).readlines()
    file = open(dppath, mode='a')
    for line in vectors:
        data = line.strip().replace('[', '').replace(']', '').split()
        filename = data[0]
        vector = np.array(list(map(float, data[1:])))
        mr = OM(0,5,512)
        dp_vector = np.dot(mr,vector)
        vr = np.random.normal(0,5,512)
        dp_vector =dp_vector + vr
        dp_vector = list(np.around(dp_vector, 7))
        a = data[0] + '  ' + str(dp_vector).replace(',', '').replace('[', '[ ').replace(']', ' ]') + '\n'
        file.write(a)
    file.close()
if __name__=='__main__':
    vectorpath=r'Xvector\vector\ys-lbs-xvector.txt'
    # dppath='Ivector\\vector\\ys-nv2-ivector-'
    for i in range(1,10):
        dppath = r'Xvector\vector\ys-nv5-xvector-'
        dppath=dppath+str(i)+'.txt'
        NV(vectorpath,dppath)