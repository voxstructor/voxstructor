#计算所有的余弦
import numpy as np
def cos_fun(a,b):
    a=np.array(a)
    b=np.array(b)
    a_norm=np.linalg.norm(a)
    b_norm= np.linalg.norm(b)
    dot_res=np.dot(a,b)
    # print('dot:',dot_res)
    return  dot_res/(a_norm*b_norm)
#根据trials来计算距离
def cos_score(vector1path,vector2path,scorepath):
    trials=open('ys_trials.txt').readlines()
    vector1={}
    vector2={}
    #将数据写入字典中
    for line in open(vector1path).readlines():
        data = line.strip().replace('[', '').replace(']', '').replace(',', '').split()
        vector1[data[0]]=list(map(float, data[1:]))
    for line in open(vector2path).readlines():
        data = line.strip().replace('[', '').replace(']', '').replace(',', '').split()
        vector2[data[0]] = list(map(float, data[1:]))

    score_file=open(scorepath,mode='a')
    for line in trials:
        ids=line.strip().split()
        ids=ids[:-1]
        v1=vector1[ids[0]]
        v2=vector2[ids[1]]
        cos_re=cos_fun(v1,v2)
        score_file.write(ids[0]+' '+ids[1]+' '+str(cos_re)+'\n')
    score_file.close()
#根据trials来计算欧式距离
def ou_score(vector1path,vector2path,scorepath):
    trials=open('ys_trials.txt').readlines()
    vector2=open(vector2path).readlines()
    vector1={}
    vector2={}
    #将数据写入字典中
    for line in open(vector1path).readlines():
        data = line.strip().replace('[', '').replace(']', '').replace(',', '').split()
        vector1[data[0]]=list(map(float, data[1:]))
    for line in open(vector2path).readlines():
        data = line.strip().replace('[', '').replace(']', '').replace(',', '').split()
        vector2[data[0]] = list(map(float, data[1:]))

    score_file=open(scorepath,mode='a')
    for line in trials:
        ids=line.strip().split()
        ids=ids[:-1]
        v1=np.array(vector1[ids[0]])
        v2=np.array(vector2[ids[1]])
        ou_re=np.linalg.norm(v1-v2)
        score_file.write(ids[0]+' '+ids[1]+' '+str(ou_re)+'\n')
    score_file.close()
if __name__=='__main__':
    #ivectorcos
    pv_path=r'Ivector\nvm\vector\ys-lbs-ivector.txt'
    # E_path=r'Ivector\nvm\vector\ysnvm0.5-ivector.txt'
    # X2E_path=r'Ivector\vector\x2el1l2-ivector.txt'
    # VS_path=r'D:\pycharm_project\SCORE\Ivector\vector\vs2_ivector.txt'
    # rand_path=r'D:\pycharm_project\SCORE\Xvector\vector\pw_rand_xvector.txt'
    # X2E_path = r'D:\pycharm_project\SCORE\Xvector\vector\i2econvmse-xvector.txt'
    # score_path= r'Ivector\nvm\ysnvm0.5-pv-cos-score.txt'
    # score_ou_path=r'Dvector\nvm\ysnvm0.5-pv-ou-score.txt'
    # cos_score(pv_path,E_path,score_path)
    # ou_score(pv_path,E_path,score_ou_path)Ivector\nm\nm0.5\vector\ysnm0.5-1-ivector.txt   -7-pv-cos-score.txt
    for i in range(1,10):
        E_path1=r'Ivector\nvm\nvm0.5\vector\ysnvm0.5-'+str(i)+'-ivector.txt'
        score_path1=r'Ivector\nvm\nvm0.5\ysnvm0.5-'+str(i)+'-pv-cos-score.txt'
        cos_score(pv_path, E_path1, score_path1)
