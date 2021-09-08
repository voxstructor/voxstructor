#threshold,eer,ASR
#EER
import numpy as np
import matplotlib.pyplot as plt
import sys
def score_pre(scorespath):
    trials = open('ys_trials.txt', 'r').readlines()
    scores = open(scorespath, 'r').readlines()
    spkrutt2target = {}
    for line in trials:
      spkr, utt, target = line.strip().split()
      spkrutt2target[spkr+utt]=target
      # print(spkr+utt)
      #分数
      score_in=[]
      score_each=[]
    for line in scores:
      spkr, utt, score = line.strip().split()
      # print("{} {}".format(score, spkrutt2target[spkr+utt]))
      # print(spkr+utt)
      if spkrutt2target[spkr+utt]== 'nontarget':
         score_each.append(float(score))
      else:
        score_in.append(float(score))
    return  score_in,score_each

#估计阈值
def mk_th(score_in,score_each):
    FRR = []
    FAR = []
    thresld = np.arange(-100, 100, 1)  # 生成模型阈值的等差列表
    # thresld = np.arange(-1, 1, 0.01)  # 生成模型阈值的等差列表
    eer = 1
    for i in range(len(thresld)):
        frr = np.sum(score_in < thresld[i]) / len(score_in)
        FRR.append(frr)

        far = np.sum(score_each > thresld[i]) / len(score_each)
        FAR.append(far)

        if (abs(frr - far) < 0.02):  # frr和far值相差很小时认为相等
            eer = abs(frr + far) / 2
            print('根据eer来确定阈值：')
            print('eer-frr:', frr*100)
            print('eer-far:', far*100)
            print('阈值：', thresld[i])
        if 0.01 < frr < 0.015:
            print('根据frr=0.01来确定阈值：')
            print('frr:', frr*100)
            print('far:', far*100)
            print('阈值：', thresld[i])
        if 0.01 < far < 0.015:
            print('根据far=0.01来确定阈值：')
            print('frr:', frr * 100)
            print('far:', far * 100)
            print('阈值：', thresld[i])
    plt.plot(thresld, FRR, 'x-', label='FRR')
    plt.plot(thresld, FAR, '+-', label='FAR')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()
#计算通过率
def cu_asr(score_in,score_each,eer_th,frr_th,far_th):
    # eer_th=-1.0
    # frr_th=-6.0
    eer_num=0
    frr_num=0
    far_num=0
    acc_eer=0
    acc_frr=0
    for item in score_in:
        if item>=eer_th:
            eer_num=eer_num+1
        if item>=frr_th:
            frr_num=frr_num+1
        if item>=far_th:
            far_num = far_num + 1

    eer_ASR = (eer_num / len(score_in)) * 100
    frr_ASR = (frr_num / len(score_in)) * 100
    far_ASR = (far_num / len(score_in)) * 100
    print('eer_ASR:{}%'.format(eer_ASR))
    print('frr_ASR:{}%'.format(frr_ASR))
    print('far_ASR:{}%'.format(far_ASR))
    print('eer通过数:',eer_num)
    print('frr通过数:',frr_num)
    print('far通过数:',far_num)
    print('长度:',len(score_in))

#欧式距离确定阈值
def mk_th_ou(score_in,score_each):
    FRR = []
    FAR = []
    ACC=[]
    thresld = np.arange(0,200,3)  # 生成模型阈值的等差列表
    eer = 1
    for i in range(len(thresld)):
        frr = np.sum(score_in > thresld[i]) / len(score_in)
        FRR.append(frr)

        far = np.sum(score_each < thresld[i]) / len(score_each)
        FAR.append(far)

        if (abs(frr - far) < 0.02):  # frr和far值相差很小时认为相等
            eer = abs(frr + far) / 2
            print('根据eer来确定阈值：')
            print('eer-frr:', frr*100)
            print('eer-far:', far*100)
            print('阈值：', thresld[i])
        if 0.01 < frr < 0.015:
            print('根据frr=0.01来确定阈值：')
            print('frr:', frr*100)
            print('far:', far*100)
            print('阈值：', thresld[i])
        if 0.01 < far < 0.015:
            print('根据far=0.01来确定阈值：')
            print('frr:', frr * 100)
            print('far:', far * 100)
            print('阈值：', thresld[i])
    plt.plot(thresld, FRR, 'x-', label='FRR')
    plt.plot(thresld, FAR, '+-', label='FAR')
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()

#计算通过率
def ou_cu_asr(score_in,score_each,eer_th,frr_th,far_th):
    # eer_th=-1.0
    # frr_th=-6.0
    eer_num=0
    frr_num=0
    far_num=0
    for item in score_in:
        if item <= eer_th:
            eer_num=eer_num+1
        if item <= frr_th:
            frr_num=frr_num+1
        if item <= far_th:
            far_num = far_num + 1
    eer_ASR = (eer_num / len(score_in))*100
    frr_ASR = (frr_num / len(score_in))*100
    far_ASR = (far_num / len(score_in))*100
    print('eer_ASR:{}%'.format(eer_ASR))
    print('frr_ASR:{}%'.format(frr_ASR))
    print('far_ASR:{}%'.format(far_ASR))
    print('eer通过数:',eer_num)
    print('frr通过数:',frr_num)
    print('far通过数:',far_num)
    print('长度:',len(score_in))

if __name__=='__main__':
    # #ivector_cos距离的阈值计算
    # iv_cos_pp=r'Ivector\iv_score_cos_pp.txt'
    # iv_ou_pp=r'deepspeaker\dv_score_ou_pp.txt'
    # xv_plda_pp=r'D:\pycharm_project\SCORE\Xvector\pv_scores_xvector-plda.txt'
    # dv_score=r'Xvector\PP\pv_scores_xvector-plda.txt'
    #测试通过率
    # d2e_sore=r'Ivector\d2e\d2econvmse-pv-cos-score-tt.txt'
    # i2e_score=r'Dvector\nvm\ysnvm0.5-pv-plda-score.txt'
    # vs_score=r'Dvector\vs\vs-pv-cos-score-tt.txt'
    # rand_score=r'Dvector\rand\rand-pv-ou-score3-tt.txt'
    # score_in, score_each=score_pre(i2e_score)
    # print('score_in:',score_in)
    # print('score_each:',score_each)
    # mk_th(score_in, score_each)
    # mk_th_ou(score_in, score_each)
    # cu_asr(score_in, score_each,-1.0, -6.0,3)#iv-plda
    # cu_asr(score_in, score_each,-3.0, -8.0,1)#xv-plda
    # cu_asr(score_in, score_each,0.06,-0.02,0.128)#iv-cos
    # cu_asr(score_in, score_each,0.67,0.53,0.77)#Xv-cos
    # ou_cu_asr(score_in, score_each,80, 87,75)#dv-ou
    # cu_asr(score_in, score_each,0.35,0.39,0.39)#dv-cos

    scores=[]
    for i in  range(1,10):
        print(str(i))
        i2e_score = r'Ivector\nvm\nvm0.5\ysnvm0.5-'+str(i)+'-pv-plda-score.txt'
        score_in, score_each = score_pre(i2e_score)
        cu_asr(score_in, score_each, -1.0, -6.0, 3)
        print("*************************")