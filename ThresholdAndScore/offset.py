
rand_ark=open(r'Ivector\vector\spk_p_ivector2.txt').readlines()
# rand_scp=open(r'Ivector\vector\spk_p_ivector2.scp',mode='a')
ark=open(r'Ivector\vector\spk_p_ivector2.txt')
read_scp=open(r'Ivector\vector\spk_p_ivector2.scp').readlines()

# offset=26
# for line in rand_ark:
#     data=line.strip().split()
#     filename=data[0]+' exp/ivectors_out_voice/ivector.1.ark'
#     res=filename+':'+str(offset)+'\n'
#     rand_scp.write(res)
#     ark.readline()
#     offset=ark.tell()+26

#测试数据
for line in read_scp:
    data=line.strip().split()
    filename=data[0]
    index=int(data[-1].split(':')[-1])
    print('索引：',index)
    # ark.seek(index-26)
    ark.seek(index)
    print(ark.readline())