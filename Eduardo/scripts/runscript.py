import os
import sys

task1 = sys.argv[1]
#task2 = sys.argv[2]
#fromR = int(sys.argv[3])
fromR = 0
toR = 99
#toR = int(sys.argv[4])
#steps = int(sys.argv[5])

for k in range(fromR,toR):
    print(k)
    ## Used for evolution of LW where 5 is the number of interneurons in a layer
    os.system('python '+task1+' LW 5 '+str(k))
    # os.system('python '+task1+' MCLW 3 '+str(k))
    # os.system('python '+task1+' MCLW 5 '+str(k))
    # os.system('python '+task1+' MCLW 10 '+str(k))
    # os.system('python '+task1+' MCLW 20 '+str(k))
    ## Used for analysis where the last number if the number of steps
    #os.system('python '+task1+' '+str(k)+' MCLW3 3 40')
    #os.system('python '+task1+' '+str(k)+' MCLW5 5 40')
    #os.system('python '+task1+' '+str(k)+' MCLW10 10 40')
    #os.system('python '+task1+' '+str(k)+' MCLW20 20 40')
    #os.system('python '+task2+' '+str(k)+' MCLW3 3 40')
    #os.system('python '+task2+' '+str(k)+' MCLW5 5 40')
    #os.system('python '+task2+' '+str(k)+' MCLW10 10 40')
    #os.system('python '+task2+' '+str(k)+' MCLW20 20 40')
