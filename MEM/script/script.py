x = []

for i in range(len(x)):
    "python3 ../main.py --dir /home/hhjung/hhjung/MEM/{}{}{}{}{}{} -w 1 0 {} {} {} 0 1 0 0 {} {} {} 0 1 --gpu {} 1> /dev/null 2> /dev/null &&".format(x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][6])

for i in range(11):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Norm/Step3/{} -w {:.1f} 1 --norm True --epoch3 300 --epoch4 0 --gpu 0 1> /dev/null 2> /dev/null &&".format(i, j))

for i in range(11):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Norm/Step4/{} -w {:.1f} 1 --norm True --epoch3 0 --epoch4 300 --gpu 0 1> /dev/null 2> /dev/null &&".format(i, j))

for i in range(11):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/None/Step3/{} -w {:.1f} 1 --norm False --epoch3 300 --epoch4 0 --gpu 0 1> /dev/null 2> /dev/null &&".format(i, j))

for i in range(11):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/None/Step4/{} -w {:.1f} 0.1 1 --norm False --epoch3 0 --epoch4 300 --gpu 0 1> /dev/null 2> /dev/null &&".format(i, j))

