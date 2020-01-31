x = []

for i in range(len(x)):
    "python3 ../main.py --dir /home/hhjung/hhjung/MEM/{}{}{}{}{}{} -w 1 0 {} {} {} 0 1 0 0 {} {} {} 0 1 --gpu {} 1> /dev/null 2> /dev/null &&".format(x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][6])

for m in range(1, 4):
    k = m*0.2
    for i in range(1, 11):
        j = i * 0.1
        print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Sym{:.1f}/Step3/{} -w {:.1f} 1 --noise-rate {:.1f} --noise-type sym --epoch3 300 --epoch4 0 --gpu 0 1> /dev/null 2> /dev/null &&".format(k, i, j, k))

for m in range(1, 4):
    k = m*0.2
    t = m*0.1
    for i in range(1, 11):
        j = i * 0.1
        print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Asym{:.1f}/Step3/{} -w {:.1f} 1 --noise-rate {:.1f} --noise-type Asym --epoch3 300 --epoch4 0 --gpu 0 1> /dev/null 2> /dev/null &&".format(t, i, j, k))        


for m in range(1, 4):
    k = m*0.2
    for i in range(1, 11):
        j = i * 0.1
        print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Sym{:.1f}/Step4/{} -w {:.1f} 1 --noise-rate {:.1f} --noise-type sym --epoch3 0 --epoch4 300 --gpu 0 1> /dev/null 2> /dev/null &&".format(k, i, j, k))

for m in range(1, 4):
    k = m*0.2
    t = m*0.1
    for i in range(1, 11):
        j = i * 0.1
        print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Asym{:.1f}/Step4/{} -w {:.1f} 1 --noise-rate {:.1f} --noise-type Asym --epoch3 0 --epoch4 300 --gpu 0 1> /dev/null 2> /dev/null &&".format(t, i, j, k))


for i in range(1,11):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Norm/Step4/{} -w {:.1f} 1 --epoch3 0 --epoch4 300 --gpu 0 1> /dev/null 2> /dev/null &&".format(i, j))

