x = [   [0.1, 0.1, 0.5],
        [0.1, 0.1, 0.1],
        [0.1, 0.1, 0.05],
        [0.1, 0.1, 0.01],
        [0.1, 0.2, 0.5],
        [0.1, 0.2, 0.1],
        [0.1, 0.2, 0.05],
        [0.1, 0.2, 0.01],
        [0.1, 0.4, 0.5],
        [0.1, 0.4, 0.1],
        [0.1, 0.4, 0.05],
        [0.1, 0.4, 0.01],
        [0.1, 0.8, 0.5],
        [0.1, 0.8, 0.1],
        [0.1, 0.8, 0.05],
        [0.1, 0.8, 0.01],
        [0.2, 0.1, 0.5],
        [0.2, 0.1, 0.1],
        [0.2, 0.1, 0.05],
        [0.2, 0.1, 0.01],
        [0.2, 0.2, 0.5],
        [0.2, 0.2, 0.1],
        [0.2, 0.2, 0.05],
        [0.2, 0.2, 0.01],
        [0.2, 0.4, 0.5],
        [0.2, 0.4, 0.1],
        [0.2, 0.4, 0.05],
        [0.2, 0.4, 0.01],
        [0.2, 0.8, 0.5],
        [0.2, 0.8, 0.1],
        [0.2, 0.8, 0.05],
        [0.2, 0.8, 0.01],
        [0.4, 0.1, 0.5],
        [0.4, 0.1, 0.1],
        [0.4, 0.1, 0.05],
        [0.4, 0.1, 0.01],
        [0.4, 0.2, 0.5],
        [0.4, 0.2, 0.1],
        [0.4, 0.2, 0.05],
        [0.4, 0.2, 0.01],
        [0.4, 0.4, 0.5],
        [0.4, 0.4, 0.1],
        [0.4, 0.4, 0.05],
        [0.4, 0.4, 0.01],
        [0.4, 0.8, 0.5],
        [0.4, 0.8, 0.1],
        [0.4, 0.8, 0.05],
        [0.4, 0.8, 0.01],
        [0.8, 0.1, 0.5],
        [0.8, 0.1, 0.1],
        [0.8, 0.1, 0.05],
        [0.8, 0.1, 0.01],
        [0.8, 0.2, 0.5],
        [0.8, 0.2, 0.1],
        [0.8, 0.2, 0.05],
        [0.8, 0.2, 0.01],
        [0.8, 0.4, 0.5],
        [0.8, 0.4, 0.1],
        [0.8, 0.4, 0.05],
        [0.8, 0.4, 0.01],
        [0.8, 0.8, 0.5],
        [0.8, 0.8, 0.1],
        [0.8, 0.8, 0.05],
        [0.8, 0.8, 0.01],   ]

for i in range(len(x)):
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Param/pseudo{}entropy{}lr{} -w {:.1f} {:.1f} --noise-rate 0.6 --lr {:.2f} --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(int(x[i][0] * 10), int(x[i][1] * 10), int(x[i][2] * 100), x[i][0], x[i][1], x[i][2]))
    # "python3 ../main.py --dir /home/hhjung/hhjung/MEM/{}{}{}{}{}{} -w 1 0 {} {} {} 0 1 0 0 {} {} {} 0 1 --gpu {} 1> /dev/null 2> /dev/null &&".format(x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][6])

for m in range(1, 4):
    k = m*0.2
    for i in range(1, 11):
        j = i * 0.1
        print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/Param/{} -w {:.1f} {:.1f} --noise-rate 0.6 --lr {:.2f} --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(k, i, j, k))

        Sym 0.6

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

