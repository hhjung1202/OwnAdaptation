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
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeA/pseudo1entropy4lr1/Sym1 -w 0.1 0.4 --noise-rate 0.1 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(int(x[i][0] * 10), int(x[i][1] * 10), int(x[i][2] * 100), x[i][0], x[i][1], x[i][2]))
    # "python3 ../main.py --dir /home/hhjung/hhjung/MEM/{}{}{}{}{}{} -w 1 0 {} {} {} 0 1 0 0 {} {} {} 0 1 --gpu {} 1> /dev/null 2> /dev/null &&".format(x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][6])

for i in range(1, 5):
    j = i * 0.025
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeB/1_{:.3f} -w 0.1 {:.3f} --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(0.4-j, 0.4-j))
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeB/1_{:.3f} -w 0.1 {:.3f} --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(0.4+j, 0.4+j))

    
for i in range(1, 9):
    j = i * 0.025
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeB/2_{:.3f} -w 0.2 {:.3f} --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(0.8-j, 0.8-j))
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeB/2_{:.3f} -w 0.2 {:.3f} --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(0.8+j, 0.8+j))

        Sym 0.6
x = [[200, 300, 400]
    ,[250, 350, 450]
    ,[250, 300, 350]
    ,[100, 200, 300, 400]]


for i in range(1, 10):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeC/Scene1/{:.1f} -w 0.1 0.4 --gamma {:.1f} -mi 200 300 --epoch4 400 --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(j, j))


for i in range(1, 10):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeC/Scene2/{:.1f} -w 0.1 0.4 --gamma {:.1f} -mi 250 350 --epoch4 450 --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(j, j))


for i in range(1, 10):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeC/Scene3/{:.1f} -w 0.1 0.4 --gamma {:.1f} -mi 250 300 --epoch4 350 --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(j, j))


for i in range(1, 10):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeC/Scene4/{:.1f} -w 0.1 0.4 --gamma {:.1f} -mi 100 200 300 --epoch4 400 --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(j, j))


