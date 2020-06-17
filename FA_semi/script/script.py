for i in range(1, 10):
    j = i * 0.1
    print("python3 ../main.py --dir /home/hhjung/hhjung/MEM/TypeC/Scene4/{:.1f} -w 0.1 0.4 --gamma {:.1f} -mi 100 200 300 --epoch4 400 --noise-rate 0.6 --lr 0.01 --noise-type sym --gpu 0 1> /dev/null 2> /dev/null &&".format(j, j))


