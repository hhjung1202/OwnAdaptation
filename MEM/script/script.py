x = [[0,0,0,0,0,0,0],
[0,0,0,0,0,1,0],
[0,0,0,0,1,0,0],
[0,0,0,0,1,1,0],
[0,0,0,1,0,0,0],
[0,0,0,1,0,1,0],
[0,0,0,1,1,0,0],
[0,0,0,1,1,1,0],
[0,0,1,0,0,0,1],
[0,0,1,0,0,1,1],
[0,0,1,0,1,0,1],
[0,0,1,0,1,1,1],
[0,0,1,1,0,0,1],
[0,0,1,1,0,1,1],
[0,0,1,1,1,0,1],
[0,0,1,1,1,1,1],
[0,1,0,0,0,0,2],
[0,1,0,0,0,1,2],
[0,1,0,0,1,0,2],
[0,1,0,0,1,1,2],
[0,1,0,1,0,0,2],
[0,1,0,1,0,1,2],
[0,1,0,1,1,0,2],
[0,1,0,1,1,1,2],
[0,1,1,0,0,0,3],
[0,1,1,0,0,1,3],
[0,1,1,0,1,0,3],
[0,1,1,0,1,1,3],
[0,1,1,1,0,0,3],
[0,1,1,1,0,1,3],
[0,1,1,1,1,0,3],
[0,1,1,1,1,1,3],
[1,0,0,0,0,0,4],
[1,0,0,0,0,1,4],
[1,0,0,0,1,0,4],
[1,0,0,0,1,1,4],
[1,0,0,1,0,0,4],
[1,0,0,1,0,1,4],
[1,0,0,1,1,0,4],
[1,0,0,1,1,1,4],
[1,0,1,0,0,0,5],
[1,0,1,0,0,1,5],
[1,0,1,0,1,0,5],
[1,0,1,0,1,1,5],
[1,0,1,1,0,0,5],
[1,0,1,1,0,1,5],
[1,0,1,1,1,0,5],
[1,0,1,1,1,1,5],
[1,1,0,0,0,0,6],
[1,1,0,0,0,1,6],
[1,1,0,0,1,0,6],
[1,1,0,0,1,1,6],
[1,1,0,1,0,0,6],
[1,1,0,1,0,1,6],
[1,1,0,1,1,0,6],
[1,1,0,1,1,1,6],
[1,1,1,0,0,0,7],
[1,1,1,0,0,1,7],
[1,1,1,0,1,0,7],
[1,1,1,0,1,1,7],
[1,1,1,1,0,0,7],
[1,1,1,1,0,1,7],
[1,1,1,1,1,0,7],
[1,1,1,1,1,1,7],]

for i in range(len(x)):
    "python3 ../main.py --dir /home/hhjung/hhjung/MEM/{}{}{}{}{}{} -w 1 0 {} {} {} 0 1 0 0 {} {} {} 0 1 --gpu {} 1> /dev/null 2> /dev/null &&".format(x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][0],x[i][1],x[i][2],x[i][3],x[i][4],x[i][5],x[i][6])