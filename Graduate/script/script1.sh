python3 ../main.py --dir ./mobileNet_1x2_cifar100 --gate True --iter 1 --db cifar100 --gpu 0 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./mobileNet_3x2_cifar100 --gate True --iter 3 --db cifar100 --gpu 1 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./mobileNet_5x2_cifar100 --gate True --iter 5 --db cifar100 --gpu 4 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./mobileNet_base_cifar100 --gate False --db cifar100 --gpu 5 1> /dev/null 2> /dev/null 

