(python3 ../main.py --dir ./dense_64_cifar100 --milestones 150 250 350 --gate False --switch False --growth 12 --init 24 --model DenseNet --layer 64 --db cifar100 --gpu 0 1> /dev/null 2> /dev/null &&
python3 ../main.py --dir ./dense_64_cifar10_gr6 --milestones 150 250 350 --gate False --switch False --growth 6 --init 24 --model DenseNet --layer 64 --db cifar100 --gpu 0 1> /dev/null 2> /dev/null) &
python3 ../main.py --dir ./dense_28_cifar100_gr6 --milestones 150 250 350 --gate False --switch False --growth 6 --init 24 --model DenseNet --layer 28 --db cifar100 --gpu 1 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./dense_40_cifar100_gr6 --milestones 150 250 350 --gate False --switch False --growth 6 --init 24 --model DenseNet --layer 40 --db cifar100 --gpu 2 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./dense_52_cifar100_gr6 --milestones 150 250 350 --gate False --switch False --growth 6 --init 24 --model DenseNet --layer 52 --db cifar100 --gpu 3 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./dense_64_cifar100_gr6 --milestones 150 250 350 --gate False --switch False --growth 6 --init 24 --model DenseNet --layer 64 --db cifar100 --gpu 4 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./dense_28_cifar10_gr6 --milestones 150 250 350 --gate False --switch False --growth 6 --init 24 --model DenseNet --layer 28 --db cifar100 --gpu 5 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./dense_40_cifar10_gr6 --milestones 150 250 350 --gate False --switch False --growth 6 --init 24 --model DenseNet --layer 40 --db cifar100 --gpu 6 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./dense_52_cifar10_gr6 --milestones 150 250 350 --gate False --switch False --growth 6 --init 24 --model DenseNet --layer 52 --db cifar100 --gpu 7 1> /dev/null 2> /dev/null &


