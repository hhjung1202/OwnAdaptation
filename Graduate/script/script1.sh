python3 ../main.py --dir ./mobileNet_cifar100 --milestones 150 250 350 --gate True --switch False --db cifar100 --gpu 0 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./mobileNet_base_cifar100 --milestones 80 120 165 --gate False --switch False --db cifar100 --gpu 5 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./mobileNet_cifar10 --milestones 150 250 350 --gate True --switch False --db cifar10 --gpu 0 1> /dev/null 2> /dev/null &
python3 ../main.py --dir ./mobileNet_base_cifar10 --milestones 80 120 165 --gate False --switch False --db cifar10 --gpu 5 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./dense_28_cifar100 --milestones 150 250 350 --gate False --switch False --growth 12 --init 24 --model DenseNet --layer 28 --db cifar100 --gpu 6 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./dense_base_28_cifar100 --milestones 80 120 165 --gate False --switch False --growth 6 --init 24 --model DenseNet_Base --layer 28 --db cifar100 --gpu 5 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./dense_28_cifar10 --milestones 150 250 350 --gate False --switch False --growth 12 --init 24 --model DenseNet --layer 28 --db cifar10 --gpu 6 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./dense_base_28_cifar10 --milestones 80 120 165 --gate False --switch False --growth 6 --init 24 --model DenseNet_Base --layer 28 --db cifar10 --gpu 5 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./dense_40_cifar100 --milestones 150 250 350 --gate False --switch False --growth 12 --init 24 --model DenseNet --layer 40 --db cifar100 --gpu 6 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./dense_base_40_cifar100 --milestones 80 120 165 --gate False --switch False --growth 6 --init 24 --model DenseNet_Base --layer 40 --db cifar100 --gpu 5 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./dense_40_cifar10 --milestones 150 250 350 --gate False --switch False --growth 12 --init 24 --model DenseNet --layer 40 --db cifar10 --gpu 6 1> /dev/null 2> /dev/null 
python3 ../main.py --dir ./dense_base_40_cifar10 --milestones 80 120 165 --gate False --switch False --growth 6 --init 24 --model DenseNet_Base --layer 40 --db cifar10 --gpu 5 1> /dev/null 2> /dev/null 