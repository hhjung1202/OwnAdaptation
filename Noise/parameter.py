parser = argparse.ArgumentParser(description='PyTorch Cycle Domain Adaptation Training')
parser.add_argument('--sd', default='mnist', type=str, help='source dataset')
parser.add_argument('--td', default='svhn', type=str, help='target dataset')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers (default: 4)')
parser.add_argument('--epoch', default=200, type=int, metavar='N', help='number of total epoch to run')
parser.add_argument('--decay-epoch', default=100, type=int, metavar='N', help='epoch from which to start lr decay')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size (default: 256)')

parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')

parser.add_argument('--latent-dim', type=int, default=100, help='dimensionality of the latent space')
parser.add_argument('--img-size', type=int, default=32, help='input image width, height size')
parser.add_argument('--max-buffer', type=int, default=1024, help='Fake GAN Buffer Image')

parser.add_argument('--layer', type=int, default=20, help='[14, 20, 32, 44, 56, 110]')

parser.add_argument('--dir', default='./', type=str, help='default save directory')
parser.add_argument('--gpu', default='0', type=str, help='Multi GPU ids to use.')

parser.add_argument('--cycle', type=float, default=10.0, help='Cycle Consistency Parameter')
parser.add_argument('--identity', type=float, default=5.0, help='Identity Consistency Parameter')
parser.add_argument('--cls', type=float, default=1.0, help='[A,y] -> G_AB -> G_BA -> [A_,y] Source Class Consistency Parameter')

parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')



parameter 

lr 1e-3
epoch 
model 
	layer
	chIn
	clsN
scheduler step
	milestones = [30, 70]
	gamma 0.1
	last_epoch 학습중인 epoch

scheduler lambdaLR
	decay_epoch = 100
	load_epoch = 50

optimizer
	Adam
		lr
		b1, b2
		weight_decay
	SGD
		lr
		momentum
		weight_decay


