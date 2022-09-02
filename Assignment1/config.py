import argparse


def parseArguments():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n_classes', help="Number of classes", type=int, default=10)
	parser.add_argument('--n_hlayers', type=int, default=2)
	parser.add_argument('-l', '--layer_sizes', nargs='+', type=int, required=False)
	parser.add_argument('--l_rate', type=float, default=0.001)
	parser.add_argument('--epochs', type=int, default=20)
	parser.add_argument('--optimizer', type=str, required=True)
	parser.add_argument('--activation', type=str, default='sigmoid')
	parser.add_argument('--loss', type=str, default='cross_entropy')
	parser.add_argument('--output_activation', type=str, default='softmax')
	parser.add_argument('--batch_size', type=int, default=1)
	parser.add_argument('--initializer', type=str, default='xavier')
	parser.add_argument('--hlayer_size', type=int, default=32)
	args = parser.parse_args()

	return args
