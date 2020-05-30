from argparse import ArgumentParser
from utils import *
from train import Train
from test import Test
from evaluate import Evaluate

def parse_args():
	parser = ArgumentParser(description='Depth Map Prediction from single image')
	parser.add_argument('--phase', type=str, default='train', help='[train / test / evaluate]')
	parser.add_argument('--dataset', '-d', type=str, default='nyu_depth',  help='[nyu_depth / kitti]')
	parser.add_argument('--model', '-m', type=str, default='eigen', help='[eigen / eigenv2]')

	parser.add_argument('--batch_size', type=int, default=32, help='The size of batch size')
	parser.add_argument('--lr', type=float, default=0.001, help='The learning rate')
	#TODO: do we need multiple epochs since we are doing training into two parts - Coarse - fine
	parser.add_argument('--epochs', type=int, default=50, help='The no of iteration for the training')

	parser.add_argument('--result_dir', type=str, default='results', help='Directory name to save the results')
	return check_args(parser.parse_args())

def check_args(args):
	# creating result directory
	check_folder(os.path.join(args.result_dir, args.model, args.dataset, 'model'))
	check_folder(os.path.join(args.result_dir, args.model, args.dataset, 'logs'))
	check_folder(os.path.join(args.result_dir, args.model, args.dataset, 'test'))

	return args

def main():
	args = parse_args()

	if args.phase == "train":
		runner = Train(args)
		runner.execute()
		print(" [*] Training finished!")

	if args.phase == "test":
		runner = Test(args)
		runner.execute()
		print(" [*] Testing finished!")

	if args.phase == "evaluate":
		runner = Evaluate(args)
		runner.execute()
		print(" [*] Evaluation finished!")

if __name__ == '__main__':
	main()