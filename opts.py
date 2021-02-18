import argparse
parser = argparse.ArgumentParser(description="PyTorch implementation of Video Re-loc")

parser.add_argument('--inter_op_parallelism_threads', default=0, type=int,
                    help='number of threads')
parser.add_argument('--intra_op_parallelism_threads', default=0, type=int,
                    help='number of threads')                    
parser.add_argument('--max_length', default=300, type=int,
                    help='max length')
parser.add_argument('--feat_dim', default=500, type=int,
                    help='feature dim')
parser.add_argument('--keep_prob', default=0.6, type=float,
                    help='keep prob')                    
parser.add_argument('--mem_dim', default=128, type=int,
                    help='hidden state dim')
parser.add_argument('--att_dim', default=128, type=int,
                    help='attention dim')    
parser.add_argument('--job_dir', default='saving', type=str,
                    help='job_dir')    
parser.add_argument('--data_dir', default='/home/yfeng23/dataset/activity_net/', type=str,
                    help='dir')               
parser.add_argument('--num_gpus', default=0, type=int,
                    help='number of gpus')
parser.add_argument('--bucket_span', default=30, type=int,
                    help='bucket_span')                    
parser.add_argument('--batch_size', default=128, type=int,
                    help='batch_size')
parser.add_argument('--max_steps', default=1000, type=int,
                    help='max_steps')
parser.add_argument('--weight_decay', default=0.0, type=float,
                    help='weight decay')                    
parser.add_argument('--learning_rate', default=0.001, type=float,
                    help='learning_rate')
parser.add_argument('--max_gradient_norm', default=5.0, type=float,
                    help='max_gradient_norm')    
parser.add_argument('--save_summary_steps', default=10, type=int,
                    help='save_summary_steps')  
parser.add_argument('--save_checkpoint_steps', default=100, type=int,
                    help='save_checkpoint_steps')  
global args
args = parser.parse_args()
