import argparse

parser = argparse.ArgumentParser()

# main mode
parser.add_argument('--do_prepare', action='store_true')
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--fp16', action='store_true')
parser.add_argument('--distributed', action='store_true')

# distributed
parser.add_argument('--local_rank', type=int, default=0)

# data prepare
parser.add_argument('--data_dir', type=str, default='glue_data')
parser.add_argument('--task', type=str, default='CoLA')
parser.add_argument('--max_len', type=int, default=150)

# model
parser.add_argument('--bert_name', type=str, default='bert-base-uncased')
parser.add_argument('--num_hidden_layers', type=int, default=4)
parser.add_argument('--num_attention_heads', type=int, default=4)
parser.add_argument('--hidden_size', type=int, default=768)
parser.add_argument('--hidden_dropout_prob', type=float, default=0.1)

# train
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_epoch', type=int, default=6)
parser.add_argument('--s_lr', type=float, default=2e-5)
parser.add_argument('--d_lr', type=float, default=2e-5)
parser.add_argument('--drop_out', type=float, default=0.1)
parser.add_argument('--print_interval', type=int, default=100)

# save & log
parser.add_argument('--log_file', type=str, default='log/log.log')
parser.add_argument('--save_dir', type=str, default='save_models/')

# parse args
args = parser.parse_args()