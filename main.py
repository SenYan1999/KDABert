import torch

from args import args
from model import KDABert
from utils import KDATrainer, init_logger
from apex import amp
from torch.utils.data import DataLoader

def prepare():
    pass

def train(logger):
    # prepare dataset and dataloader
    dataset = torch.load(args.train_data)
    train_size = int(len(dataset) * (1 - args.eval_rate))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True), \
                                        DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=True)

    # define model and optimzier
    model_config = {'hidden_size': args.hidden_size,
                    'hidden_dropout_prob': args.hidden_dropout_prob,
                    'num_hidden_layers': args.num_hidden_layers,
                    'num_attention_heads': args.num_attention_heads}

    model = KDABert(config=model_config, bert_name=args.bert_name)
    s_optimizer = torch.optim.Adam(model.student.parameters(), lr=args.s_lr)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=args.d_lr)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # convert model to cuda
    model = model.cuda()

    # use apex
    if args.fp16:
        amp.initialize(model, [s_optimizer, d_optimizer], opt_level='O1')

    # define trainer and begin training
    train_config = {'num_epoch': args.num_epoch,
                    'fp16': args.fp16,
                    'device': device,
                    'save_dir': args.save_dir,
                    'save_name': 'model.pt'}
    trainer = KDATrainer(model, s_optimizer, d_optimizer, train_dataloader, eval_dataloader, logger, train_config)
    trainer.train()

if __name__ == "__main__":
    logger = init_logger(args.log_file)
    if args.do_prepare:
        prepare()
    elif args.do_train:
        train(logger)
    else:
        print('Please choose a running mode.')