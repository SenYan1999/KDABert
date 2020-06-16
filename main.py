import torch
import os

from args import args
from model import KDABert
from utils import KDATrainer, init_logger, PretrainedDataset
from apex import amp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

def prepare():
    train = PretrainedDataset(os.path.join(args.data_dir, args.train_h5), args.num, args.max_pred_len)
    test = PretrainedDataset(os.path.join(args.data_dir, args.test_h5), args.num, args.max_pred_len)

    torch.save(train, os.path.join(args.data_dir, args.train_dataset))
    torch.save(test, os.path.join(args.data_dir, args.test_dataset))

def train(logger):
    # distributed
    logger.info('Let\'s use {} gpus.'.format(torch.cuda.device_count()))
    torch.distributed.init_process_group(backend='nccl')
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)

    # prepare dataset and dataloader
    train_dataset = torch.load(os.path.join(args.data_dir, args.train_dataset))
    eval_dataset = torch.load(os.path.join(args.data_dir, args.test_dataset))
    train_dataloader, eval_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=DistributedSampler(train_dataset)), \
                                        DataLoader(eval_dataset, batch_size=args.batch_size, sampler=DistributedSampler(eval_dataset))

    # define model and optimzier
    model_config = {'hidden_size': args.hidden_size,
                    'hidden_dropout_prob': args.hidden_dropout_prob,
                    'num_hidden_layers': args.num_hidden_layers,
                    'num_attention_heads': args.num_attention_heads}

    model = KDABert(config=model_config, bert_name=args.bert_name, batch_size=args.batch_size, device=device)
    s_optimizer = torch.optim.Adam(model.student.parameters(), lr=args.s_lr)
    d_optimizer = torch.optim.Adam(model.discriminator.parameters(), lr=args.d_lr)

    # convert model to cuda
    model = model.to(device)

    # use apex
    if args.fp16:
        amp.initialize(model, [s_optimizer, d_optimizer], opt_level="O2", loss_scale="dynamic", cast_model_outputs=torch.float16)
    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

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
    if args.do_train:
        train(logger)
    if not(args.do_prepare or args.do_train):
        print('Please choose a running mode.')