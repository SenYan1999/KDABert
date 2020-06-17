import torch
import torch.nn.functional as F
import numpy as np
import os
import time

from tqdm import tqdm
from sklearn.metrics import matthews_corrcoef, accuracy_score
from apex import amp

class Trainer:
    def __init__(self, train_dataloader, dev_dataloader, model, optimizer, scheduler, task, logger, fp16, device):
        self.train_data = train_dataloader
        self.dev_data = dev_dataloader
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.device = device
        self.task = task
        self.logger = logger
        self.fp16 = fp16

        self.metrics = {'CoLA': 'MCC', 'QNLI': 'ACC', 'SST-2': 'ACC', 'MNLI': 'ACC', 'WNLI': 'ACC', 'QQP': 'acc',\
                        'MRPC': 'ACC', 'RTE': 'ACC'}

    def calculate_result(self, pred, truth):
        pred = torch.argmax(pred, dim=-1).detach().cpu().numpy().astype(np.float)
        truth = truth.detach().cpu().numpy().astype(np.float)

        if self.task == 'CoLA':
            result = matthews_corrcoef(truth, pred)
        elif self.task in ['QNLI', 'SST-2', 'WNLI', 'QQP', 'MNLI', 'MRPC', 'RTE']:
            result = accuracy_score(truth, pred)
        else:
            raise('Task error!')

        return result

    def train_epoch(self, epoch, print_interval):
        self.logger.info('Epoch: %2d: Training Model...' % epoch)
        pbar = tqdm(total = len(self.train_data))
        self.model.train()

        losses, accs = [], []
        for batch in self.train_data:
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)
            kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

            out = self.model(kwargs)
            loss = F.cross_entropy(out, labels)
            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()

            losses.append(loss.item())
            acc = self.calculate_result(out, labels)
            accs.append(acc)

            pbar.set_description('Epoch: %2d | LOSS: %2.3f | %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))
            pbar.update(1)

        pbar.close()
        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def evaluate_epoch(self, epoch):
        self.logger.info('Epoch: %2d: Evaluating Model...' % epoch)
        self.model.eval()

        losses, precise, recall, f1s, accs = [], [], [], [], []
        for batch in self.dev_data:
            input_ids, attention_mask, token_type_ids, labels = map(lambda i: i.to(self.device), batch)
            kwargs = {'input_ids': input_ids, 'attention_mask': attention_mask, 'token_type_ids': token_type_ids}

            with torch.no_grad():
                out = self.model(kwargs)
            loss = F.cross_entropy(out, labels)

            acc = self.calculate_result(out, labels)
            losses.append(loss.item())
            accs.append(acc)

        self.logger.info('Epoch: %2d | LOSS: %2.3f %s: %1.3f' % (epoch, np.mean(losses), self.metrics[self.task], np.mean(accs)))

    def train(self, num_epoch, save_path, print_interval):
        for epoch in range(num_epoch):
            self.train_epoch(epoch, print_interval)
            self.evaluate_epoch(epoch)

            # save state dict
            path = os.path.join(save_path, 'state_%d_epoch.pt' % epoch)
            self.save_dict(path)
            self.logger.info('')

    def save_dict(self, save_path):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state_dict, save_path)

    def load_dict(self, path):
        state_dict = torch.load(path)

        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
