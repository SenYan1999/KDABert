import torch
import numpy as np
import os

from tqdm import tqdm
from apex import amp

class Trainer(object):
    def __init__(self, model, optimizer, train_data, dev_data, logger, config):
        self.model = model
        self.optimizer = optimizer
        self.train_data = train_data
        self.dev_data = dev_data
        self.logger = logger

        self.num_epoch = config['num_epoch']
        self.fp16 = config['fp16']
        self.device = config['device']
        self.save_dir = config['save_dir']
        self.save_name = config['save_name']

    def train_epoch(self, epoch, train_data):
        self.model.train()
        pbar = tqdm(total=len(train_data))
        for batch in train_data:
            batch = map(lambda x: x.to(self.device), batch)
            input = {'input_ids': batch[0],
                     'attention_mask': batch[1],
                     'token_type_ids': batch[2],
                     'labels': batch[3]}

            loss, logits = self.model(**input)

            if self.fp16:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.optimzier.zero_grad()

            acc = (torch.argmax(logits, dim=-1) == input['labels']).sum().cpu().float() / logits.shape[0].float()
            pbar.set_description('Epoch: %2d | Loss: %1.3f | Accuracy: %1.3f' % \
                                 (epoch, loss.item(), acc.item()))
            pbar.update(1)

    def evaluate_epoch(self, epoch, dev_data):
        self.model.eval()
        acc_mean = []
        for batch in dev_data:
            batch = map(lambda x: x.to(self.device), batch)
            input = {'input_ids': batch[0],
                     'attention_mask': batch[1],
                     'token_type_ids': batch[2],
                     'labels': batch[3]}

            loss, logits = self.model(**input)

            acc = (torch.argmax(logits, dim=-1) == input['labels']).sum().cpu().float() / logits.shape[0].float()
            acc_mean.append(acc.item())
            self.logger.info('Epoch: %2d | Loss: %1.3f | Accuracy: %1.3f' % \
                             (epoch, loss.item(), acc.item()))

        return np.mean(acc_mean)


    def train(self):
        best_acc = 0
        for epoch in range(self.num_epoch):
            self.train_epoch(epoch, self.train_data)
            epoch_acc = self.evaluate_epoch(epoch, self.dev_data)
            if epoch_acc > best_acc:
                self.save_state(self.save_name)

    def save_state(self, name):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

        torch.save(state_dict, os.path.join(self.save_dir, name))

    def load_state(self, name):
        state_dict = torch.load(os.path.join(self.save_dir, name))
        self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])

class KDATrainer(Trainer):
    def __init__(self, model, s_optimizer, d_optimizer, train_data, dev_data, logger, config):
        ''':param
        model: KDAGan
        s_optimizer: student model's optimzier
        d_optimizer: discriminator's optimizer
        train_data: training dataloader
        dev_data: dev dataloader
        logger: logger file
        config: other config parameters
        '''

        super(KDATrainer, self).__init__(model, None, train_data, dev_data, logger, config)
        self.s_optimizer = s_optimizer
        self.d_optimizer = d_optimizer

    def train_epoch(self, epoch, train_data):
        self.model.train()
        pbar = tqdm(total=len(train_data))
        for batch in train_data:
            batch = map(lambda x: x.to(self.device), batch)
            input = {'input_ids': batch[0],
                     'attention_mask': batch[1],
                     'token_type_ids': batch[2]}
            labels = batch[3]

            student_logits, (student_pool_out, teacher_pool_out) = self.model(input)

            # training student model
            student_loss = self.model.get_loss_for_training_student(student_logits, labels)
            if self.fp16:
                with amp.scale_loss(student_loss, self.s_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                student_loss.backward()
            self.s_optimizer.step()
            self.s_optimizer.zero_grad()

            # training discriminator model
            discriminator_loss=  self.model.get_loss_for_training_discriminator(student_pool_out, teacher_pool_out)
            if self.fp16:
                with amp.scale_loss(discriminator_loss, self.s_optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                discriminator_loss.backward()
            self.d_optimizer.step()
            self.d_optimizer.zero_grad()

            acc = (torch.argmax(student_logits, dim=-1) == labels).sum().cpu().float() / labels.shape[0].float()
            pbar.set_description('Epoch: %2d | StuLoss: %1.3f | DisLoss: % 1.3f | Accuracy: %1.3f' % \
                                 (epoch, student_loss.item(), discriminator_loss.item(), acc.item()))
            pbar.update(1)

    def evaluate_epoch(self, epoch, dev_data):
        self.model.eval()
        acc_mean = []
        for batch in dev_data:
            batch = map(lambda x: x.to(self.device), batch)
            input = {'input_ids': batch[0],
                     'attention_mask': batch[1],
                     'token_type_ids': batch[2]}
            labels = batch[3]

            _, logits = self.model.student(input)
            loss = self.model.student.loss_fn(logits, labels)

            acc = (torch.argmax(logits, dim=-1) == input['labels']).sum().cpu().float() / logits.shape[0].float()
            acc_mean.append(acc.item())
            self.logger.info('Epoch: %2d | Loss: %1.3f | Accuracy: %1.3f' % \
                             (epoch, loss.item(), acc.item()))

        return np.mean(acc_mean)

    def save_state(self, name):
        state_dict = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        student_model = self.model.student

        torch.save(state_dict, os.path.join(self.save_dir, name))
        torch.save(student_model, os.path.join(self.save_dir, 'student_' + name))
