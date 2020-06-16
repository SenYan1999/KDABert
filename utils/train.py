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

    def train_epoch(self, epoch, train_data, print_interval=250):
        self.model.train()
        pbar = tqdm(total=len(train_data))
        for iter, batch in enumerate(train_data):
            input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels = \
                map(lambda x: x.to(self.device), batch)
            input = {'input_ids': input_ids, 'attention_mask': attention_mask,
                     'token_type_ids': token_type_ids}

            acc = {'nsp': None, 'mlm': None}
            s_loss = {'nsp': None, 'mlm': None}
            d_loss = {'nsp': None, 'mlm': None}
            for task in ['nsp', 'mlm']:
                self.s_optimizer.zero_grad()
                self.d_optimizer.zero_grad()

                student_logits, (student_last_hidden, teacher_last_hidden) = self.model(input, task)

                # prepare logits and label for compute logits
                if task == 'nsp':
                    student_logits = student_logits
                    label = next_sentence_labels
                elif task == 'mlm':
                    student_logits = student_logits.reshape(-1, student_logits.shape[-1])
                    label = masked_lm_labels.reshape(-1)
                else:
                    raise Exception('Error Task!')

                # training student model
                try:
                    student_loss = self.model.get_loss_for_training_student(student_logits, student_last_hidden, label)
                except:
                    student_loss = self.model.module.get_loss_for_training_student(student_logits, student_last_hidden, label)
                if self.fp16:
                    with amp.scale_loss(student_loss, self.s_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    student_loss.backward()
                self.s_optimizer.step()

                # training discriminator model
                try:
                    discriminator_loss = self.model.get_loss_for_training_discriminator(student_last_hidden, teacher_last_hidden)
                except:
                    discriminator_loss = self.model.module.get_loss_for_training_discriminator(student_last_hidden, teacher_last_hidden)
                if self.fp16:
                    with amp.scale_loss(discriminator_loss, self.s_optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    discriminator_loss.backward()
                self.d_optimizer.step()

                pred = torch.argmax(student_logits, dim=-1)
                valid_label = torch.where(label != -1)
                acc[task] = (pred[valid_label] == label[valid_label]).sum().float() / label[valid_label].shape[0]
                s_loss[task] = student_loss
                d_loss[task] = discriminator_loss

            stu_loss = (s_loss['mlm'].item() + s_loss['nsp'].item()) / 2
            dis_loss = (d_loss['mlm'].item() + d_loss['nsp'].item()) / 2
            pbar.set_description('Epoch: %2d | StuLoss: %1.3f | DisLoss: % 1.3f | MLM Accuracy: %1.3f | NSP Accuracy: %1.3f' % \
                                 (epoch, stu_loss, dis_loss, acc['mlm'].item(), acc['nsp'].item()))
            pbar.update(1)
            if iter % print_interval == 0:
                self.logger.info('Epoch: %2d | StuLoss: %1.3f | DisLoss: % 1.3f | MLM Accuracy: %1.3f | NSP Accuracy: %1.3f' % \
                                 (epoch, stu_loss, dis_loss, acc['mlm'].item(), acc['nsp'].item()))

    def evaluate_epoch(self, epoch, dev_data, print_interval=250):
        self.model.eval()
        pbar = tqdm(total=len(dev_data))
        global_acc = {'nsp': [], 'mlm': []}
        global_s_loss = {'nsp': [], 'mlm': []}
        global_d_loss = {'nsp': [], 'mlm': []}
        with torch.no_grad():
            for iter, batch in enumerate(dev_data):
                input_ids, attention_mask, token_type_ids, masked_lm_labels, next_sentence_labels = \
                    map(lambda x: x.to(self.device), batch)
                input = {'input_ids': input_ids, 'attention_mask': attention_mask,
                         'token_type_ids': token_type_ids}

                acc = {'nsp': None, 'mlm': None}
                s_loss = {'nsp': None, 'mlm': None}
                d_loss = {'nsp': None, 'mlm': None}
                for task in ['nsp', 'mlm']:
                    student_logits, (student_last_hidden, teacher_last_hidden) = self.model(input, task)

                    # prepare logits and label for compute logits
                    if task == 'nsp':
                        student_logits = student_logits
                        label = next_sentence_labels
                    elif task == 'mlm':
                        student_logits = student_logits.reshape(-1, student_logits.shape[-1])
                        label = masked_lm_labels.reshape(-1)
                    else:
                        raise Exception('Error Task!')

                    # training student model
                    try:
                        student_loss = self.model.get_loss_for_training_student(student_logits, student_last_hidden, label)
                    except:
                        student_loss = self.model.module.get_loss_for_training_student(student_logits, student_last_hidden, label)

                    # training discriminator model
                    try:
                        discriminator_loss = self.model.get_loss_for_training_discriminator(student_last_hidden, teacher_last_hidden)
                    except:
                        discriminator_loss = self.model.module.get_loss_for_training_discriminator(student_last_hidden, teacher_last_hidden)

                    pred = torch.argmax(student_logits, dim=-1)
                    valid_label = torch.where(label != -1)
                    acc[task] = (pred[valid_label] == label[valid_label]).sum().float() / label[valid_label].shape[0]
                    s_loss[task] = student_loss
                    d_loss[task] = discriminator_loss

                    global_acc[task].append(acc[task].item())
                    global_s_loss[task].append(s_loss[task].item())
                    global_d_loss[task].append(d_loss[task].item())

                stu_loss = (s_loss['mlm'].item() + s_loss['nsp'].item()) / 2
                dis_loss = (d_loss['mlm'].item() + d_loss['nsp'].item()) / 2
                pbar.set_description('Epoch: %2d | StuLoss: %1.3f | DisLoss: % 1.3f | MLM Accuracy: %1.3f | NSP Accuracy: %1.3f' % \
                                     (epoch, stu_loss, dis_loss, acc['mlm'].item(), acc['nsp'].item()))
                pbar.update(1)
                if iter % print_interval == 0:
                    self.logger.info('Epoch: %2d | StuLoss: %1.3f | DisLoss: % 1.3f | MLM Accuracy: %1.3f | NSP Accuracy: %1.3f' % \
                                     (epoch, stu_loss, dis_loss, acc['mlm'].item(), acc['nsp'].item()))

        mean_acc = (np.mean(global_acc['nsp']) + np.mean(global_acc['mlm'])) / 2
        self.logger.info('')
        self.logger.info('-'*70)
        self.logger.info('Epoch %2d: Evaluate Results' % (epoch))
        self.logger.info('NSP:')
        self.logger.info('Student Loss: %1.3f | Discriminator Loss: %1.3f | Accuracy: %1.3f' %
                         (np.mean(global_s_loss['nsp']), np.mean(global_d_loss['nsp']), np.mean(global_acc['nsp'])))
        self.logger.info('MLM:')
        self.logger.info('Student Loss: %1.3f | Discriminator Loss: %1.3f | Accuracy: %1.3f' %
                         (np.mean(global_s_loss['mlm']), np.mean(global_d_loss['mlm']), np.mean(global_acc['mlm'])))
        self.logger.info('-'*70)
        self.logger.info('')

        return mean_acc

    def save_state(self, name):
        state_dict = {
            'model': self.model.state_dict(),
            'd_optimizer': self.d_optimizer.state_dict(),
            's_optimizer': self.s_optimizer.state_dict(),
        }
        try:
            student_model = self.model.student
        except:
            student_model = self.model.module.student

        torch.save(state_dict, os.path.join(self.save_dir, name))
        torch.save(student_model, os.path.join(self.save_dir, 'student_' + name))
