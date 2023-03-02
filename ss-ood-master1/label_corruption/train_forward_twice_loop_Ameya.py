# -*- coding: utf-8 -*-
import argparse
import os
import time
import math
import json
import torch
from torch.autograd import Variable as V
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms
from models.wrn_with_pen import WideResNet
from models.resnet import resnet20
import numpy as np
from load_corrupted_data import CIFAR10, CIFAR100
from PIL import Image
import torch.nn as nn

# Line 283 is where I stopped
# note: nosgdr, schedule, and epochs are highly related settings
# Line 155 can be changed to try training on resnet20
# Setting default corruption probability to 0.8, can deal with that later

parser = argparse.ArgumentParser(description='Trains WideResNet on CIFAR',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# Dataset arguments
parser.add_argument('--data_path', type=str, default='./data/cifarpy',
    help='Root for the Cifar dataset.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
    help='Choose between CIFAR-10, CIFAR-100.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.') # Changed to 2 to see whole thing, default 100
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--gold_fraction', '-gf', type=float, default=0, help='What fraction of the data should be trusted?')
parser.add_argument('--corruption_prob', '-cprob', type=float, default=0.8, help='The label corruption probability.')
parser.add_argument('--corruption_type', '-ctype', type=str, default='unif', help='Type of corruption ("unif" or "flip").')
parser.add_argument('--no_ss', action='store_true', help='Turns off self-supervised auxiliary objective(s)')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--test_bs', type=int, default=128)
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs. Use when SGDR is off.')
parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')
# Checkpoints
parser.add_argument('--save', '-s', type=str, default='./', help='Folder to save checkpoints.')
parser.add_argument('--load', '-l', type=str, default='', help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
#  - need to change this to be resnet 20? Could alternatively use wrn and just decrease number of layers/widen factor
#parser.add_argument('--layers', default=40, type=int, help='total number of layers (default: 28)')
#parser.add_argument('--widen-factor', default=2, type=int, help='widen factor (default: 10)')
parser.add_argument('--droprate', default=0.1, type=float, help='dropout probability (default: 0.0)')
parser.add_argument('--nonlinearit\y', type=str, default='relu', help='Nonlinearity (relu, elu, gelu).')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=2, help='Pre-fetching threads.')
# i/o
parser.add_argument('--log', type=str, default='./', help='Log folder.')
# random seed
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()


np.random.seed(args.seed)

cudnn.benchmark = True  # fire on all cylinders

import socket
print()
print("This is on machine:", socket.gethostname())
print()
print(args)
print()

# Init logger
if not os.path.isdir(args.log):
    os.makedirs(args.log)
log = open(os.path.join(args.log, args.dataset + '_log.txt'), 'w')
state = {k: v for k, v in args._get_kwargs()}
log.write(json.dumps(state) + '\n')

# Init dataset
if not os.path.isdir(args.data_path):
    os.makedirs(args.data_path)

train_transform = transforms.Compose(
    [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor()])
test_transform = transforms.Compose(
    [transforms.ToTensor()])

if args.dataset == 'cifar10':
    train_data = CIFAR10(
        args.data_path, True, False, 0, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=None)
    train_data_deterministic = CIFAR10(
        args.data_path, True, False, 0, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=None)
    test_data = CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 10

elif args.dataset == 'cifar100':
    train_data = CIFAR100(
        args.data_path, True, False, 0, args.corruption_prob, args.corruption_type,
        transform=train_transform, download=True, shuffle_indices=None)
    train_data_deterministic = CIFAR100(
        args.data_path, True, False, 0, args.corruption_prob, args.corruption_type,
        transform=test_transform, download=True, shuffle_indices=None)
    test_data = CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
    num_classes = 100


class Dataset(object):
    def __getitem__(self, index):
        raise NotImplementedError
    def __len__(self):
        raise NotImplementedError


class TensorDataset(Dataset):
    def __init__(self, data_tensor, target_tensor, transform):
        # assert data_tensor.size(0) == target_tensor.size(0)
        self.data_tensor = data_tensor
        self.target_tensor = target_tensor
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data_tensor[index], self.target_tensor[index]

        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, target

    def __len__(self):
        return self.target_tensor.size()[0]

train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

train_loader_deterministic = torch.utils.data.DataLoader(
    train_data_deterministic,
    batch_size=args.batch_size, shuffle=True,
    num_workers=args.prefetch, pin_memory=True)

test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.test_bs, shuffle=False,
                                          num_workers=args.prefetch, pin_memory=True)



# Create 
# Think I should be able to swap this straight for Resnet 20, let me check it out
#net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
net = resnet20()
net.rot_pred = nn.Linear(64, 4) # change first entry to 64 for resnet20 layer, 128 for wide resnet

net2 = resnet20()
net2.rot_pred = nn.Linear(64, 4) # change first entry to 64 for resnet20 layer, 128 for wide resnet

start_epoch = 0

# Restore model if desired
if args.load != '':
    for i in range(1000 - 1, -1, -1):
        model_name = os.path.join(args.load, args.dataset + '_' + 'wrn' +
                                  '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            net.load_state_dict(torch.load(model_name))
            print('Model restored! Epoch:', i)
            start_epoch = i + 1
            break
    if start_epoch == 0:
        assert False, "could not resume"

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
    net2 = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

if args.ngpu > 0:
    
    net.cuda()
    net2.cuda()
    torch.cuda.manual_seed(1)

cudnn.benchmark = True  # fire on all cylinders

optimizer = torch.optim.SGD(
    list(net.parameters()) +list(net2.parameters()), state['learning_rate'], momentum=state['momentum'],
    weight_decay=state['decay'], nesterov=True)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))

# Might need to do scheduler for the second model too?
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        args.epochs * len(train_loader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / args.learning_rate))


# Might need to do scheduler for the second model too?

# Need a for loop in the training
#  for i in range(self.model_num):
#            # build models
#            model = resnet20()
#            if self.use_gpu:
#                model.cuda()           
#            self.models.append(model)

# Define KL loss for later
#loss_kl = nn.KLDivLoss(reduction='batchmean')
# train function (forward, backward, update)
# This performs a training step, need it to call both models in here
def train(no_correction=True, C_hat_transpose=None, C_hat_transpose2=None,T = 0.2, scheduler=scheduler):
    net.train()     # enter train mode # what does that mean?
    net2.train() 
    loss_avg = 0.0
    loss_avg2 = 0.0

    scaler = torch.cuda.amp.GradScaler()
    for bx, by in train_loader:
        bx, by = bx.cuda(), by.cuda()

        optimizer.zero_grad()
        
        #for model in models: (indent stuff below)
        # forward
        with torch.autocast(device_type = 'cuda', dtype = torch.float16):
            logits, _ = net(bx * 2 - 1) # change 'net' to 'models'? could also leave it as kind of scrappy code and manually write 'net' and 'nets'
            logits2, _ = net2(bx * 2 - 1)

            # backward
            scheduler.step()
         
            if no_correction:
                loss = F.cross_entropy(logits, by)
                loss2 = F.cross_entropy(logits2, by)
            else:
                pre1 = C_hat_transpose[torch.cuda.LongTensor(by.data)] 
                pre2 = torch.mul(F.softmax(logits), pre1) + 1e-6
                loss = -(torch.log(pre2.sum(1))).mean() 

                pre12 = C_hat_transpose2[torch.cuda.LongTensor(by.data)]
                pre22 = torch.mul(F.softmax(logits2), pre12) + 1e-6
                loss2 = -(torch.log(pre22.sum(1))).mean()
            
            if not args.no_ss:
                curr_batch_size = bx.size(0)
                by_prime = torch.cat((torch.zeros(bx.size(0)), torch.ones(bx.size(0)),
                                  2*torch.ones(bx.size(0)), 3*torch.ones(bx.size(0))), 0).long()
            #bx = bx.cpu().numpy()
            #bx = np.concatenate((bx, np.rot90(bx, 1, axes=(2, 3)),
            #                     np.rot90(bx, 2, axes=(2, 3)), np.rot90(bx, 3, axes=(2, 3))), 0)
            #bx = torch.FloatTensor(bx)
                bx = torch.cat((bx, torch.rot90(bx,1, dims=[2,3]),torch.rot90(bx, 2, dims=[2, 3]), torch.rot90(bx, 3, dims=[2, 3])), 0)
                bx, by_prime = bx.cuda(), by_prime.cuda()

                _, pen = net(bx * 2 - 1)
                _, pen2 = net2(bx * 2 - 1)

                #pen = pen.cuda()
                #pen2 = pen2.cuda()


                loss += 0.5 * F.cross_entropy(net.rot_pred(pen), by_prime)
                loss2 += 0.5 * F.cross_entropy(net2.rot_pred(pen2), by_prime)


            # KL loss, set to 0 for now
            # KL loss, set to 0 for now, this part just gets ignored? /0 gives no error but also doesn't train
            # Should ask it to state KL loss over time, compare to other parts
            
                #DML_loss = TT*F.cross_entropy(net.rot_pred(pen), F.softmax(net2.rot_pred(pen2),dim=1))
                #loss += DML_loss 
            
            #kl_loss2 += 0.01*loss_kl(F.log_softmax(net2.rot_pred(pen2),dim = 1), F.softmax(net.rot_pred(pen),dim = 1))
                #DML_loss2 = TT*F.cross_entropy(net2.rot_pred(pen2), F.softmax(net.rot_pred(pen),dim=1))
                #loss2 += DML_loss2 

                A = F.softmax(net2.rot_pred(pen2),dim=1).clone().detach().requires_grad_(False)
                DML_loss = T*F.cross_entropy(net.rot_pred(pen), A)
                loss += DML_loss 
                B = F.softmax(net.rot_pred(pen),dim=1).clone().detach().requires_grad_(False)
            #kl_loss2 += 0.01*loss_kl(F.log_softmax(net2.rot_pred(pen2),dim = 1), F.softmax(net.rot_pred(pen),dim = 1))
                DML_loss2 = T*F.cross_entropy(net2.rot_pred(pen2), B)
                loss2 += DML_loss2 


            loss3 = loss + loss2
            scaler.scale(loss3).backward()

            scaler.step(optimizer) # Ignore warning, this is the right order        
            
            scaler.update()
        # exponential moving average
        loss_avg = loss_avg * 0.95 + loss.item() * 0.05
        loss_avg2 = loss_avg2 * 0.95 + loss2.item() * 0.05


    state['train_loss'] = loss_avg
    state['train_loss2'] = loss_avg2
    state['Avg_DML_Loss'] = 0.5*(DML_loss.item() + DML_loss2.item())
# Now to TEST

# test function (forward only)
def test():
    torch.set_grad_enabled(False)
    net.eval()
    net2.eval()
    loss_avg = 0.0
    correct = 0

    loss_avg2 = 0.0
    correct2 = 0

    with torch.autocast(device_type = 'cuda', dtype = torch.float16):
        for bx, by in test_loader:
            bx, by = bx.cuda(), by.cuda()

        # forward
            logits, pen = net(bx * 2 - 1) # Does the -1 mean look at the penultimate layer? Or is it just the ','
            loss = F.cross_entropy(logits, by)

            logits2, pen2 = net2(bx * 2 - 1)
            loss2 = F.cross_entropy(logits2, by)

        
        # accuracy
            pred = logits.data.max(1)[1]
            correct += pred.eq(by.data).sum().item()

            pred2 = logits2.data.max(1)[1]
            correct2 += pred2.eq(by.data).sum().item()


        # test loss average
            loss_avg += loss.item()
            loss_avg2 += loss2.item()

        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)
        state['test_loss2'] = loss_avg2 / len(test_loader)
        state['test_accuracy2'] = correct2 / len(test_loader.dataset)

    torch.set_grad_enabled(True)


# Main loop
n = 10
performance = np.empty((n, 0)).tolist()
for j in range(n):
    # Create 
# Think I should be able to swap this straight for Resnet 20, let me check it out
#net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)
    net = resnet20()
    net.rot_pred = nn.Linear(64, 4) # change first entry to 64 for resnet20 layer, 128 for wide resnet

    net2 = resnet20()
    net2.rot_pred = nn.Linear(64, 4) # change first entry to 64 for resnet20 layer, 128 for wide resnet

    start_epoch = 0

    # Restore model if desired
    
    if args.ngpu > 1:
        net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))
        net2 = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    if args.ngpu > 0:
    
        net.cuda()
        net2.cuda()
        torch.cuda.manual_seed(1)

        cudnn.benchmark = True  # fire on all cylinders

        optimizer = torch.optim.SGD(
            list(net.parameters()) +list(net2.parameters()), state['learning_rate'], momentum=state['momentum'],
            weight_decay=state['decay'], nesterov=True)

        def cosine_annealing(step, total_steps, lr_max, lr_min):
            return lr_min + (lr_max - lr_min) * 0.5 * (
                    1 + np.cos(step / total_steps * np.pi))

    # Might need to do scheduler for the second model too?
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                args.epochs * len(train_loader),
                1,  # since lr_lambda computes multiplicative factor
                1e-6 / args.learning_rate))
    
    for epoch in range(args.epochs):
        state['epoch'] = epoch

        begin_epoch = time.time()
        train(scheduler=scheduler, T = 0.05*j)
        print('Epoch', epoch, '| Time Spent:', round(time.time() - begin_epoch, 2))

        if (epoch%1==0):
            test()
        performance[j] = np.append(performance[j],state['test_accuracy'])
        

        log.write('%s\n' % json.dumps(state))
        log.flush()
        print(state)
        
    print(performance)

log.close()
