import argparse
import numpy as np

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch import nn
from torch.utils.data import DataLoader

from data_loader import MelDataset
#from models.modern_cnn import cnn_model
from models.resnet import ResNet18
from metrics import compute_confusion_matrix
from utils import create_folder, monte_carlo_dropout

USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

MODEL_PATH = '../pt/classifier/'


def train(model, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    correct = 0
    train_loss = 0
    for batch_idx, samples in enumerate(train_loader):
        data, target = samples

        data = data.to(DEVICE)
        target = target.to(DEVICE)
        target = torch.squeeze(target)

        output = model(data)

        loss = criterion(output, target)
        train_loss += loss.item()

        prediction = output.max(1, keepdim=True)[1]
        correct += prediction.eq(target.view_as(prediction)).sum().item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Epoch: {:3d}\tBatch Index: {:2d}\tLoss: {:.4f}'.format(epoch, batch_idx, loss.item()))

    train_loss /= len(train_loader.dataset)
    train_acc = 100. * correct / (len(train_loader.dataset))

    return train_loss, train_acc



def evaluate(model, test_loader, mc_dropout=False):
    model.eval()
    if mc_dropout:
        model = monte_carlo_dropout(model)
    criterion = nn.CrossEntropyLoss().to(DEVICE)

    test_loss = 0
    correct = 0
    confusion_matrix = np.zeros([3, 3])
    with torch.no_grad():
        for samples in test_loader:
            data, target = samples

            data = data.to(DEVICE)
            target = target.to(DEVICE)
            target = torch.squeeze(target)

            total_output = []
            for i in range(32):
                output = model(data)
                total_output.append(output.detach().cpu().numpy())
            total_output = np.sum(np.array(total_output), axis=0)
            output = torch.FloatTensor(total_output).to(DEVICE)

            loss = criterion(output, target)
            test_loss += loss.item()

            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(target.view_as(prediction)).sum().item()

            #prediction = torch.clamp(prediction, 0, 2)
            #target = torch.clamp(target, 0, 2)
            confusion_matrix += compute_confusion_matrix(target.detach().cpu().numpy(), prediction.detach().cpu().numpy())
            #print(confusion_matrix)

    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / (len(test_loader.dataset))
    print(confusion_matrix)

    return test_loss, test_acc



def save_model(modelpath, model, optimizer, scheduler):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict()
    }
    torch.save(state, modelpath)

    print('model saved')



def load_model(modelpath, model, optimizer=None, scheduler=None):
    state = torch.load(modelpath, map_location=torch.device(DEVICE))
    model.load_state_dict(state['model'])
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    if scheduler is not None:
        scheduler.load_state_dict(state['scheduler'])

    print('model loaded')



def main(args):
    # train
    if args.mode == 'train':
        train_dataset = MelDataset(mode='train2')
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)

        test_dataset = MelDataset(mode='val2')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=0)
        print(train_dataloader, test_dataloader)

        model = ResNet18()
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        # set optimizer
        optimizer = AdamW(
            [param for param in model.parameters() if param.requires_grad], lr=args.lr)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)

        # load model
        #modelpath = MODEL_PATH + 'classifier-model-23-0.000257-69.1498.pt'
        #load_model(modelpath, model, optimizer=None, scheduler=None)

        prev_acc = 0.0
        for epoch in range(args.epoch):
            # train set
            loss, acc = train(model, train_dataloader, optimizer, epoch)
            # validate set
            val_loss, val_acc = evaluate(model, test_dataloader)

            print('Epoch:{}\tTrain Loss:{:.6f}\tTrain Acc:{:2.4f}'.format(epoch, loss, acc))
            print('Val Loss:{:.6f}\tVal Acc:{:2.4f}'.format(val_loss, val_acc))

            if val_acc > prev_acc:
                prev_acc = val_acc
                create_folder(MODEL_PATH)
                modelpath = MODEL_PATH + 'classifier-model-{:d}-{:.6f}-{:2.4f}.pt'.format(epoch, val_loss, val_acc)
                save_model(modelpath, model, optimizer, scheduler)

            # scheduler update
            scheduler.step()
    # evaluation
    elif args.mode == 'evaluate':
        print('mode is evaluation')

        test_dataset = MelDataset(mode='val2')
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        print(test_dataloader)

        model = ResNet18()
        if torch.cuda.device_count() > 1:
            print('multi gpu used!')
            model = nn.DataParallel(model)
        model = model.to(DEVICE)

        # load model
        modelpath = MODEL_PATH + 'classifier-model-117-0.120010-70.4234.pt'
        load_model(modelpath, model, optimizer=None, scheduler=None)

        test_loss, test_acc = evaluate(model, test_dataloader)
        print('Test Loss:{:.6f}\tTest Acc:{:2.4f}'.format(test_loss, test_acc))
    # prediction
    elif args.mode == 'predict':
        print('mode is prediction')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch_size',
        help='the number of samples in mini-batch',
        default=2048,
        type=int)
    parser.add_argument(
        '--epoch',
        help='the number of training iterations',
        default=1,
        type=int)
    parser.add_argument(
        '--lr',
        help='learning rate',
        default=1e-3,
        type=float)
    parser.add_argument(
        '--shuffle',
        help='True, or False',
        default=True,
        type=bool)
    parser.add_argument(
        '--mode',
        help='train, evaluate, or predict',
        default='train',
        type=str)

    args = parser.parse_args()
    print(args)

    main(args)