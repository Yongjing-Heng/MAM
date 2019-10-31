import torch as t
import torchvision as tv
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.utils.data
import datetime
import argparse

from basic_layers import ResidualBlock
from RandomDisruption import RandomDisrupt
from Mixed_Attention_Module import MixedAttentionBlock


WORKERS = 4
PARAS_FN = 'cifar10_mixed_attention_module.pkl'
ROOT = 'E:/cifar10'
loss_func = nn.CrossEntropyLoss()
best_acc = 0
global_train_error = []
global_test_error = []




class MixedAttentionModel(nn.Module):
    def __init__(self):
        super(MixedAttentionModel, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.residual_block1 = MixedAttentionBlock(32, 128)  # 32*32
        self.residual_block2 = MixedAttentionBlock(128, 256, 2)  # 16*16
        self.residual_block3 = MixedAttentionBlock(256, 512, 2)  # 4*4
        self.residual_block4 = MixedAttentionBlock(512, 1024)  # 8*8
        self.residual_block5 = MixedAttentionBlock(1024, 1024)  # 8*8
        self.residual_block6 = MixedAttentionBlock(1024, 1024)  # 8*8
        self.mpool2 = nn.Sequential(
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(kernel_size=8)
        )
        self.fc = nn.Linear(1024,10)

    def forward(self, x):
        out = self.conv1(x)
        out = self.residual_block1(out)
        out = self.residual_block2(out)
        out = self.residual_block3(out)
        out = self.residual_block4(out)
        out = self.residual_block5(out)
        out = self.residual_block6(out)
        out = self.mpool2(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out




def net_train(net, train_data_load, optimizer, epoch, log_interval):
    net.train()
    begin = datetime.datetime.now()
    total = len(train_data_load.dataset)
    train_loss = 0
    ok = 0

    for i, data in enumerate(train_data_load, 0):
        img, label = data
        img, label = img.cuda(), label.cuda()
        optimizer.zero_grad()
        outs = net(img)
        loss = loss_func(outs, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, predicted = t.max(outs.data, 1)
        ok += (predicted == label).sum()

        if (i + 1) % log_interval == 0:
            loss_mean = train_loss / (i + 1)
            traind_total = (i + 1) * len(label)
            acc = 100. * ok / traind_total
            progress = 100. * traind_total / total
            print('Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}  Acc: {:.6f}'.format(
                epoch, traind_total, total, progress, loss_mean, acc))

    error = (100 - acc).cpu().numpy()
    global_train_error.append(float("%.2f" % error))

    end = datetime.datetime.now()
    print('time spend: ', end - begin)



def net_test(net, test_data_load, epoch):
    net.eval()
    ok = 0
    test_loss = 0
    for i, data in enumerate(test_data_load):
        img, label = data
        img, label = img.cuda(), label.cuda()
        outs = net(img)
        loss = loss_func(outs, label)
        test_loss += loss.item()
        _, pre = t.max(outs.data, 1)
        ok += (pre == label).sum()

    acc = ok.item() * 100. / (len(test_data_load.dataset))
    print('EPOCH:{}, ACC:{}\n'.format(epoch, acc))
    global_test_error.append(round((100 - acc), 2))
    global best_acc
    if acc > best_acc:
        best_acc = acc




def show_error_curv():
    train_x = list(range(len(global_train_error)))
    train_y = global_train_error
    test_x = list(range(len(global_test_error)))
    test_y = global_test_error

    plt.title('CIFAR10 MAM ERROR')
    plt.plot(train_x, train_y, color='green', label='training error')
    plt.plot(test_x, test_y, color='red', label='testing error')
    plt.legend()
    plt.xlabel('iterations')
    plt.ylabel('error %')
    plt.show()


def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFA10 Mixed Attention Module')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--test-batch-size', type=int, default=20, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=200, metavar='N',
                        help='number of epochs to train (default: 200)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: 10)')
    parser.add_argument('--no-train', action='store_true', default=False,
                        help='If train the Model')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    args = parser.parse_args()

    transform_train = tv.transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop((32, 32), padding=4),

#        RandomDisrupt(2),

        transforms.ToTensor(),
    ])

    transform_test = tv.transforms.Compose([
        transforms.ToTensor(),
    ])

    train_data = tv.datasets.CIFAR10(root=ROOT, train=True, download=True, transform=transform_train)
    test_data = tv.datasets.CIFAR10(root=ROOT, train=False, download=False, transform=transform_test)
    train_load = t.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=WORKERS)
    test_load = t.utils.data.DataLoader(test_data, batch_size=args.test_batch_size, shuffle=False, num_workers=WORKERS)

    net = MixedAttentionModel()
    net = net.cuda()
    print(net)

#    net = nn.DataParallel(net)
#    cudnn.benchmark = True

    if args.no_train:
        net.load_state_dict(t.load(PARAS_FN))
        net_test(net, test_load, 0)
        return

    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum, nesterov=True, weight_decay=0.0001)
    start_time = datetime.datetime.now()

    for epoch in range(1, args.epochs + 1):
        net_train(net, train_load, optimizer, epoch, args.log_interval)
        net_test(net, test_load, epoch)

        if (epoch+1) == 60:
            lr = args.lr / 10
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])

        if (epoch + 1) == 120:
            lr = args.lr / 100
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])

        if (epoch + 1) == 180:
            lr = args.lr / 1000
            print('reset learning rate to:', lr)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print(param_group['lr'])

    end_time = datetime.datetime.now()

    print("global train error: \n", global_train_error)
    print()
    print("global test error: \n", global_test_error)

    global best_acc
    print('CIFAR10 Mixed Attention Network: EPOCH:{}, BATCH_SZ:{}, LR:{}, ACC:{}'.format(args.epochs, args.batch_size, args.lr, best_acc))
    print('train spend time: ', end_time - start_time)

    show_error_curv()

    if args.save_model:
        t.save(net.state_dict(), PARAS_FN)

if __name__ == '__main__':
    main()
