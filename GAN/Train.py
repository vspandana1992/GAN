""" SpandanaVemulapalli : Generative Adversarial Network """

import torch,argparse,os
from torch import nn
from torch.autograd import Variable
from LIB.GAN import Generative,Discriminative
from LIB.DataLoader import DL
from  torchvision import utils
import torch.optim as optim

""" Implementing DCGAN https://arxiv.org/pdf/1511.06434v2.pdf """
if not os.path.isdir("./results"):
    os.mkdir("./results")

class _train(object):
    def __call__(self,args):
        dataloader = torch.utils.data.DataLoader(DL('cifar10','./Data'),batch_size=args.bsz, shuffle=True)
        loss_fn    = nn.BCELoss()
        optimizerD = optim.Adam(Discriminative.parameters(), lr = args.lr, betas = (0.5, 0.999))
        optimizerG = optim.Adam(Generative.parameters(), lr = args.lr, betas = (0.5, 0.999))
        Generative.train()
        for epoch in range(args.epochs):

            for i, data in enumerate(dataloader):

                Discriminative.zero_grad()
                original  = data
                input       = Variable(original)
                target      = Variable(torch.ones(input.size()[0]))
                output      = Discriminative(input)
                D_loss_Oinp = loss_fn(output, target)
                random_input    = Variable(torch.randn(input.size()[0], 100, 1, 1))
                Ginp            = Generative(random_input)
                target          = Variable(torch.zeros(input.size()[0]))
                output          = Discriminative(Ginp.detach())
                D_loss_Ginp     = loss_fn(output, target)

                D_loss          = D_loss_Oinp + D_loss_Ginp
                D_loss.backward()
                optimizerD.step()

                Generative.zero_grad()
                target = Variable(torch.ones(input.size()[0]))
                output = Discriminative(Ginp)
                G_loss = loss_fn(output, target)
                G_loss.backward()
                optimizerG.step()


                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f' % (epoch, args.epochs, i, len(dataloader), D_loss.data[0], G_loss.data[0]))
            utils.save_image(original, '%s/original.png' % "./results", normalize = True)
            Ginp = Generative(random_input)
            utils.save_image(Ginp.data, '%s/generated_epoch_%03d.png' % ("./results", epoch), normalize = True)
        torch.save(Generative.state_dict(), 'model.t7')
Train = _train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='GAN simple illustration with Pytorch')
    parser.add_argument('-bsz', type=int, default=128, metavar='batch-size',
                        help='input batch size for training (default: 128)')
    parser.add_argument('-epochs', type=int, default=100, metavar='epochs',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('-lr', type=float, default=0.0002, metavar='Learning Rate',
                        help='learning rate (default: 0.0002)')
    args = parser.parse_args()
    Train(args)
