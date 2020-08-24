import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self):
        super(Generator, self).__init__()
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(in_channels = 100, out_channels =512, kernel_size = 4, stride=1, padding=0, bias = False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(in_channels = 512, out_channels =256, kernel_size = 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(in_channels = 256, out_channels = 128, kernel_size = 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(in_channels = 128, out_channels = 64, kernel_size = 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace = True),
            
            nn.ConvTranspose2d(in_channels = 64, out_channels = 3, kernel_size = 4, stride=2, padding=1, bias = False),
            #nn.BatchNorm2d()                        1511.06434 consiglia di non usare batchnorm per maggiore stabilitÃƒÂ 
            nn.Tanh()
        )
       
    
    def forward(self, x):
        return self.generator(x)
      

def _generator4():
    model = Generator()
    # Iinizializzo i layer
    for i in range(len(model.generator)):                              
       if type(model.generator[i]) == torch.nn.modules.conv.ConvTranspose2d:
          nn.init.normal_(model.generator[i].weight.data, mean = 0, std = 0.02)
       elif type(model.generator[i]) == torch.nn.modules.batchnorm.BatchNorm2d:
          nn.init.normal_(model.generator[i].weight.data, mean = 1, std = 0.02)
          nn.init.constant_(model.generator[i].bias.data, 0)
          
    return model


def Generator4():
    return _generator4()
    

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels = 3, out_channels =64, kernel_size = 4, stride=2, padding=1, bias = False),
            #nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(in_channels = 64, out_channels =128, kernel_size = 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(in_channels = 128, out_channels =256, kernel_size = 4, stride=2, padding=1, bias = False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size=4, stride=2, padding = 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(inplace = True, negative_slope = 0.2),
            nn.Conv2d(in_channels = 512, out_channels = 1, kernel_size=4, stride=1, padding = 0, bias=False),
            nn.Sigmoid()
        )
       
    
    def forward(self, x):
        return self.discriminator(x)
      

def _discriminator4():
    model = Discriminator()

    for i in range(len(model.discriminator)):                              
       if type(model.discriminator[i]) == torch.nn.modules.conv.Conv2d:
          nn.init.normal_(model.discriminator[i].weight.data, mean = 0, std = 0.02)
       elif type(model.discriminator[i]) == torch.nn.modules.batchnorm.BatchNorm2d:
          nn.init.normal_(model.discriminator[i].weight.data, mean = 1, std = 0.02)
          nn.init.constant_(model.discriminator[i].bias.data, 0)
    return model


def Discriminator4():
    return _discriminator4()

