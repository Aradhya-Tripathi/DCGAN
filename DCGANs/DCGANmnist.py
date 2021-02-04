import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
torch.manual_seed(0) 

device = "cpu"

def show_tensor_images(image_tensor, num_images=25, size=(1, 28, 28)):
    '''
    Function for visualizing images: Given a tensor of images, number of images, and
    size per image, plots and prints the images in an uniform grid.
    '''
    image_tensor = (image_tensor + 1) / 2
    image_unflat = image_tensor.detach().cpu()
    image_grid = make_grid(image_unflat[:num_images], nrow=5)
    plt.imshow(image_grid.permute(1, 2, 0).squeeze())
    plt.show()
    
class Generator(nn.Module):
  '''
  Generator Class 
  parameters -> z_dim, hidden_dim, out_channels
  returns -> transpose convolution of the noise vector

  '''
  def __init__(self, z_dim=10, hidden_dim=64, out_channels=1):
    super(Generator, self).__init__()
    
    self.z_dim = z_dim
    self.gen = nn.Sequential(
            self.makegenblock(self.z_dim, hidden_dim * 4),
            self.makegenblock(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=1),
            self.makegenblock(hidden_dim * 2, hidden_dim),
            self.makegenblock(hidden_dim, out_channels, kernel_size=4, lastlayer=True),
        )
    
  def makegenblock(self, im_channels, out_channels, kernel_size=3, stride=2, lastlayer=False):
    if not lastlayer:
      gen = nn.Sequential(
                    nn.ConvTranspose2d(im_channels, out_channels, kernel_size=kernel_size, stride=stride),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(),
                    
                    )
    else:
      gen = nn.Sequential(
                    nn.ConvTranspose2d(im_channels, out_channels, kernel_size=kernel_size, stride=stride),
                    nn.Tanh() ## ranges between -1, 1  
      )

    return gen

  def reshape_noise(self, noise):
    '''
    Reshape the noise vector (num_samples, z_dim)
    to vector of dimentions (num_samples, z_dim, 1, 1) for convolution 
    
    '''
    return noise.view(len(noise), self.z_dim, 1, 1)

  def forward(self, noise):
    noise = self.reshape_noise(noise)
    return self.gen(noise)

gen = Generator(z_dim=64).to(device)


class Discriminator(nn.Module):
  '''
  Discriminator class 
  parameters -> hidden_dim, out_channel
  returns -> vector of shape (num_samples, 1) for sigmoid activation

  '''
  def __init__(self, hidden_dim=16, im_channels=1):
    super(Discriminator, self).__init__()
    self.disc = nn.Sequential(
            self.getdiscblock(im_channels, hidden_dim),
            self.getdiscblock(hidden_dim, hidden_dim * 2),
            self.getdiscblock(hidden_dim * 2, 1, lastlayer=True),
        )

  
  def getdiscblock(self,im_channels, out_channels, kernel_size=4, stride=2, lastlayer=False):
    
    if not lastlayer:
      disc = nn.Sequential(
                      nn.Conv2d(im_channels, out_channels, kernel_size=kernel_size, stride=stride),
                      nn.BatchNorm2d(out_channels),
                      nn.LeakyReLU(0.2)
      )

    else:
      disc = nn.Sequential(
                      nn.Conv2d(im_channels, out_channels, kernel_size=kernel_size, stride=stride),
      )

    return disc
  def forward(self, image):

    return self.disc(image).view(len(image), -1)

disc = Discriminator().to(device)
criterion = nn.BCEWithLogitsLoss()
z_dim = 64
display_step = 500
batch_size = 128

lr = 0.0002

beta_1 = 0.5 
beta_2 = 0.999
device = 'cpu'

# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

dataloader = DataLoader(
    MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)


gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))


def weights_init(m):
  '''
  Initialize weights to normal distribution with mean 0 and std of 0.02

  '''
  if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
  if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

def make_noise(num_samples, z_dim):
  '''
  Generates random noise of normal dist.
  returns -> noise of shape (num_samples, z_dim)

  '''
  return torch.randn(num_samples, z_dim, device=device)


cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

## training
for i in range(200):
  for x, _ in tqdm(dataloader):

    ## training discriminator
    x = x.to(device)
    noise = make_noise(len(x), z_dim)
    discfakeout = disc(gen(noise).detach())
    discrealout = disc(x)

    lossfakeout = criterion(discfakeout, torch.zeros_like(discfakeout))
    lossrealout = criterion(discrealout, torch.ones_like(discrealout))

    total_loss = (lossfakeout + lossrealout) / 2
    mean_discriminator_loss += total_loss.item() / display_step
    disc_opt.zero_grad()
    total_loss.backward(retain_graph=True)

    disc_opt.step()


    ## training generator
    noise = make_noise(len(x), z_dim)
    genfakeout = disc(gen(noise))

    lossgenout = criterion(genfakeout, torch.ones_like(genfakeout))
    
    gen_opt.zero_grad()

    lossgenout.backward()
    mean_generator_loss += lossgenout.item() / display_step
    gen_opt.step()

    if cur_step % display_step == 0 and cur_step > 0:
            print(f"Step {cur_step}: Generator loss: {mean_generator_loss}, discriminator loss: {mean_discriminator_loss}")
            fake = gen(noise)
            show_tensor_images(fake)
            show_tensor_images(x)
            mean_generator_loss = 0
            mean_discriminator_loss = 0
    cur_step += 1