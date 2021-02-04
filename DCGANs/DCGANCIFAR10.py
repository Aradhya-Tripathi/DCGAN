import torch
from torch import nn
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
# torch.manual_seed(0) 

device = "cuda"

def show_tensor_images(image_tensor, num_images=25, size=(3, 32, 32)):
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
  def __init__(self, z_dim=100, out_channels=3, hidden_size=64):
    super(Generator, self).__init__()

    self.z_dim = z_dim
    self.generator = nn.Sequential(

                      nn.ConvTranspose2d(self.z_dim, hidden_size*4, kernel_size=4, stride=1, padding=0, bias=False),
                      nn.BatchNorm2d(hidden_size*4),
                      nn.ReLU(inplace=True),

                      nn.ConvTranspose2d(hidden_size*4, hidden_size*2, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(hidden_size*2),
                      nn.ReLU(inplace=True), 

                      nn.ConvTranspose2d(hidden_size*2, hidden_size, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.BatchNorm2d(hidden_size),
                      nn.ReLU(inplace=True),

                      nn.ConvTranspose2d(hidden_size, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                      nn.Tanh(),
    )

  
  def changeNoise(self, noise):
    return noise.view(len(noise), self.z_dim, 1, 1)


  def forward(self, vector):
    return self.generator(self.changeNoise(vector))


class Discriminator(nn.Module):
  def __init__(self, im_channels=3, hidden_dim=64):
    super(Discriminator, self).__init__()

    self.discriminator = nn.Sequential(

                        nn.Conv2d(im_channels, hidden_dim , kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(hidden_dim),
                        nn.LeakyReLU(0.2, inplace=True),

                        nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(hidden_dim * 2),
                        nn.LeakyReLU(0.2, inplace=True),
                        
                        nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
                        nn.BatchNorm2d(hidden_dim * 4),
                        nn.LeakyReLU(0.2, inplace=True),
                        
                        nn.Conv2d(hidden_dim * 4, 1, kernel_size=4, stride=1, padding=0, bias=False),
                      
    )

  
  def forward(self, image):
    return self.discriminator(image)


criterion = nn.BCEWithLogitsLoss()
z_dim = 64
display_step = 500
batch_size = 64

lr = 0.0002

beta_1 = 0.5 
beta_2 = 0.999


# You can tranform the image values to be between -1 and 1 (the range of the tanh activation)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

dataloader = DataLoader(
    CIFAR10('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)



gen = Generator(z_dim).to(device)
gen_opt = torch.optim.Adam(gen.parameters(), lr=lr, betas=(beta_1, beta_2))
disc = Discriminator().to(device) 
disc_opt = torch.optim.Adam(disc.parameters(), lr=lr, betas=(beta_1, beta_2))

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
gen = gen.apply(weights_init)
disc = disc.apply(weights_init)

def make_noise(num_samples, z_dim):
  return torch.randn(num_samples, z_dim, device=device)


cur_step = 0
mean_generator_loss = 0
mean_discriminator_loss = 0

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