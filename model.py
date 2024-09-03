import torch
import torch.nn as nn
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

class DoppiaConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoppiaConv, self).__init__()
    self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,3,1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1,1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

  def forward(self,x):
    return self.conv(x)


class UNET(nn.Module):
    def __init__(self, in_channels, out_channels, features=[64, 128, 256, 512]):
      super(UNET, self).__init__()
      #Crea un contenitore (ModuleList) per memorizzare gli strati di upsampling e di downsampling
      self.ups=nn.ModuleList()
      self.downs=nn.ModuleList()
      #Crea uno strato di pooling massimo con dimensione del kernel di 2x2 e stride di 2 per il downsampling.
      self.pool=nn.MaxPool2d(kernel_size=2, stride=2)

      #Down
      for f in features:
        self.downs.append(DoppiaConv(in_channels, f))
        in_channels=f

      #Up
      for f in reversed(features):
        self.ups.append(nn.ConvTranspose2d(f*2, f, kernel_size=2, stride=2))
        self.ups.append(DoppiaConv(f*2,f))

      #Crea un'istanza finale della classe DoppiaConv come "collo di bottiglia" nella parte più profonda della rete
      #Questo strato ha il doppio del numero di canali di output rispetto all'ultimo strato nel percorso discendente.
      self.bneck=DoppiaConv(features[-1], features[-1]*2)
      self.f_conv=nn.Conv2d(features[0],out_channels, kernel_size=1)

    def forward(self, x):
      #Crea una lista vuota per memorizzare le feature map da ogni strato di downsampling (connessioni di salto).
      skip_connections=[]

      for d in self.downs:
        x=d(x)
        skip_connections.append(x)
        x=self.pool(x)

      x=self.bneck(x)
      #Inverte l'ordine della lista
      skip_connections=skip_connections[: : -1]

      #percorso di salita
      #itero a coppie
      for idx in range(0, len(self.ups),2):
        x=self.ups[idx](x)
        #Recupera la connessione di salto corrispondente dalla lista skip_connections usando l'indice idx//2.
        skip_connection=skip_connections[idx//2]

        #Se le dimensioni sono diverse, utilizza un'operazione di ridimensionamento
        if x.shape != skip_connection.shape:
          x=TF.resize(x, size=skip_connection.shape[2:])

        #Concatena le feature map di skip_connection e x lungo la dimensione del canale (dim=1)
        #per combinare le informazioni da diversi livelli di astrazione.
        concat_skip=torch.cat((skip_connection, x), dim=1)
        x=self.ups[idx+1](concat_skip)

      return self.f_conv(x)