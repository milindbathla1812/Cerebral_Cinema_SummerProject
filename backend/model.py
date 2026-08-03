import torch
import torch.nn as nn
class PositionalEmbedding(nn.Module):

    def __init__(self,d_model,max_len=8):

        super().__init__()

        self.pos=nn.Parameter(

            torch.randn(

                1,

                max_len,

                d_model

            )

        )

    def forward(self,x):

        return x+self.pos

class CLSToken(nn.Module):

    def __init__(self,d_model):

        super().__init__()

        self.cls=nn.Parameter(

            torch.randn(

                1,

                1,

                d_model

            )

        )

    def forward(self,x):

        B=x.size(0)

        cls=self.cls.expand(

            B,

            -1,

            -1

        )

        return torch.cat(

            [

                cls,

                x

            ],

            dim=1

        )
class ModalityDropout(nn.Module):

    def __init__(self,p=0.15):

        super().__init__()

        self.p=p

    def forward(self,x):

        if not self.training:

            return x

        B=x.size(0)

        device=x.device

        text_keep=(torch.rand(B,device=device)>self.p).float()

        video_keep=(torch.rand(B,device=device)>self.p).float()

        x=x.clone()

        x[:,:,:2048]*=text_keep[:,None,None]

        x[:,:,2048:]*=video_keep[:,None,None]

        return x
class TribeLite(nn.Module):

    def __init__(self):

        super().__init__()

        self.modality_dropout=ModalityDropout(0.15)

        self.input_proj=nn.Sequential(

            nn.LayerNorm(2816),

            nn.Linear(2816,384),

            nn.GELU(),

            nn.Dropout(0.1)

        )

        self.pos=PositionalEmbedding(

            384,

            max_len=8

        )

        self.cls=CLSToken(384)

        encoder_layer=nn.TransformerEncoderLayer(

            d_model=384,

            nhead=8,

            dim_feedforward=1024,

            dropout=0.1,

            activation="gelu",

            batch_first=True

        )

        self.encoder=nn.TransformerEncoder(

            encoder_layer,

            num_layers=4

        )

        self.head=nn.Sequential(

            nn.LayerNorm(384),

            nn.Linear(

                384,

                768

            ),

            nn.GELU(),

            nn.Dropout(0.2),

            nn.Linear(

                768,

                1000

            )

        )

    def forward(self,x):

        x=self.modality_dropout(x)

        x=self.input_proj(x)

        x=self.pos(x)

        x=self.cls(x)

        x=self.encoder(x)

        cls=x[:,0]

        return self.head(cls)
    

def load_model(weights_path, device=None):

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = TribeLite()

    state_dict = torch.load(
        weights_path,
        map_location=device
    )

    model.load_state_dict(state_dict)

    model.to(device)
    model.eval()

    return model