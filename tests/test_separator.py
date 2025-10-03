import torch
import math
from src.models.separator import Separator
from src.models.encoder import Encoder
from src.models.decoder import Decoder

def main():
    # connect up an instantiation of encoder -> separator -> decoder
    enc = Encoder()
    sep = Separator()
    dec = Decoder()
    x = torch.randn(2,1,16000) # simulating 2 batches of 1 second audio

    enc_out = enc(x)                             # (B, N, T')
    masks = sep(enc_out)                         # (B, C, N, T')

     # apply masks and decode
    masked = masks * enc_out.unsqueeze(1)        # (B, C, N, T')
    BATCH, C_, N_, Tprime = masked.shape
    masked_bc = masked.view(BATCH * C_, N_, Tprime)
    dec_out = dec(masked_bc)                      # (B*C, 1, T) or (B*C, T)
    dec_out = dec_out.view(BATCH, C_, -1)         # (B, C, T)

    # minimal prints
    print("x:", x.shape)
    print("enc_out:", enc_out.shape)
    print("masks:", masks.shape)
    print("masked_bc:", masked_bc.shape)
    print("dec_out:", dec_out.shape)

if __name__ == "__main__":
    main()
