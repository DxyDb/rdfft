import torch

import extension_cpp




myrfft = extension_cpp.ops.myrfft
myirfft = extension_cpp.ops.myirfft


batch, seq_len, d = 1, 1, 128
block_size = 128
rows = (d + d%block_size) // block_size # out_features
cols = (d + d%block_size) // block_size 

x = torch.randn([batch,seq_len, d], dtype=torch.double, requires_grad=True).to("cuda")

### test irfft
x = x.view([-1, 1, cols, block_size])
original_x = x.clone()
x = myirfft(myrfft(x))
diff = original_x - x
print(diff.abs().max()) # this may cause "Backward is not reentrant..."