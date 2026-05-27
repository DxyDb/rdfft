# import sys
# import os

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print("Importing extension_cpp.ops")
import numpy as np
import torch

import extension_cpp
print(extension_cpp.ops.__file__)
myfft = extension_cpp.ops.myrfft
myifft = extension_cpp.ops.myirfft


r=512
sub= 0
sub_x = 0
for _ in range(5):
    x = torch.rand(1,1,1,r).to('cuda')
    x1 = x.clone()

    x = x.view(-1,r)
    x = torch.fft.fft(x, dim=-1)
    x = torch.fft.ifft(x, dim=-1)
    x = x.view(1,1,1,r).to('cpu')


    # print(torch.cuda.max_memory_allocated(device=None))
    x1 = myfft(x1)
    # print(torch.cuda.max_memory_allocated(device=None))
    x1 = myifft(x1).to('cpu')

    m = (x1-x).abs()
    n = (m/x).abs()
    sub += m.max()
    sub_x += n.max()
print(r, sub/5, sub_x/5)
    # print(torch.cuda.max_memory_allocated(device=None))
    # print(np.allclose(x, x1, atol=1e-5))




