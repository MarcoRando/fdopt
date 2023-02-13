import math
import numbers

import torch
from torch import nn
import torch.nn.functional as F

from eszd.directions.butterfly_multiply import butterfly_multiply
#from torch_butterfly.complex_utils import  complex_reshape


class Butterfly(nn.Module):
    """Product of log N butterfly factors, each is a block 2x2 of diagonal matrices.
    Compatible with torch.nn.Linear.
    Parameters:
        in_size: size of input
        out_size: size of output
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: a torch.Tensor, or 'randn', 'ortho', 'identity', 'fft_no_br', or 'ifft_no_br'.
            Whether the weight matrix should be initialized to from randn twiddle, or to be
            randomly orthogonal/unitary, or to be the identity matrix, or the normalized FFT/iFFT
            twiddle (without the bit-reversal permutation).
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, increasing_stride=True, init='randn', dtype=torch.float32, nblocks=1):
        super().__init__()
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.nstacks = int(math.ceil(out_size / self.n))
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        self.dtype = dtype#torch.get_default_dtype() #if not self.complex else real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = (self.nstacks, nblocks, log_n, n // 2, 2, 2)
        if isinstance(init, torch.Tensor):
            self.init = None
            assert init.shape == twiddle_shape
            assert init.dtype == dtype
            self.twiddle = nn.Parameter(init.clone())
        else:
            assert init in ['empty', 'randn', 'ortho', 'identity', 'fft_no_br', 'ifft_no_br']
            self.init = init
            self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
#        if bias:
#            self.bias = nn.Parameter(torch.empty(out_size, dtype=dtype))
#        else:
        self.register_parameter('bias', None)
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize bias the same way as torch.nn.Linear."""
#        if self.bias is not None:
#            bound = 1 / math.sqrt(self.in_size)
#            nn.init.uniform_(self.bias, -bound, bound)
        twiddle = self.twiddle
        if self.init is None or self.init == 'empty':
            return
        elif self.init == 'randn':
            # complex randn already has the correct scaling of stddev=1.0
            scaling = 1.0 / math.sqrt(2)
            with torch.no_grad():
                twiddle.copy_(torch.randn(twiddle.shape, dtype=twiddle.dtype) * scaling)
        elif self.init == 'ortho':
            twiddle_core_shape = twiddle.shape[:-2]
            theta = torch.rand(twiddle_core_shape) * math.pi * 2
            c, s = torch.cos(theta), torch.sin(theta)
            det = torch.randint(0, 2, twiddle_core_shape, dtype=c.dtype) * 2 - 1  # Rotation (+1) or reflection (-1)
            with torch.no_grad():
                twiddle.copy_(torch.stack((torch.stack((det * c, -det * s), dim=-1),
                                            torch.stack((s, c), dim=-1)), dim=-2))

        elif self.init == 'identity':
            twiddle_eye = torch.eye(2, dtype=twiddle.dtype).reshape(1, 1, 1, 1, 2, 2)
            twiddle_eye = twiddle_eye.expand(*twiddle.shape).contiguous()
            with torch.no_grad():
                twiddle.copy_(twiddle_eye)


    def forward(self, input, transpose=False, conjugate=False, subtwiddle=False):
        """
        Parameters:
            input: (batch, *, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
            subtwiddle: allow using only part of the parameters for smaller input.
                Could be useful for weight sharing.
                out_size is set to self.nstacks * self.n in this case
        Return:
            output: (batch, *, out_size)
        """
        twiddle = self.twiddle
        output = self.pre_process(input)
        output_size = self.out_size if self.nstacks == 1 else None
        if subtwiddle:
            log_n = int(math.ceil(math.log2(input.size(-1))))
            n = 1 << log_n
            twiddle = (twiddle[:, :, :log_n, :n // 2] if self.increasing_stride
                       else twiddle[:, :, -log_n:, :n // 2])
            output_size = None
#        if conjugate and self.complex:
#            twiddle = twiddle.conj()
        if not transpose:
            output = butterfly_multiply(twiddle, output, self.increasing_stride, output_size)
        else:
            twiddle = twiddle.transpose(-1, -2).flip([1, 2])
            last_increasing_stride = self.increasing_stride != ((self.nblocks - 1) % 2 == 1)
            output = butterfly_multiply(twiddle, output, not last_increasing_stride, output_size)
        if not subtwiddle:
            return self.post_process(input, output)
        else:
            return self.post_process(input, output, out_size=output.size(-1))

    def pre_process(self, input):
        # Reshape to (N, in_size)
        input_size = input.size(-1)
        output = input.reshape(-1, input_size) #complex_reshape(input, -1, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(1).expand(batch, self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.nstacks * output.size(-1))
        if out_size != output.shape[-1]:  # Take top rows
            output = output[:, :out_size]
#        if self.bias is not None:
#            output = output + self.bias[:out_size]
        return output.view(*input.size()[:-1], out_size)

    def __imul__(self, scale):
        """In-place multiply the whole butterfly matrix by some scale factor, by multiplying the
        twiddle.
        Scale must be nonnegative
        """
        assert isinstance(scale, numbers.Number)
        assert scale >= 0
        self.twiddle *= scale ** (1.0 / self.twiddle.shape[1] / self.twiddle.shape[2])
        return self

    



class ButterflyBmm(Butterfly):
    """Same as Butterfly, but performs batched matrix multiply.
    Compatible with torch.nn.Linear.
    Parameters:
        in_size: size of input
        out_size: size of output
        matrix_batch: how many butterfly matrices
        bias: If set to False, the layer will not learn an additive bias.
                Default: ``True``
        complex: whether complex or real
        increasing_stride: whether the first butterfly block will multiply with increasing stride
            (e.g. 1, 2, ..., n/2) or decreasing stride (e.g., n/2, n/4, ..., 1).
        init: 'randn', 'ortho', or 'identity'. Whether the weight matrix should be initialized to
            from randn twiddle, or to be randomly orthogonal/unitary, or to be the identity matrix.
        nblocks: number of B or B^T blocks. The B and B^T will alternate.
    """

    def __init__(self, in_size, out_size, matrix_batch=1, increasing_stride=True, init='randn', nblocks=1):
        nn.Module.__init__(self)
        self.in_size = in_size
        self.log_n = log_n = int(math.ceil(math.log2(in_size)))
        self.n = n = 1 << log_n
        self.out_size = out_size
        self.matrix_batch = matrix_batch
        self.nstacks = int(math.ceil(out_size / self.n))
        self.increasing_stride = increasing_stride
        assert nblocks >= 1
        self.nblocks = nblocks
        dtype = torch.get_default_dtype() #if not self.complex else real_dtype_to_complex[torch.get_default_dtype()]
        twiddle_shape = (self.matrix_batch * self.nstacks, nblocks, log_n, n // 2, 2, 2)
        if isinstance(init, torch.Tensor):
            self.init = None
            assert init.shape == twiddle_shape
            assert init.dtype == dtype
            self.twiddle = nn.Parameter(init.clone())
        else:
            assert init in ['randn', 'ortho', 'identity', 'fft_no_br', 'ifft_no_br']
            self.init = init
            self.twiddle = nn.Parameter(torch.empty(twiddle_shape, dtype=dtype))
#        if bias:
#            self.bias = nn.Parameter(torch.empty(self.matrix_batch, out_size, dtype=dtype))
#        else:
        self.register_parameter('bias', None)
        self.twiddle._is_structured = True  # Flag to avoid weight decay
        self.reset_parameters()

    def forward(self, input, transpose=False, conjugate=False):
        """
        Parameters:
            input: (batch, *, matrix_batch, in_size)
            transpose: whether the butterfly matrix should be transposed.
            conjugate: whether the butterfly matrix should be conjugated.
        Return:
            output: (batch, *, matrix_batch, out_size)
        """
        return super().forward(input, transpose, conjugate, subtwiddle=False)

    def pre_process(self, input):
        # Reshape to (N, matrix_batch, in_size)
        input_size = input.size(-1)
        assert input.size(-2) == self.matrix_batch
        output = input.reshape(-1, self.matrix_batch, input_size) #complex_reshape(input, -1, self.matrix_batch, input_size)
        batch = output.shape[0]
        output = output.unsqueeze(2).expand(batch, self.matrix_batch, self.nstacks, input_size)
        output = output.reshape(batch, self.matrix_batch * self.nstacks, input_size)
        return output

    def post_process(self, input, output, out_size=None):
        if out_size is None:
            out_size = self.out_size
        batch = output.shape[0]
        output = output.view(batch, self.matrix_batch, self.nstacks * output.size(-1))
        if out_size != output.shape[-1]:  # Take top rows
            output = output[:, :, :out_size]
#        if self.bias is not None:
#            output = output + self.bias[:, :out_size]
        return output.view(*input.size()[:-2], self.matrix_batch, self.out_size)

