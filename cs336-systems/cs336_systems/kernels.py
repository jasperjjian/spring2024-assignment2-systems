import torch.nn as nn
import torch
import triton
import triton.language as tl

class rms_norm_triton_torch(nn.Module):
    def __init__(self, H, eps=1e-5):
        super(rms_norm_triton_torch, self).__init__()
        self.H = H
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(H))
        
    def forward(self, x):
        return rms_norm.apply(x, self.weight, self.eps)
    

class rms_norm_triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-5):
    # Remember x and weight for the backward pass, when we
    # only receive the gradient wrt. the output tensor, and
    # need to compute the gradients wrt. x and weight.
        ctx.save_for_backward(x, weight)
        
        H, output_dims = x.shape[-1], x.shape[:-1]

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        flattened_x = x.view(-1, H)
        flattened_y = torch.empty(flattened_x.shape, device=x.device)

    # Launch our kernel with n instances in our 2D grid??
        n_rows = flattened_y.numel()
        rms_norm_fwd[(n_rows, )](
        flattened_x, weight, flattened_x.stride(0), flattened_y.stride(0), flattened_y, H, eps,
        BLOCK_SIZE=ctx.BLOCK_SIZE)
        return flattened_y.view(x.shape)
    
    def backward(ctx, grad_output):
        print(ctx.saved_tensors)
        x, weight = ctx.saved_tensors
        grad_g = rms_norm_triton.backward_g(ctx, grad_output, x, weight)
        grad_x = rms_norm_triton.backward_x(ctx, grad_output, x, weight)

        return grad_x, grad_g
    
    def backward_g(ctx, grad_output, x, weight, eps: float = 1e-5):
        H, output_dims = x.shape[-1], x.shape[:-1]

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        flattened_x = x.view(-1, H)
        flattened_grad_g = torch.empty(flattened_x.shape, device=x.device)
        flattened_grad_output = grad_output.view(-1, H)
        n_rows = flattened_grad_g.numel()
        rms_norm_backwards_g[(n_rows, )](
        
        flattened_x, flattened_grad_output, flattened_x.stride(0), flattened_grad_output.stride(0), flattened_grad_g.stride(0), flattened_grad_g, H, eps,
        BLOCK_SIZE=ctx.BLOCK_SIZE)
        
        dims_to_sum = tuple(range(len(flattened_grad_g.shape) - 1))
        flattened_grad_g = torch.sum(flattened_grad_g, dim=dims_to_sum, keepdim=True)
        
        return flattened_grad_g

    def backward_x(ctx, grad_output, x, weight, eps: float = 1e-5):
        H, output_dims = x.shape[-1], x.shape[:-1]

        assert len(weight.shape) == 1 and weight.shape[0] == H, "Dimension mismatch"
        assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

        ctx.BLOCK_SIZE = triton.next_power_of_2(H)
        flattened_x = x.view(-1, H)
        flattened_grad_x = torch.empty(flattened_x.shape, device=x.device)
        flattened_grad_output = grad_output.view(-1, H)
        
        n_rows = flattened_grad_x.numel()

        rms_norm_backwards_x[(n_rows, )](
            
        flattened_x, flattened_grad_output, weight, flattened_x.stride(0), flattened_grad_output.stride(0), flattened_grad_x.stride(0), flattened_grad_x, H, eps,
        BLOCK_SIZE=ctx.BLOCK_SIZE)
                
        return flattened_grad_x.view(x.shape)
    
@triton.jit
def rms_norm_backwards_x(
    x_ptr : tl.pointer_type,
    grad_ptr : tl.pointer_type,
    weight_ptr : tl.pointer_type,
    x_row_stride : tl.uint32,
    grad_row_stride : tl.uint32,
    y_row_stride : tl.uint32,
    output_ptr : tl.pointer_type,
    H : tl.uint32,
    eps,
    BLOCK_SIZE: tl.constexpr):
    
    #load data
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)

    grad_idx = tl.program_id(0)
    grad_start_ptr = grad_ptr + grad_idx * grad_row_stride
    grad_ptrs = grad_start_ptr + offsets
    grad_row = tl.load(grad_ptrs, mask=mask, other=0)

    weight_ptrs = weight_ptr + offsets
    weight = tl.load(weight_ptrs, mask=mask, other=0)

    #compute rms
    rms = 1 / tl.sqrt(1/H * tl.sum(row * row) + eps)
    ms = 1/H * tl.sum(row * row) + eps
    lhs = weight * grad_row
    rhs = row * tl.sum(lhs * row) / (H * ms)
    
    
    output = (lhs - rhs) * rms

    #output is a vector of same length as x
    output_ptrs = (output_ptr + row_idx * y_row_stride) + offsets
    tl.store(output_ptrs, output, mask=mask)

@triton.jit
def rms_norm_backwards_g(
    x_ptr : tl.pointer_type,
    grad_ptr : tl.pointer_type,
    x_row_stride : tl.uint32,
    grad_row_stride : tl.uint32,
    y_row_stride : tl.uint32,
    output_ptr : tl.pointer_type,
    H : tl.uint32,
    eps,
    BLOCK_SIZE: tl.constexpr):
    
    #load data
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    #weight_ptrs = weight_ptr + offsets
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)

    grad_idx = tl.program_id(0)
    grad_start_ptr = grad_ptr + grad_idx * grad_row_stride
    grad_ptrs = grad_start_ptr + offsets
    grad_row = tl.load(grad_ptrs, mask=mask, other=0)

    #compute rms
    rms = 1 / tl.sqrt(1/H * tl.sum(row * row) + eps) 
    x_over_rms = row * rms
    #weight = tl.load(weight_ptrs, mask=mask, other=0)
    output = x_over_rms * grad_row

    #output is a vector of same length as x
    output_ptrs = (output_ptr + row_idx * y_row_stride) + offsets
    tl.store(output_ptrs, output, mask=mask)

@triton.jit
def rms_norm_fwd(
    x_ptr : tl.pointer_type,
    weight_ptr : tl.pointer_type,
    x_row_stride : tl.uint32,
    y_row_stride : tl.uint32,
    output_ptr : tl.pointer_type,
    H : tl.uint32,
    eps,
    BLOCK_SIZE: tl.constexpr):
    
    #load data
    row_idx = tl.program_id(0)
    row_start_ptr = x_ptr + row_idx * x_row_stride
    offsets = tl.arange(0, BLOCK_SIZE)
    x_ptrs = row_start_ptr + offsets
    weight_ptrs = weight_ptr + offsets
    mask = offsets < H
    row = tl.load(x_ptrs, mask=mask, other=0)

    #compute rms
    rms = 1 / tl.sqrt(1/H * tl.sum(row * row) + eps) 
    x_over_rms = row * rms
    weight = tl.load(weight_ptrs, mask=mask, other=0)
    output = x_over_rms * weight

    #output is a vector of same length as x
    output_ptrs = (output_ptr + row_idx * y_row_stride) + offsets
    tl.store(output_ptrs, output, mask=mask)

class rms_norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps: float = 1e-5,):
        ctx.save_for_backward(x, weight)
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        x = x * rms
        output = weight * x 
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        
        x.grad = rms_norm.backward_x(grad_output, x, weight) / 2
        weight.grad = rms_norm.backward_g(grad_output, x, weight).squeeze(0).squeeze(0) / 2
        return x.grad, weight.grad
    
    @staticmethod
    def backward_g(grad_output, x, weight, eps: float = 1e-5):
        # Calculates the backward pass wrt g
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        x_over_rms = x * rms
        
        # pointwise multiply x_over_rms with grad_output
        dims_to_sum = tuple(range(len(x_over_rms.shape) - 1))

        # Sum over all initial dimensions while keeping the last dimension intact
        grad_g = x_over_rms * grad_output
        return torch.sum(grad_g, dim=dims_to_sum, keepdim=True) 
    
    @staticmethod
    def backward_x(grad_output, x, weight, eps: float = 1e-5):
        # Calculates the backward pass wrt x
        H = x.shape[-1]
        flattened_x = x.view(-1, H)
        rms = torch.rsqrt(flattened_x.pow(2).mean(-1, keepdim=True) + eps)
        
        grad_x = torch.sum(flattened_x * weight * grad_output.view(-1, H), dim=-1, keepdim=True)
        output = (weight * grad_output.view(-1, H)) * rms - flattened_x * grad_x  * rms.pow(3) / (H)
        
        return output.view(x.shape)

class RMSNorm(nn.Module):
    """
    This module implements root mean square layer normalization, as
    described in Eq. 4 of https://arxiv.org/abs/1910.07467

    Args:
        hidden_size: int
            Dimensionality of the input to normalize.
        eps: float, default is 1e-5
            A value added to the denominator for numerical stability.

    Returns:
        FloatTensor of same shape as input.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        """
        Args:
            x: FloatTensor of shape `(batch_size, *)`.
                The input to apply root mean square layer normalization on.

        Returns:
            FloatTensor of same shape as input
        """
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        x = x * rms
        return self.weight * x