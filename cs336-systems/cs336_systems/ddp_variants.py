import torch
import torch.distributed as dist
import torch.nn as nn
import torch

class OverlapIndParamDDP(nn.Module):
    def __init__(self, module: torch.nn.Module):
        super(OverlapIndParamDDP, self).__init__()
        
        self.module = module
        self.parameters = module.parameters
        self.handles = []

        for param in self.parameters():
            if param.requires_grad:
                torch.Tensor.register_post_accumulate_grad_hook(param, lambda x : self.call_all_reduce(x))
                param._all_reduce_called = False
        self._broadcast_parameters()
        
        return
    
    def call_all_reduce(self, param):
        if param.requires_grad and not param._all_reduce_called:
            dist.all_reduce(tensor=param.grad.data, op=dist.ReduceOp.SUM, async_op=False)
            param._all_reduce_called = True
        return

    def _broadcast_parameters(self):
        for param in self.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        outputs = self.module(*inputs, **kwargs)
        return outputs

    def finish_gradient_synchronization(self):
        """for handle in self.handles:
            handle.wait()
        self.handles.clear()"""
        for i, param in enumerate(self.parameters()):
            if param.requires_grad:
                """if i == 0:
                    print(param.grad.data)"""
                param.grad.data.div_(dist.get_world_size())
                param._all_reduce_called = False
                """if i == 0:
                    print(param.grad.data)"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return

class BucketOverlapIndParamDDP(nn.Module):
    def __init__(self, module: torch.nn.Module, bucket_size_mb: float):
        super(BucketOverlapIndParamDDP, self).__init__()
        
        self.module = module
        self.parameters = module.parameters
        self.handles = []
        self.param_buckets = { 0 : []}

        bucket_number = 0
        bucket_current_mb = 0

        # iterate through self.parameters backwards
        # set the first one to the first bucket
        #print(list(self.parameters()))

        #torch.Tensor.register_post_accumulate_grad_hook(list(self.parameters())[0], lambda x : self.call_all_reduce(x))

        for param in reversed(list(self.parameters())):
            if param.requires_grad:                
                param_size_mb = self.memory_size_mb(param)
                
                if bucket_current_mb + param_size_mb > bucket_size_mb and len(self.param_buckets[bucket_number]) > 0:
                    try:
                        torch.Tensor.register_post_accumulate_grad_hook(self.param_buckets[bucket_number][-1], lambda x : self.call_all_reduce(x))
                    except IndexError:
                        print(bucket_number)
                        print(self.param_buckets)
                        raise IndexError
                    bucket_number += 1
                    bucket_current_mb = param_size_mb
                    self.param_buckets[bucket_number] = []
                
                bucket_current_mb += param_size_mb
                self.param_buckets[bucket_number].append(param)
                param._bucket_id = bucket_number    
                param._all_reduce_called = False
        
        torch.Tensor.register_post_accumulate_grad_hook(self.param_buckets[bucket_number][-1], lambda x : self.call_all_reduce(x))
        #print([param.grad for param in self.param_buckets[0] if param.requires_grad])
        
        self._broadcast_parameters()
        
        return
    
    def call_all_reduce(self, param):
        bucket_tensors = self.param_buckets[param._bucket_id]
        try:
            flattened_grads = torch.cat([param.grad.view(-1) for param in bucket_tensors if param.requires_grad])
        except AttributeError:
            print([param.grad for param in bucket_tensors if param.requires_grad])
        
        if param.requires_grad and not param._all_reduce_called:
            dist.all_reduce(tensor=flattened_grads, op=dist.ReduceOp.SUM)

        start_idx = 0
        for param in bucket_tensors:
            param.grad.data = flattened_grads[start_idx:start_idx+param.numel()].view_as(param.grad.data)
            start_idx += param.numel()
            param._all_reduce_called = True
        
        return

    def _broadcast_parameters(self):
        for param in self.parameters():
            dist.broadcast(param.data, src=0)

    def forward(self, *inputs, **kwargs):
        outputs = self.module(*inputs, **kwargs)
        return outputs

    def finish_gradient_synchronization(self):
        """for handle in self.handles:
            handle.wait()
        self.handles.clear()"""
        for i, param in enumerate(self.parameters()):
            if param.requires_grad:
                """if i == 0:
                    print(param.grad.data)"""
                param.grad.data.div_(dist.get_world_size())
                param._all_reduce_called = False
                """if i == 0:
                    print(param.grad.data)"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        return
    
    @staticmethod
    def memory_size_mb(tensor):
        # Get the number of elements in the tensor
        numel = torch.numel(tensor)
        
        # Get the size of each element in bytes
        element_size = tensor.element_size()
        
        # Calculate the total memory size in bytes
        total_bytes = numel * element_size
        
        # Convert bytes to megabytes
        total_mb = total_bytes / (1024 * 1024)
        
        return total_mb