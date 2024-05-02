import os
import torch
from torch.optim import Optimizer
import torch.distributed as dist

from src.utils import DynamicLossScaler
from deepspeed.runtime.zero import GatheredParameters
import deepspeed


class LOMO(Optimizer):

    def __init__(self, model, llama_type, general_importance_dir, lr=1e-3, clip_grad_norm=None, clip_grad_value=None):
        self.is_update, self.is_end = None, None
        self.epoch = None

        self.model = model
        self.lr = lr
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = dist.get_world_size()
        self.clip_grad_norm = clip_grad_norm
        self.clip_grad_value = clip_grad_value


        if self.clip_grad_norm is not None and self.clip_grad_norm <= 0:
            raise ValueError(f"clip_grad_norm should be positive, got {self.clip_grad_norm}.")
        self.gather_norm = False
        self.grad_norms = []
        self.clip_coef = None
        self.llama_type = llama_type
        self.general_importance_dir = general_importance_dir

        if self.llama_type == "7b":
            param1, param2 = 4096, 11008
            self.increment = 32
        elif self.llama_type == "13b":
            param1, param2 = 5120, 13824
            self.increment = 40
        elif self.llama_type == "70b":
            param1, param2, param3 = 8192, 28672, 1024
            self.increment = 80

        if self.llama_type == "7b" or self.llama_type == "13b"::
            self.importance = [torch.zeros(param1, param1, dtype=torch.float32).to("cuda") for i in range(self.increment*4)]
        elif self.llama_type == "70b":
            self.importance = [torch.zeros(param1, param1, dtype=torch.float32).to("cuda") for i in range(self.increment)] + [torch.zeros(param3, param1, dtype=torch.float32).to("cuda") for i in range(self.increment*2)] + [torch.zeros(param1, param1, dtype=torch.float32).to("cuda") for i in range(self.increment)]

        for i in range(3):
            for j in range(self.increment):
                if i == 0:
                    self.importance.append(torch.zeros(param2, param1, dtype=torch.float32).to("cuda"))
                elif i == 1:
                    self.importance.append(torch.zeros(param1, param2, dtype=torch.float32).to("cuda"))
                elif i == 2:
                    self.importance.append(torch.zeros(param2, param1, dtype=torch.float32).to("cuda"))




        # check if zero3 is enabled
        p0 = list(self.model.parameters())[0]
        if hasattr(p0, 'ds_tensor'):  # zero3 is enabled
            self.grad_func = self.fuse_update_zero3()
        else:
            self.grad_func = self.fuse_update()
        # check if fp16 is enabled
        if p0.dtype == torch.float16:
            self.loss_scaler = DynamicLossScaler(
                init_scale=2 ** 16,
            )
            if self.clip_grad_norm is None:
                raise ValueError(
                    "Loss scaling is recommended to be used with grad norm to get better performance."
                )
        else:
            self.loss_scaler = None



        # register hook function, which will be called through the backward process
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                p.register_hook(self.grad_func)
        defaults = dict(lr=lr, clip_grad_norm=clip_grad_norm, clip_grad_value=clip_grad_value)
        super(LOMO, self).__init__(self.model.parameters(), defaults)



    def fuse_update(self):


        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.requires_grad and p.grad is not None:
                        if self.loss_scaler and self.loss_scaler.has_overflow_serial or self.loss_scaler._has_inf_or_nan(p.grad):
                            # if the overflow is detected, drop the gradient
                            p.grad = None
                            self.loss_scaler.has_overflow_serial = True
                            break
                        grad_fp32 = p.grad.to(torch.float32)
                        p.grad = None
                        if self.loss_scaler:
                            grad_fp32.div_(self.loss_scaler.loss_scale)
                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm compuation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:
                            if self.clip_grad_value is not None and self.clip_grad_value > 0:
                                # Clipping gradients by their value
                                grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                            if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:
                                # Normalize the gradient according to its norm (computed in another pass)
                                grad_fp32.mul_(self.clip_coef)
                            p_fp32 = p.data.to(torch.float32)
                            p_fp32.add_(grad_fp32, alpha=-self.lr)
                            p.data.copy_(p_fp32)

            return x

        return func

    def fuse_update_zero3(self):
        def func(x):
            with torch.no_grad():
                for n, p in self.model.named_parameters():
                    if p.grad is not None:
                        torch.distributed.all_reduce(p.grad, op=torch.distributed.ReduceOp.AVG, async_op=False)
                        if self.loss_scaler and self.loss_scaler.has_overflow_serial or self.loss_scaler._has_inf_or_nan(p.grad):
                            # if the overflow is detected, drop the gradient
                            p.grad = None
                            self.loss_scaler.has_overflow_serial = True
                            break
                                
                        grad_fp32 = p.grad.to(torch.float32)
                        lambda_weight = 0.1

                        if p.requires_grad:
                            if "self_attn.q_proj" in n:
                                layer = int(n.split(".")[2])
                                general = torch.load(self.general_importance_dir + 'q_proj/' + str(layer) + '.pt').to("cuda")
                                grad_fp32 = grad_fp32 + lambda_weight * 2 * pow(self.lr, 2) * torch.mul(general, torch.pow(grad_fp32, 2)) / 1000
                            elif "self_attn.k_proj" in n:
                                layer = int(n.split(".")[2])
                                general = torch.load(self.general_importance_dir + 'k_proj/' + str(layer) + '.pt').to("cuda")
                                grad_fp32 = grad_fp32 + lambda_weight * 2 * pow(self.lr, 2) * torch.mul(general, torch.pow(grad_fp32, 2)) / 1000
                            elif "self_attn.v_proj" in n:
                                layer = int(n.split(".")[2])
                                general = torch.load(self.general_importance_dir + 'v_proj/' + str(layer) + '.pt').to("cuda")
                                grad_fp32 = grad_fp32 + lambda_weight * 2 * pow(self.lr, 2) * torch.mul(general, torch.pow(grad_fp32, 2)) / 1000
                            elif "self_attn.o_proj" in n:
                                layer = int(n.split(".")[2])
                                general = torch.load(self.general_importance_dir + 'o_proj/' + str(layer) + '.pt').to("cuda")
                                grad_fp32 = grad_fp32 + lambda_weight * 2 * pow(self.lr, 2) * torch.mul(general, torch.pow(grad_fp32, 2)) / 1000
                            elif "mlp.gate_proj" in n:
                                layer = int(n.split(".")[2])
                                general = torch.load(self.general_importance_dir + 'gate_proj/' + str(layer) + '.pt').to("cuda")
                                grad_fp32 = grad_fp32 + lambda_weight * 2 * pow(self.lr, 2) * torch.mul(general, torch.pow(grad_fp32, 2)) / 1000
                            elif "mlp.down_proj" in n:
                                layer = int(n.split(".")[2])
                                general = torch.load(self.general_importance_dir + 'down_proj/' + str(layer) + '.pt').to("cuda")
                                grad_fp32 = grad_fp32 + lambda_weight * 2 * pow(self.lr, 2) * torch.mul(general, torch.pow(grad_fp32, 2)) / 1000
                            elif "mlp.up_proj" in n:
                                layer = int(n.split(".")[2])
                                general = torch.load(self.general_importance_dir + 'up_proj/' + str(layer) + '.pt').to("cuda")
                                grad_fp32 = grad_fp32 + lambda_weight * 2 * pow(self.lr, 2) * torch.mul(general, torch.pow(grad_fp32, 2)) / 1000
                        
                        if self.is_update and p.requires_grad:
                            if "self_attn.q_proj" in n:
                                layer = int(n.split(".")[2])
                                with GatheredParameters(p):
                                    temp_compute = torch.mul(grad_fp32, p.data)
                                    self.importance[layer] += torch.abs(temp_compute + torch.pow(temp_compute, 2) / 2000 + torch.pow(p.data, 3))


                            elif "self_attn.k_proj" in n:
                                layer = int(n.split(".")[2])
                                with GatheredParameters(p):
                                    temp_compute = torch.mul(grad_fp32, p.data)
                                    self.importance[self.increment+layer] += torch.abs(temp_compute + torch.pow(temp_compute, 2) / 2000 + torch.pow(p.data, 3))

                            elif "self_attn.v_proj" in n:
                                layer = int(n.split(".")[2])
                                with GatheredParameters(p):
                                    temp_compute = torch.mul(grad_fp32, p.data)
                                    self.importance[self.increment*2+layer] += torch.abs(temp_compute + torch.pow(temp_compute, 2) / 2000 + torch.pow(p.data, 3))

                            elif "self_attn.o_proj" in n:
                                layer = int(n.split(".")[2])
                                with GatheredParameters(p):
                                    temp_compute = torch.mul(grad_fp32, p.data)
                                    self.importance[self.increment*3+layer] += torch.abs(temp_compute + torch.pow(temp_compute, 2) / 2000 + torch.pow(p.data, 3))

                            elif "mlp.gate_proj" in n:
                                layer = int(n.split(".")[2])
                                with GatheredParameters(p):
                                    temp_compute = torch.mul(grad_fp32, p.data)
                                    self.importance[self.increment*4+layer] += torch.abs(temp_compute + torch.pow(temp_compute, 2) / 2000 + torch.pow(p.data, 3))

                            elif "mlp.down_proj" in n:
                                layer = int(n.split(".")[2])
                                with GatheredParameters(p):
                                    temp_compute = torch.mul(grad_fp32, p.data)
                                    self.importance[self.increment*5+layer] += torch.abs(temp_compute + torch.pow(temp_compute, 2) / 2000 + torch.pow(p.data, 3))

                            elif "mlp.up_proj" in n:
                                layer = int(n.split(".")[2])
                                with GatheredParameters(p):
                                    temp_compute = torch.mul(grad_fp32, p.data)
                                    self.importance[self.increment*6+layer] += torch.abs(temp_compute + torch.pow(temp_compute, 2) / 2000 + torch.pow(p.data, 3))



                        p.grad = None
                        param_fp32 = p.ds_tensor.to(torch.float32)
                        if self.loss_scaler:
                            grad_fp32.div_(self.loss_scaler.loss_scale)

                        if self.gather_norm:
                            # we adopt two backward pass for gradient norm compuation and parameter update, respectively.
                            self.grad_norms.append(torch.norm(grad_fp32, 2.0))
                        else:  # update param
                            one_dim_grad_fp32 = grad_fp32.view(-1)
                            partition_size = p.ds_tensor.numel()
                            start = partition_size * self.local_rank
                            end = min(start + partition_size, grad_fp32.numel())
                            partitioned_grad_fp32 = one_dim_grad_fp32.narrow(0, start, end - start)

                            if self.clip_grad_value is not None:

                                partitioned_grad_fp32.clamp_(min=-self.clip_grad_value, max=self.clip_grad_value)
                            if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is not None:

                                partitioned_grad_fp32.mul_(self.clip_coef)

                            partitioned_p = param_fp32.narrow(0, 0, end - start)
                            partitioned_p.add_(partitioned_grad_fp32, alpha=-self.lr)
                            p.ds_tensor[ : end - start] = partitioned_p
            return x

        return func

    def fused_backward(self, loss, lr, is_end=False, epoch=None):
        self.is_update, self.is_end = False, False
        self.lr = lr
        if self.clip_grad_norm is not None and self.clip_grad_norm > 0 and self.clip_coef is None:
            raise ValueError(
                "clip_grad_norm is not None, but clip_coef is None. "
                "Please call optimizer.grad_norm() before optimizer.fused_backward()."
            )
        if self.loss_scaler:
            loss = loss * self.loss_scaler.loss_scale
        # self.count += 1
        loss.backward()

        self.is_update, self.is_end = True, is_end
        # self.count += 1
        if is_end:
            self.epoch = epoch
        self.grad_func(0)
        # if is_end:
        #     self.epoch += 1



    def grad_norm(self, loss):
        """
        计算梯度的范数。

        :param loss: 模型的loss值
        """
        self.gather_norm = True
        self.grad_norms = []
        if self.loss_scaler:
            self.loss_scaler.has_overflow_serial = False
            loss = loss * self.loss_scaler.loss_scale

        loss.backward(retain_graph=True)

        self.is_update, self.is_end = False, False
        self.grad_func(0)

        if self.loss_scaler and self.loss_scaler.has_overflow_serial:
            self.loss_scaler.update_scale(overflow=True)
            with torch.no_grad():  # clear gradients
                for n, p in self.model.named_parameters():
                    p.grad = None
            return


        with torch.no_grad():
            self.grad_norms = torch.stack(self.grad_norms)

            total_norm = torch.norm(self.grad_norms, 2.0)
            self.clip_coef = float(self.clip_grad_norm) / (total_norm + 1e-6)
            self.clip_coef = torch.clamp(self.clip_coef, max=1.0)
        self.gather_norm = False
