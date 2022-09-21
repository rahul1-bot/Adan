
from __future__ import annotations
import torch, math
from torch.optim import Optimizer
from typing import Optional, Any, Callable, Union

__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Education': 'Amity University, Noida : Btech CSE (Final Year)',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

__license__: str = r'''
    MIT License
    Copyright (c) 2022 Rahul Sawhney
    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:
    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.
    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
'''


#@: Nesterov Momentum Algorithm
class Adan_Optimizer(Optimizer):
    def __init__(self, parameters: Any, learning_rate: Optional[float] = 1e-3, 
                                    betas: tuple[float, float, float] = (0.02, 0.07, 0.01),
                                    epsilon: Optional[float] = 1e-8,
                                    weight_decay: Optional[Union[int, float]] = 0,
                                    restart_condition: Optional[Callable[Any]] = None) -> None:
        if not len(betas) == 3:
            raise ValueError
        
        self.defaults_params: dict[str, Any] =  locals()
        self.defaults_params.pop('self')
        super(Adan_Optimizer, self).__init__(parameters, default_params)
    
    

    
    def step(self, closure: Callable[Any]) -> Union[int, float, NoneType]:
        loss: Union[int, float, NoneType] = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            learning_rate: float = group['lr']
            beta1, beta2, beta3 = group['betas']
            weight_decay: Union[int, float] = group['weight_decay']
            epsilon: float = group['eps']
            restart_condition: Callable[Any] = group['restart_cond']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                data, grad = p.data, p.grad.data
                state = self.state[p]  

                if len(state) == 0:
                    state['step'] = 0
                    state['prev_grad'] = torch.zeros_like(grad)
                    state['m'] = torch.zeros_like(grad)
                    state['v'] = torch.zeros_like(grad)
                    state['n'] = torch.zeros_like(grad)
                    
                step, m, v, n, prev_grad = state['step'], state['m'], state['v'], state['n'], state['prev_grad']
                if step > 0:
                    prev_grad = state['prev_grad']
                    m.mul_(1 - beta1).add_(grad, alpha = beta1)
                    grad_diff = grad - prev_grad
                    v.mul_(1 - beta2).add_(grad_diff, alpha = beta2)
                    next_n = (grad + (1 - beta2) * grad_diff) ** 2
                    n.mul_(1 - beta3).add_(next_n, alpha = beta3)
                
                step += 1
                correct_m, correct_v, correct_n = map(lambda n: 1 / (1 - (1 - n) ** step), (beta1, beta2, beta3))

                def grad_step_(data, m, v, n):
                    weighted_step_size = learning_rate / (n * correct_n).sqrt().add_(epsilon)
                    denom = 1 + weight_decay * learning_rate
                    data.addcmul_(weighted_step_size, (m * correct_m + (1 - beta2) * v * correct_v), value = -1.).div_(denom)
                    
                
                grad_step_(data, m, v, n)
                
                if restart_condition is not None and restart_condition(state):
                    m.data.copy_(grad)
                    v.zero_()
                    n.data.copy_(grad ** 2)
                    grad_step_(data, m, v, n)
                
                prev_grad.copy_(grad)
                state['step'] = step
                
        return loss           
                

