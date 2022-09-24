
from __future__ import annotations
import torch, math
from typing import Optional, Any, Callable, Union, Iterable

__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Education': 'Amity University, Noida : Btech CSE (Final Year)',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

#@: date = 24 September, 2022

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


class LayerBatchOptimizer(torch.optim.Optimizer):
    def __init__(self, parameters: Iterable[Any], learning_rate: Optional[float] = 1e-3, 
                                                  betas: Optional[tuple[float, float]] = (0.9, 0.999),
                                                  epsilon: Optional[float] = 1e-6,
                                                  weight_decay: Optional[Union[int, float]] = 0, 
                                                  adam_compare: Optional[bool] = False) -> None:
        
        if not isinstance(learning_rate, float) or learning_rate <= 0.0:
            raise ValueError
        
        if not isinstance(epsilon, float) or epsilon <= 0.0:
            raise ValueError
        
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0 or not isinstance(betas, tuple):
            raise ValueError
        
        default_params: dict[str, Any] = {
            key : value for key, value in locals().items() if not key in [
                'self', 'parameters', 'adam_compare'
            ]
        }
        self.adam_compare = self.adam_compare
        super(LayerBatchOptimizer, self).__init__(parameters, default_params)
    
    
    
    def step(self, closure: Optional[Union[Callable[Any], NoneType]] = None) -> Union[int, float, NoneType]:
        loss: Union[int, float, NoneType] = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad: Any = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError
                
                state: Any = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']   
                
                state['step'] += 1
                
                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)
                step_size = group['lr'] 
                weight_norm = p.data.pow(2).sum().sqrt().clamp(0, 10)
                adam_step = exp_avg / exp_avg_sq.sqrt().add(group['eps'])
                
                if group['weight_decay'] != 0:
                    adam_step.add_(p.data, alpha=group['weight_decay'])
                
                adam_norm: Union[int, float] = adam_step.pow(2).sum().sqrt()
                
                if weight_norm == 0 or adam_norm == 0:
                    trust_ratio: Union[int, float] = 1
                else:
                    trust_ratio: Union[int, float] = weight_norm / adam_norm
                
                state['weight_norm'] = weight_norm
                state['adam_norm'] = adam_norm
                state['trust_ratio'] = trust_ratio
                
                if self.adam:
                    trust_ratio: Union[int, float] = 1
                
                p.data.add_(adam_step, alpha = -step_size * trust_ratio)
                
        return loss            
                
                    
                    
if __name__ == '__main__':
    print('hemllo')