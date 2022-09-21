
from __future__ import annotations
import torch, math, time
from torch.optim import Optimizer
from typing import Optional, Any, Callable, Union, Iterable

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


class MetroNome:
    def __init__(self, func: Callable[Any], name: Optional[str] = 'Timer') -> None:
        self.display: str = 'seconds'
        self._func: Callable[Any] = func
        self.timer_name: str = name
                
        
    def __call__(self, *args: Iterable[Any]) -> Union[float, NoneType, Any]:
        start_time: float = time.time()
        func_result: Any = self._func(*args)
        end_time: float = time.time()
        print(f'{self._func} took {end_time - start_time:.3f} {self.display}')
        return func_result
        


class AdaptiveGradientDynamicBounds(Optimizer):
    def __init__(self, parameters: Iterable[Any], learning_rate: Optional[float] = 1e-3,
                                                  betas: Optional[tuple[float, float]] = (0.9, 0.999),
                                                  final_learning_rate: Optional[float] = 0.1,
                                                  gamma: Optional[float] = 1e-3,
                                                  epsilon: Optional[float] = 1e-8,
                                                  weight_decay: Optional[Union[int, float]] = 0,
                                                  amsbound: Optional[bool] = False) -> None:
        
        self.defaults_parameters: dict[str, Any] = locals()
        self.defaults_parameters.pop('self')
        super(AdaptiveGradientDynamicBounds, self).__init__(parameters, self.defaults_parameters)
        self.base_lrs: list[Any] = list(map(lambda group: group['lr'], self.param_groups))
        
    
    
    def __setstate__(self, state: Any) -> None:
        super(AdaptiveGradientDynamicBounds, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsbound', False)
    
    
    
    @MetroNome
    def step(self, closure: Union[Callable[Any], NoneType] = None) -> Union[int, float, NoneType]:
        loss: Union[int, float, NoneType] = None
        if closure is not None:
            loss = closure()
        
        for group, base_lr in zip(self.param_groups, self.base_lrs):
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                amsbound = group['amsbound']
                state = self.state[p]
                
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                    if amsbound:
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)
                        
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsbound:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']
                
                state['step'] += 1
                
                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)
                
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                if amsbound:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out = max_exp_avg_sq)
                    denom = max_exp_avg_sq.sqrt().add_(group['eps'])
                else:
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1
                
                final_lr = group['final_lr'] * group['lr'] / base_lr
                lower_bound = final_lr * (1 - 1 / (group['gamma'] * state['step'] + 1))
                upper_bound = final_lr * (1 + 1 / (group['gamma'] * state['step']))
                step_size = torch.full_like(denom, step_size)
                step_size.div_(denom).clamp_(lower_bound, upper_bound).mul_(exp_avg)

                p.data.add_(-step_size)

        return loss
    
    
    
if __name__.__contains__('__main__'):
    print('hemllo')