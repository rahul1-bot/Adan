
from __future__ import annotations
import torch, math
from torch.optim import Optimizer
from typing import Optional, Any, Callable, Union, Mapping

__author_info__: dict[str, Union[str, list[str]]] = {
    'Name': 'Rahul Sawhney',
    'Education': 'Amity University, Noida : Btech CSE (Final Year)',
    'Mail': [
        'sawhney.rahulofficial@outlook.com', 
        'rahulsawhney321@gmail.com'
    ]
}

#@: date = 23 September, 2022
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

class LR_Scheduler_Interface(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer: torch.optim.optimizer, learning_rate: float) -> None:
        self.optimizer = optimizer
        self.learning_rate = learning_rate
    
    def __repr__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(['Module', 'Name', 'ObjectID'], [self.__module__, type(self).__name__, hex(id(self))])
        })
        
    def __str__(self) -> str(dict[str, Any]):
        return str({
            x: y for x, y in zip(
                ['Optimizer', 'Learning Rate'], [self.optimizer, self.learning_rate] 
            )
        })
        
    def step(self, *args: Iterable[Any], **kwargs: Mapping[Any, Any]) -> NotImplementedError:
        raise NotImplementedError
    
    @staticmethod
    def set_learning_rate(optimizer: torch.optim.optimizer, learning_rate: float) -> None:
        for g in optimizer.param_groups:
            g['lr'] = learning_rate
    
    def get_learning_rate(self) -> float:
        for g in self.optimizer.param_groups:
            return g['lr']

    

class TripleJuncture_LRScheduler(LR_Scheduler_Interface):
    def __init__(self, optimizer: torch.optim.optimizer, init_learning_rate: float, peak_learning_rate: float,
                                                                                    final_learning_rate: float,
                                                                                    init_lr_scale: float,
                                                                                    final_lr_scale: float,
                                                                                    warmup_steps: int, 
                                                                                    hold_steps: int, 
                                                                                    decay_steps: int, 
                                                                                    total_steps: int) -> None:
        
        if not isinstance(warmup_steps, int) or not isinstance(total_steps, int):
            raise ValueError
        
        self.param_dict: dict[str, Any] = {
            key : value for key, value in locals().items() if not key in [
                'self', 'init_lr_scale', 'final_lr_scale', 'total_steps'
            ]
        }
        super(TripleJuncture_LRScheduler, self).__init__(optimizer, init_learning_rate)
        
        if self.param_dict['warmup_steps'] != 0:
            warmup_rate: float = (self.param_dict['peak_learning_rate'] - self.param_dict['init_learning_rate']) / self.param_dict[warmup_steps]
        else: 
            warmup_rate: float = 0.0        
        
        decay_factor: float = -math.log(self.param_dict['final_learning_rate']) / self.param_dict['decay_steps']
        learning_rate: float = self.param_dict['init_learning_rate']
        
        new_params: dict[str, Any] = {
            'warmup_rate' :  warmup_rate, 'decay_factor' : decay_factor, 
            'learning_rate' : learning_rate, 'update_steps' : 0 
        }
        self.param_dict.update(new_params)
        
    
    
    
    def step(self, val_loss: Optional[Union[torch.FloatTensor, NoneType]] = None) -> float:
        stage, steps_in_stage = float('nan'), float('nan')
        if self.param_dict['update_steps'] < self.param_dict['warmup_steps']:
            stage, steps_in_stage = 0, self.param_dict['update_steps']
        
        offset: int = self.param_dict['warmup_steps']
        
        if self.param_dict['update_steps'] < offset + self.param_dict['hold_steps']:
            stage, steps_in_stage = 1, self.param_dict['update_steps'] - offset
        
        offset += self.param_dict['hold_steps']
        
        if self.param_dict['update_steps'] <= offset + self.param_dict['decay_steps']:
            stage, steps_in_stage = 2, self.param_dict['update_steps'] - offset
        
        offset += self.param_dict['decay_steps']
        
        if math.isnan(stage) or math.isnan(steps_in_stage):
            stage, steps_in_stage = 3, self.param_dict['update_steps'] - offset
        
        
        if stage == 0:
            self.param_dict['learning_rate'] = self.param_dict['init_learning_rate'] + self.param_dict['warmup_rate'] * steps_in_stage
        
        elif stage == 1:
            self.param_dict['learning_rate'] = self.param_dict['peak_learning_rate']
        
        elif stage == 2:
            self.param_dict['learning_rate'] = self.param_dict['peal_learning_rate'] * math.exp(-self.param_dict['decay_factor'] * steps_in_stage)
        
        elif stage == 3:
            self.param_dict['learning_rate'] = self.param_dict['final_learning_rate']
        else:
            raise ValueError
        
        self.set_learning_rate(self.param_dict['optimizer'], self.param_dict['learning_rate'])
        self.param_dict['update_steps'] += 1
        
        return self.param_dict['learning_rate']