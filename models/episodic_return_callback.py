import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from typing import Dict, Any, Optional
import collections


class EpisodicReturnCallback(BaseCallback):
    """
    一个记录训练期间情节回报的回调函数
    
    这个回调会在每个情节结束时记录情节回报，并每隔log_freq步记录最近log_freq步的平均值
    """
    
    def __init__(self, log_freq: int = 5000, verbose: int = 0):
        """
        初始化情节回报回调
        
        Args:
            log_freq: 日志记录频率，每隔log_freq步记录一次平均值
            verbose: 详细程度 (0 = 无输出, 1 = 基本信息)
        """
        super().__init__(verbose)
        self.log_freq = log_freq
        
    def _on_step(self) -> bool:
        """
        在每个环境步骤后调用
        """
        # 使用stable_baselines3内置的episode信息记录机制
        
        # 检查是否需要记录日志（每log_freq步记录一次）
        if self.num_timesteps % self.log_freq == 0 and len(self.model.ep_info_buffer) > 0:
            self._log_periodic_stats()
                
        return True
    
    def _log_periodic_stats(self):
        try:
            if len(self.model.ep_info_buffer) > 0:
                # get episode returns and lengths
                episode_returns = [info['r'] for info in self.model.ep_info_buffer]
                episode_lengths = [info['l'] for info in self.model.ep_info_buffer]
                wandb.log({
                    "charts/episodic_return": np.mean(episode_returns),
                    "charts/episodic_length": np.mean(episode_lengths),
                }, step=self.num_timesteps)  # 使用 num_timesteps
        except Exception as e:
            print(f"Error logging stats: {e}")
