import torch
import torch.nn as nn

class PracticalEasyFocusedLoss(nn.Module):
    """
    实用的关注易分样本损失函数
    解决梯度消失问题
    """
    def __init__(self, alpha=0.5, gamma=2.0, temperature=2.0, epsilon=1e-7):
        """
        Args:
            alpha: 正类权重 [0,1]
            gamma: 聚焦参数，控制对易分样本的关注度
            temperature: 温度参数，平滑权重分布（>1使权重分布更均匀）
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.temperature = temperature
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        probs = torch.softmax(logits / self.temperature, dim=1)  # 温度缩放
        p = probs[:, 1]
        p = torch.clamp(p, self.epsilon, 1 - self.epsilon)
        
        # 使用 softplus 避免梯度消失
        def safe_log(x):
            return torch.log(x + self.epsilon)
        
        # 计算损失
        loss = torch.where(
            targets == 1,
            -self.alpha * torch.pow(p, self.gamma) * safe_log(p),
            -(1 - self.alpha) * torch.pow(1 - p, self.gamma) * safe_log(1 - p)
        )
        
        # 添加小的 L2 正则化项，防止梯度完全消失
        reg = 0.01 * torch.mean(torch.pow(p - 0.5, 2))
        
        return loss.mean() + reg

class HybridEasyFocusedLoss(nn.Module):
    """
    混合标准CE和易分样本损失的函数
    保证模型始终有梯度信号
    """
    def __init__(self, alpha=0.5, gamma=2.0, ce_weight=0.5, epsilon=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce_weight = ce_weight  # 交叉熵损失权重
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        p = probs[:, 1]
        p = torch.clamp(p, self.epsilon, 1 - self.epsilon)
        
        # 易分样本损失
        easy_loss = torch.where(
            targets == 1,
            -self.alpha * torch.pow(p, self.gamma) * torch.log(p),
            -(1 - self.alpha) * torch.pow(1 - p, self.gamma) * torch.log(1 - p)
        )
        
        # 标准交叉熵损失（保证梯度）
        ce_loss = torch.where(
            targets == 1,
            -torch.log(p),
            -torch.log(1 - p)
        )
        
        # 混合损失
        loss = self.ce_weight * ce_loss + (1 - self.ce_weight) * easy_loss
        
        return loss.mean()
    
class LogSpaceEasyFocusedLoss(nn.Module):
    """
    在对数空间计算损失，避免数值过小
    """
    def __init__(self, alpha=0.5, gamma=2.0, epsilon=1e-7):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
    
    def forward(self, logits, targets):
        probs = torch.softmax(logits, dim=1)
        p = probs[:, 1]
        p = torch.clamp(p, self.epsilon, 1 - self.epsilon)
        
        # 在对数空间计算
        log_p = torch.log(p)
        log_1_minus_p = torch.log(1 - p)
        
        # 对数空间的损失
        loss = torch.where(
            targets == 1,
            -self.alpha * torch.exp(self.gamma * log_p) * log_p,
            -(1 - self.alpha) * torch.exp(self.gamma * log_1_minus_p) * log_1_minus_p
        )
        
        return loss.mean()