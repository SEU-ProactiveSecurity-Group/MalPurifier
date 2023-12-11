"""
代表检测对抗性恶意软件的抽象类
"""

class DetectorTemplate(object):
    def __init__(self):
        self.tau = None                     # 阈值变量
        self.is_detector_enabled = True     # 表示检测器是否启用的标志

    def forward(self, x):
        """
        类预测与密度估计
        """
        raise NotImplementedError

    def get_threshold(self):
        """
        计算拒绝异常值的阈值
        """
        raise NotImplementedError

    def get_tau_sample_wise(self):
        """
        获取每个样本的tau值
        """
        raise NotImplementedError

    def indicator(self):
        """
        返回一个布尔标志向量，指示是否拒绝一个样本
        """
        raise NotImplementedError
