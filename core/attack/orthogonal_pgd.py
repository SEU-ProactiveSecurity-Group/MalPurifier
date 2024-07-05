"""
Evading Adversarial Example Detection Defenses with Orthogonal Projected Gradient Descent
Codes are adapted from https://github.com/v-wangg/OrthogonalPGD
"""

import torch
import torch.nn.functional as F

from core.attack import PGD
from tools.utils import get_x0, round_x
from config import logging, ErrorHandler

logger = logging.getLogger('core.attack.orthogonal_pgd')
logger.addHandler(ErrorHandler)


class OrthogonalPGD(PGD):
    """
    Projected gradient descent (ascent).

    Parameters
    ---------
    @param norm, 'l2' or 'linf'
    @param project_detector: if True, take gradients of g onto f
    @param project_classifier: if True, take gradients of f onto g
    @param k, if not None, take gradients of g onto f at every kth step
    @param use_random, Boolean,  whether use random start point
    @param rounding_threshold, float, a threshold for rounding real scalars
    @param is_attacker, Boolean, play the role of attacker (note: the defender conducts adversarial training)
    @param manipulation_x, manipulations
    @param omega, the indices of interdependent apis corresponding to each api
    @param device, 'cpu' or 'cuda'
    """

    def __init__(self, norm, project_detector=False, project_classifier=False, k=None,
                 use_random=False, rounding_threshold=0.5,
                 is_attacker=True, manipulation_x=None, omega=None, device=None):
        super(OrthogonalPGD, self).__init__(norm, use_random, rounding_threshold, is_attacker,
                                            False, 1.0, manipulation_x, omega, device)
        self.k = k
        self.project_detector = project_detector
        self.project_classifier = project_classifier

    def _perturb(self,
                 model,
                 x,
                 label=None,
                 steps=10,
                 step_length=1.,
                 ):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of iterations
        @param step_length: float, the step length in each iteration
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x.clone().detach()
        batch_size = x.shape[0]

        assert hasattr(
            model, 'is_detector_enabled'), 'Expected an adversary detector'
        model.eval()

        # 遍历预定的步骤数来扰动输入数据
        for t in range(steps):

            # 如果是第一次迭代并且设置了 use_random 标志，
            # 使用 get_x0 函数可能会向对抗样本添加一些随机噪声
            if t == 0 and self.use_random:
                adv_x = get_x0(
                    adv_x, rounding_threshold=self.round_threshold, is_sample=True)

            # 将对抗样本转换为 PyTorch 变量，这样在反向传播期间
            # 我们可以相对于它计算梯度
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)

            # 对模型进行前向传播，获取分类任务 (logits_classifier) 和对抗检测任务 (logits_detector) 的 logits
            logits_classifier, logits_detector = model.forward(var_adv_x)

            # 计算分类器预测的 logits 与真实标签之间的交叉熵损失。
            # 这度量了模型的预测与实际标签的匹配程度
            ce = torch.mean(F.cross_entropy(
                logits_classifier, label, reduction='none'))

            # 反向传播此损失以计算损失相对于模型参数和输入对抗样本的梯度
            ce.backward(retain_graph=True)

            # 从计算图中分离出计算的梯度并存储它们。
            # 这些梯度指示了损失相对于每个输入像素的变化方向和速率
            grad_classifier = var_adv_x.grad.detach().data

            # 使用自定义函数可能转换或处理梯度
            grad_classifier = self.trans_grads(grad_classifier, adv_x)

            # 在计算下一组梯度之前清除 var_adv_x 的梯度，以防止累积
            var_adv_x.grad = None

            # 计算探测器的损失。负号意味着我们可能试图
            # 最大化这个损失（例如，使对抗样本更难以检测）
            loss_detector = -torch.mean(logits_detector)

            # 反向传播探测器的损失来计算其梯度
            loss_detector.backward()

            # 为探测器分离出从计算图中计算的梯度
            grad_detector = var_adv_x.grad.detach().data

            grad_detector = self.trans_grads(grad_detector, adv_x)

            if self.project_detector:
                # using Orthogonal Projected Gradient Descent
                # projection of gradient of detector on gradient of classifier
                # then grad_d' = grad_d - (project grad_d onto grad_c)
                grad_detector_proj = grad_detector - torch.bmm(
                    (torch.bmm(grad_detector.view(batch_size, 1, -1), grad_classifier.view(batch_size, -1, 1))) / (
                        1e-20 + torch.bmm(grad_classifier.view(batch_size, 1, -1),
                                          grad_classifier.view(batch_size, -1, 1))).view(-1, 1, 1),
                    grad_classifier.view(batch_size, 1, -1)).view(grad_detector.shape)
            else:
                grad_detector_proj = grad_detector

            if self.project_classifier:
                # using Orthogonal Projected Gradient Descent
                # projection of gradient of detector on gradient of classifier
                # then grad_c' = grad_c - (project grad_c onto grad_d)
                grad_classifier_proj = grad_classifier - torch.bmm(
                    (torch.bmm(grad_classifier.view(batch_size, 1, -1), grad_detector.view(batch_size, -1, 1))) / (
                        1e-20 + torch.bmm(grad_detector.view(batch_size, 1, -1),
                                          grad_detector.view(batch_size, -1, 1))).view(-1, 1, 1),
                    grad_detector.view(batch_size, 1, -1)).view(grad_classifier.shape)
            else:
                grad_classifier_proj = grad_classifier

            # has_attack_succeeded = (logits_classifier.argmax(1) == 0.)[:, None].float()
            disc_logits_classifier, _1 = model.forward(round_x(adv_x))
            disc_logits_classifier[range(
                batch_size), 0] = disc_logits_classifier[range(batch_size), 0] - 20
            has_attack_succeeded = (disc_logits_classifier.argmax(1) == 0.)[
                :, None].float()  # customized label

            if self.k:
                # take gradients of g onto f every kth step
                if t % self.k == 0:
                    grad = grad_detector_proj
                else:
                    grad = grad_classifier_proj
            else:
                grad = grad_classifier_proj * (
                    1. - has_attack_succeeded) + grad_detector_proj * has_attack_succeeded

            # if torch.any(torch.isnan(grad)):
            #     print(torch.mean(torch.isnan(grad)))
            #     print("ABORT")
            #     break
            if self.norm == 'linf':
                perturbation = torch.sign(grad)
            elif self.norm == 'l2':
                l2norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
                perturbation = torch.minimum(
                    torch.tensor(1., dtype=x.dtype, device=x.device),
                    grad / l2norm
                )
                perturbation = torch.where(
                    torch.isnan(perturbation), 0., perturbation)
                perturbation = torch.where(
                    torch.isinf(perturbation), 1., perturbation)
            elif self.norm == 'l1':
                val, idx = torch.abs(grad).topk(int(1. / step_length), dim=-1)
                perturbation = F.one_hot(
                    idx, num_classes=adv_x.shape[-1]).sum(dim=1)
                perturbation = torch.sign(grad) * perturbation
                # if self.is_attacker:
                #     perturbation += (
                #             torch.any(perturbation[:, self.api_flag] < 0, dim=-1,
                #                       keepdim=True) * nonexist_api)
            else:
                raise ValueError("Expect 'l2', 'linf' or 'l1' norm.")
            adv_x = torch.clamp(adv_x + perturbation *
                                step_length, min=0., max=1.)
        # round
        return round_x(adv_x)

    def _perturb_dae(self,
                     predict_model,
                     dae_model,
                     x,
                     label=None,
                     steps=10,
                     step_length=1.,
                     ):
        """
        perturb node feature vectors

        Parameters
        -----------
        @param model, a victim model
        @param x: torch.FloatTensor, node feature vectors (each represents the occurrences of apis in a graph) with shape [batch_size, vocab_dim]
        @param label: torch.LongTensor, ground truth labels
        @param steps: Integer, maximum number of iterations
        @param step_length: float, the step length in each iteration
        """
        if x is None or x.shape[0] <= 0:
            return []
        adv_x = x.clone().detach()
        batch_size = x.shape[0]

        # assert hasattr(predict_model, 'is_detector_enabled'), 'Expected an adversary detector'
        predict_model.eval()
        dae_model.eval()

        # 遍历预定的步骤数来扰动输入数据
        for t in range(steps):

            # 如果是第一次迭代并且设置了 use_random 标志，
            # 使用 get_x0 函数可能会向对抗样本添加一些随机噪声
            if t == 0:
                adv_x = get_x0(
                    adv_x, rounding_threshold=self.round_threshold, is_sample=True)

            # 将对抗样本转换为 PyTorch 变量，这样在反向传播期间
            # 我们可以相对于它计算梯度
            var_adv_x = torch.autograd.Variable(adv_x, requires_grad=True)

            # 对模型进行前向传播，获取分类任务 (logits_classifier) 和对抗检测任务 (logits_detector) 的 logits
            logits_classifier = predict_model.forward(var_adv_x)

            # 计算分类器预测的 logits 与真实标签之间的交叉熵损失。
            # 这度量了模型的预测与实际标签的匹配程度
            ce = torch.mean(F.cross_entropy(
                logits_classifier, label, reduction='none'))

            # 反向传播此损失以计算损失相对于模型参数和输入对抗样本的梯度
            ce.backward(retain_graph=True)

            # 从计算图中分离出计算的梯度并存储它们。
            # 这些梯度指示了损失相对于每个输入像素的变化方向和速率
            grad_classifier = var_adv_x.grad.detach().data

            # 使用自定义函数可能转换或处理梯度
            grad_classifier = self.trans_grads(grad_classifier, adv_x)

            # 在计算下一组梯度之前清除 var_adv_x 的梯度，以防止累积
            var_adv_x.grad = None

            # 计算探测器的损失。负号意味着我们可能试图
            # 最大化这个损失（例如，使对抗样本更难以检测）

            var_adv_x = var_adv_x.to(torch.float32).to(dae_model.device)

            # 使用DAE模型清洗对抗样本
            Purified_adv_x_batch = dae_model(var_adv_x).to(torch.float64)
            var_adv_x = var_adv_x.to(torch.float64).to(dae_model.device)

            # 确保var_adv_x可以计算梯度
            var_adv_x = var_adv_x.clone().detach().requires_grad_(True)

            # 使用DAE的损失函数
            total_loss = dae_model.loss_function(
                var_adv_x, Purified_adv_x_batch, label, predict_model)

            loss_detector = -torch.mean(total_loss)

            # 反向传播该损失以计算梯度
            loss_detector.backward()

            # 分离出计算的梯度
            if var_adv_x.grad is not None:
                grad_detector = var_adv_x.grad.detach().data
            else:
                raise ValueError(
                    "var_adv_x does not have gradients. Check the loss computation.")

            # 使用自定义函数可能转换或处理梯度
            grad_detector = self.trans_grads(grad_detector, adv_x)

            if self.project_detector:
                # using Orthogonal Projected Gradient Descent
                # projection of gradient of detector on gradient of classifier
                # then grad_d' = grad_d - (project grad_d onto grad_c)
                grad_detector_proj = grad_detector - torch.bmm(
                    (torch.bmm(grad_detector.view(batch_size, 1, -1), grad_classifier.view(batch_size, -1, 1))) / (
                        1e-20 + torch.bmm(grad_classifier.view(batch_size, 1, -1),
                                          grad_classifier.view(batch_size, -1, 1))).view(-1, 1, 1),
                    grad_classifier.view(batch_size, 1, -1)).view(grad_detector.shape)
            else:
                grad_detector_proj = grad_detector

            if self.project_classifier:
                # using Orthogonal Projected Gradient Descent
                # projection of gradient of detector on gradient of classifier
                # then grad_c' = grad_c - (project grad_c onto grad_d)
                grad_classifier_proj = grad_classifier - torch.bmm(
                    (torch.bmm(grad_classifier.view(batch_size, 1, -1), grad_detector.view(batch_size, -1, 1))) / (
                        1e-20 + torch.bmm(grad_detector.view(batch_size, 1, -1),
                                          grad_detector.view(batch_size, -1, 1))).view(-1, 1, 1),
                    grad_detector.view(batch_size, 1, -1)).view(grad_classifier.shape)
            else:
                grad_classifier_proj = grad_classifier

            # has_attack_succeeded = (logits_classifier.argmax(1) == 0.)[:, None].float()
            disc_logits_classifier = predict_model.forward(round_x(adv_x))
            disc_logits_classifier[range(
                batch_size), 0] = disc_logits_classifier[range(batch_size), 0] - 20
            has_attack_succeeded = (disc_logits_classifier.argmax(1) == 0.)[
                :, None].float()  # customized label

            if self.k:
                # take gradients of g onto f every kth step
                if t % self.k == 0:
                    grad = grad_detector_proj
                else:
                    grad = grad_classifier_proj
            else:
                grad = grad_classifier_proj * (
                    1. - has_attack_succeeded) + grad_detector_proj * has_attack_succeeded

            # if torch.any(torch.isnan(grad)):
            #     print(torch.mean(torch.isnan(grad)))
            #     print("ABORT")
            #     break
            if self.norm == 'linf':
                perturbation = torch.sign(grad)
            elif self.norm == 'l2':
                l2norm = torch.linalg.norm(grad, dim=-1, keepdim=True)
                perturbation = torch.minimum(
                    torch.tensor(1., dtype=x.dtype, device=x.device),
                    grad / l2norm
                )
                perturbation = torch.where(
                    torch.isnan(perturbation), 0., perturbation)
                perturbation = torch.where(
                    torch.isinf(perturbation), 1., perturbation)
            elif self.norm == 'l1':
                val, idx = torch.abs(grad).topk(int(1. / step_length), dim=-1)
                perturbation = F.one_hot(
                    idx, num_classes=adv_x.shape[-1]).sum(dim=1)
                perturbation = torch.sign(grad) * perturbation
                # if self.is_attacker:
                #     perturbation += (
                #             torch.any(perturbation[:, self.api_flag] < 0, dim=-1,
                #                       keepdim=True) * nonexist_api)
            else:
                raise ValueError("Expect 'l2', 'linf' or 'l1' norm.")
            adv_x = torch.clamp(adv_x + perturbation *
                                step_length, min=0., max=1.)
        # round
        return round_x(adv_x)

    def perturb(self, model, x, label=None,
                steps=10,
                step_length=1.,
                verbose=False):
        """
        Enhance the attack.
        """
        # Asserting to ensure steps and step_length are non-negative
        assert steps >= 0 and step_length >= 0
        # Set the model to evaluation mode
        model.eval()
        # Clone the input x to create adversarial examples
        adv_x = x.detach().clone()
        with torch.no_grad():
            # Get the loss and completion status for the current sample
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
        # If all samples are done, return immediately
        if torch.all(done):
            return adv_x
        # Perturb the samples that are not yet done
        pert_x = self._perturb(model, adv_x[~done], label[~done],
                               steps,
                               step_length
                               )
        # Update the perturbed samples in adv_x
        adv_x[~done] = pert_x
        with torch.no_grad():
            # Get the loss and completion status again
            _, done = self.get_loss(model, adv_x, label, self.lambda_)
            # If verbose is enabled, log the attack effectiveness
            if verbose:
                logger.info(
                    f"pgd {self.norm}: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        # Return the perturbed samples
        return adv_x

    def perturb_dae(self, predict_model, purifier, x, label=None,
                    steps=10,
                    step_length=1.,
                    verbose=False,
                    oblivion=False):
        """
        Enhance the attack.
        """
        # Asserting to ensure steps and step_length are non-negative
        assert steps >= 0 and step_length >= 0
        # Set the model to evaluation mode
        predict_model.eval()
        purifier.eval()
        # Clone the input x to create adversarial examples
        adv_x = x.detach().clone()
        with torch.no_grad():
            # Get the loss and completion status for the current sample
            purified_adv = purifier(
                adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(
                predict_model, purified_adv, label, self.lambda_)
        # If all samples are done, return immediately
        if torch.all(done):
            return adv_x
        # Perturb the samples that are not yet done
        pert_x = self._perturb_dae(predict_model, purifier, adv_x[~done], label[~done],
                                   steps,
                                   step_length
                                   )
        # Update the perturbed samples in adv_x
        adv_x[~done] = pert_x
        with torch.no_grad():
            # Get the loss and completion status again
            purified_adv = purifier(
                adv_x.detach().clone().float()).to(torch.double)
            _, done = self.get_loss(
                predict_model, purified_adv, label, self.lambda_)
            # If verbose is enabled, log the attack effectiveness
            if verbose:
                logger.info(
                    f"pgd {self.norm}: attack effectiveness {done.sum().item() / done.size()[0] * 100:.3f}%.")
        # Return the perturbed samples
        return adv_x

    def trans_grads(self, gradients, adv_features):
        # 查找允许的位置。
        # 1. 只有在特征值从 1 减少和从 0 增加的情况下是被允许的
        #    1.1 API 插入
        # 寻找可插入的位置。这些位置的特征值小于或等于 0.5 且大于或等于 0
        pos_insertion = (adv_features <= 0.5) * 1 * (adv_features >= 0.)

        # 只有当梯度是正的（即值应该增加）且该位置允许插入时，我们才考虑这些梯度
        grad4insertion = (gradients >= 0) * pos_insertion * gradients

        #    2 API 移除
        # 寻找可移除的位置。这些位置的特征值大于 0.5
        pos_removal = (adv_features > 0.5) * 1

        # 只有当梯度是负的（即值应该减少），且该位置允许移除和满足某些其他条件(self.manipulation_x)时，我们才考虑这些梯度
        grad4removal = (gradients < 0) * (pos_removal &
                                          self.manipulation_x) * gradients

        # 将两组梯度结合在一起并返回
        return grad4removal + grad4insertion
