import torch
from torch import Tensor
from torch.nn.modules.loss import _Loss

from monai.networks.layers.utils import get_act_layer
from monai.utils import LossReduction
from monai.utils.enums import StrEnum
from typing import Union

class KLDivergenceLoss:
    """
    A class for computing the Kullback-Leibler divergence loss.
    """
    
    def __call__(self, z_mu: Tensor, z_sigma: Tensor) -> Tensor:
        """
        Computes the KL divergence loss for the given parameters.

        Args:
            z_mu (Tensor):  The mean of the distribution.
            z_sigma (Tensor): The standard deviation of the distribution.

        Returns:
            Tensor: The computed KL divergence loss, averaged over the batch size.
        """

        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]

class AdversarialCriterions(StrEnum):
    BCE = "bce"
    HINGE = "hinge"
    LEAST_SQUARE = "least_squares"



class CustomAdversarialLoss(_Loss):
    """
    Calculates an adversarial loss on a Patch Discriminator or a Multi-scale Patch Discriminator.
    Warning: due to the possibility of using different criterions, the output of the discrimination
    mustn't be passed to a final activation layer. That is taken care of internally within the loss.

    Args:
        reduction: {``"none"``, ``"mean"``, ``"sum"``}
            Specifies the reduction to apply to the output. Defaults to ``"mean"``.

            - ``"none"``: no reduction will be applied.
            - ``"mean"``: the sum of the output will be divided by the number of elements in the output.
            - ``"sum"``: the output will be summed.

        criterion: which criterion (hinge, least_squares or bce) you want to use on the discriminators outputs.
            Depending on the criterion, a different activation layer will be used. Make sure you don't run the outputs
            through an activation layer prior to calling the loss.
        no_activation_leastsq: if True, the activation layer in the case of least-squares is removed.
    """

    def __init__(
        self,
        reduction: Union[LossReduction, str] = LossReduction.MEAN,
        criterion: str = AdversarialCriterions.LEAST_SQUARE,
        no_activation_leastsq: bool = False,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(reduction=LossReduction(reduction))

        if criterion.lower() not in list(AdversarialCriterions):
            raise ValueError(
                "Unrecognised criterion entered for Adversarial Loss. Must be one in: %s"
                % ", ".join(AdversarialCriterions)
            )

        # Depending on the criterion, a different activation layer is used.
        self.real_label = 1.0
        self.fake_label = 0.0
        self.loss_fct: _Loss
        if criterion == AdversarialCriterions.BCE:
            self.activation = get_act_layer("SIGMOID")
            self.loss_fct = torch.nn.BCELoss(reduction=reduction)
        elif criterion == AdversarialCriterions.HINGE:
            self.activation = get_act_layer("TANH")
            self.fake_label = -1.0
        elif criterion == AdversarialCriterions.LEAST_SQUARE:
            if no_activation_leastsq:
                self.activation = None
            else:
                self.activation = get_act_layer(name=("LEAKYRELU", {"negative_slope": 0.05}))
            self.loss_fct = torch.nn.MSELoss(reduction=reduction)

        self.criterion = criterion
        self.reduction = reduction
        self.label_smoothing = label_smoothing

    def get_target_tensor(self, input: torch.Tensor, target_is_real: bool) -> torch.Tensor:
        """
        Gets the ground truth tensor for the discriminator depending on whether the input is real or fake.

        Args:
            input: input tensor from the discriminator (output of discriminator, or output of one of the multi-scale
            discriminator). This is used to match the shape.
            target_is_real: whether the input is real or wannabe-real (1s) or fake (0s).
        Returns:
        """
        filling_label = self.real_label if target_is_real else self.fake_label
        label_tensor = torch.tensor(1).fill_(filling_label).type(input.type()).to(input[0].device)
        if self.label_smoothing > 0: #randomly perturb the label for values +- self.label_smoothing with values between -1 and 1
            label_tensor = label_tensor + torch.rand_like(label_tensor) * 2 * self.label_smoothing - self.label_smoothing
        label_tensor.requires_grad_(False)
        return label_tensor.expand_as(input)

    def get_zero_tensor(self, input: torch.Tensor) -> torch.Tensor:
        """
        Gets a zero tensor.

        Args:
            input: tensor which shape you want the zeros tensor to correspond to.
        Returns:
        """

        zero_label_tensor = torch.tensor(0).type(input[0].type()).to(input[0].device)
        zero_label_tensor.requires_grad_(False)
        return zero_label_tensor.expand_as(input)

    def forward(
        self, 
        input: Union[torch.Tensor, list],
        target_is_real: bool,
        for_discriminator: bool
    ) -> Union[torch.Tensor, list[torch.Tensor]]:
        """

        Args:
            input: output of Multi-Scale Patch Discriminator or Patch Discriminator; being a list of tensors
                or a tensor; they shouldn't have gone through an activation layer.
            target_is_real: whereas the input corresponds to discriminator output for real or fake images
            for_discriminator: whereas this is being calculated for discriminator or generator loss. In the last
                case, target_is_real is set to True, as the generator wants the input to be dimmed as real.
        Returns: if reduction is None, returns a list with the loss tensors of each discriminator if multi-scale
            discriminator is active, or the loss tensor if there is just one discriminator. Otherwise, it returns the
            summed or mean loss over the tensor and discriminator/s.

        """

        if not for_discriminator and not target_is_real:
            target_is_real = True  # With generator, we always want this to be true!
            warnings.warn(
                "Variable target_is_real has been set to False, but for_discriminator is set"
                "to False. To optimise a generator, target_is_real must be set to True."
            )

        if not isinstance(input, list):
            input = [input]
        target_ = []
        for _, disc_out in enumerate(input):
            if self.criterion != AdversarialCriterions.HINGE:
                target_.append(self.get_target_tensor(disc_out, target_is_real))
            else:
                target_.append(self.get_zero_tensor(disc_out))

        # Loss calculation
        loss_list = []
        for disc_ind, disc_out in enumerate(input):
            if self.activation is not None:
                disc_out = self.activation(disc_out)
            if self.criterion == AdversarialCriterions.HINGE and not target_is_real:
                loss_ = self._forward_single(-disc_out, target_[disc_ind])
            else:
                loss_ = self._forward_single(disc_out, target_[disc_ind])
            loss_list.append(loss_)

        loss: torch.Tensor | list[torch.Tensor]
        if loss_list is not None:
            if self.reduction == LossReduction.MEAN:
                loss = torch.mean(torch.stack(loss_list))
            elif self.reduction == LossReduction.SUM:
                loss = torch.sum(torch.stack(loss_list))
            else:
                loss = loss_list
        return loss

    def _forward_single(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        forward: torch.Tensor
        if self.criterion == AdversarialCriterions.BCE or self.criterion == AdversarialCriterions.LEAST_SQUARE:
            forward = self.loss_fct(input, target)
        elif self.criterion == AdversarialCriterions.HINGE:
            minval = torch.min(input - 1, self.get_zero_tensor(input))
            forward = -torch.mean(minval)
        return forward