import torch
import torch.nn as nn

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_of_classes, label_smoothing: float = 0.1):
        super(LabelSmoothingLoss, self).__init__()

        self.num_of_classes = num_of_classes
        self.label_smoothing = label_smoothing
        self.confidence = 1 - self.label_smoothing

        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.KLDivLoss(reduction='batchmean')
        self.ce_criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, targets, mask):
        """
        :param outputs (Float tensor): batch_size X num_of_classes
        :param targets: (Long tensor): batch_size X 1
        :param targets: (Long tensor): batch_size X (num_of_classes + number of docs)
        :return:
        """
        batch_size, class_num = outputs.size()
        target_mask = mask
        n_class_num = torch.sum(target_mask, dim = 1) #batch_size X 1
        n_class_num_matrix = n_class_num.unsqueeze(1).repeat(1, self.num_of_classes)
        smoothed_targets = self.label_smoothing / (n_class_num_matrix - 1) * target_mask
        targets_flat = targets.view(batch_size)
        smoothed_targets.scatter_(1, targets_flat.unsqueeze(1), self.confidence)

        outputs_log_softmax = self.log_softmax(outputs)
        loss = self.criterion(outputs_log_softmax, smoothed_targets)
        return loss