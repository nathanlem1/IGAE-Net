import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
cross_entropy = CrossEntropyLoss(label_smoothing=0.1)  # Label smoothing can be applied here


def get_loss(net_output, ground_truth):
    """ Computes loss
    Args:
        net_output: predicted output labels
        ground_truth: ground-truth labels
    Return:
        total loss and loss for each attribute: identity, gender and age

    """

    # identity_loss = F.cross_entropy(net_output['identity'], ground_truth['identity_labels'])
    # gender_loss = F.cross_entropy(net_output['gender'], ground_truth['gender_labels'])
    # age_loss = F.cross_entropy(net_output['age'], ground_truth['age_labels'])

    identity_loss = cross_entropy(net_output['identity'], ground_truth['identity_labels'])
    gender_loss = cross_entropy(net_output['gender'], ground_truth['gender_labels'])
    age_loss = cross_entropy(net_output['age'], ground_truth['age_labels'])

    loss = identity_loss + gender_loss + age_loss

    return loss, {'identity': identity_loss, 'gender': gender_loss, 'age': age_loss}