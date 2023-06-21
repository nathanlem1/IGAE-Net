"""
Multi-task representation learning framework based on different CNN-based and transformer-based up-to-date deep learning
architectures.
https://www.learnopencv.com/multi-label-image-classification-with-pytorch/ is useful for understanding of how multi-task
learning works.

"""

import copy
import torch
import torch.nn as nn
import torchvision.models as models


class Identity(torch.nn.Module):
    """
    Defines an Identity layer to replace the FC layer of the pretrained model
    """
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class IGAE_ResNet50(nn.Module):
    """
    IGAE (Identity, Gender and Age Estimation) model based on ResNet50.
    """

    def __init__(self, n_identity_classes, n_gender_classes, n_age_classes):
        """
        Args:
            n_identity_classes (int): number of identity classes
            n_gender_classes (int): number of gender classes
            n_age_classes (int): number of age classes
        """
        super(IGAE_ResNet50, self).__init__()
        resnet50 = models.resnet50   # from torchvision.models import resnet50, ResNet50_Weights  is also possible!
        weights = models.ResNet50_Weights.DEFAULT  # https: // pytorch.org / vision / stable / models.html
        model = resnet50(weights=weights)   # Initialize the model with the BEST available weights for transfer learning
        fc = model.fc
        model.fc = Identity()
        self.backbone = model
        last_channel = fc.in_features

        # Create separate classifiers for our outputs
        self.identity = nn.Linear(in_features=last_channel, out_features=n_identity_classes)
        self.gender = nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        self.age = nn.Linear(in_features=last_channel, out_features=n_age_classes)

    def forward(self, x):
        x = self.backbone(x)

        return {
            'identity': self.identity(x),
            'gender': self.gender(x),
            'age': self.age(x)
        }


class IGAE_DenseNet121(nn.Module):
    """
    IGAE (Identity, Gender and Age Estimation) model based on DenseNet121.
    """

    def __init__(self, n_identity_classes, n_gender_classes, n_age_classes):
        """
        Args:
            n_identity_classes (int): number of identity classes
            n_gender_classes (int): number of gender classes
            n_age_classes (int): number of age classes
        """
        super(IGAE_DenseNet121, self).__init__()
        densenet121 = models.densenet121
        weights = models.DenseNet121_Weights.DEFAULT
        model = densenet121(weights=weights)  # Initialize the model with the available weights for transfer learning
        classifier = model.classifier
        model.classifier = Identity()
        self.backbone = model
        last_channel = classifier.in_features

        # Create separate classifiers for our outputs
        self.identity = nn.Linear(in_features=last_channel, out_features=n_identity_classes)
        self.gender = nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        self.age = nn.Linear(in_features=last_channel, out_features=n_age_classes)

    def forward(self, x):
        x = self.backbone(x)

        return {
            'identity': self.identity(x),
            'gender': self.gender(x),
            'age': self.age(x)
        }


class IGAE_ConvNext_t(nn.Module):
    """
    IGAE (Identity, Gender and Age Estimation) model based on ConvNext-Tiny.
    """

    def __init__(self, n_identity_classes, n_gender_classes, n_age_classes):
        """
        Args:
            n_identity_classes (int): number of identity classes
            n_gender_classes (int): number of gender classes
            n_age_classes (int): number of age classes
        """
        super(IGAE_ConvNext_t, self).__init__()
        convnext_t = models.convnext_tiny
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT
        model = convnext_t(weights=weights)  # For transfer learning
        self.backbone = model.features  # take the model without classifier
        classifier = model.classifier
        last_channel = classifier[2].in_features  # size of the layer before classifier

        # The input for the classifier should be two-dimensional, but we will have [batch_size, channels, width, height]
        # so, let's do the spatial averaging: reduce width and height to 1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Create separate classifiers for our outputs
        self.identity = copy.deepcopy(classifier)
        self.identity[2] = nn.Linear(in_features=last_channel, out_features=n_identity_classes)
        self.gender = copy.deepcopy(classifier)
        self.gender[2] = nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        self.age = copy.deepcopy(classifier)
        self.age[2] = nn.Linear(in_features=last_channel, out_features=n_age_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = self.avgpool(x)

        return {
            'identity': self.identity(x),
            'gender': self.gender(x),
            'age': self.age(x)
        }


class IGAE_vit_b_16(nn.Module):
    """
    IGAE (Identity, Gender and Age Estimation) model based on vision transformer (ViT_B_16).
    """

    def __init__(self, n_identity_classes, n_gender_classes, n_age_classes):
        """
        Args:
            n_identity_classes (int): number of identity classes
            n_gender_classes (int): number of gender classes
            n_age_classes (int): number of age classes
        """
        super(IGAE_vit_b_16, self).__init__()
        vit_b_16 = models.vit_b_16
        weights = models.ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)    # For transfer learning
        head = model.heads.head
        model.heads.head = Identity()
        self.backbone = model
        last_channel = head.in_features

        # Create separate classifiers for our outputs
        self.identity = nn.Linear(in_features=last_channel, out_features=n_identity_classes)
        self.gender = nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        self.age = nn.Linear(in_features=last_channel, out_features=n_age_classes)

    def forward(self, x):
        x = self.backbone(x)

        return {
            'identity': self.identity(x),
            'gender': self.gender(x),
            'age': self.age(x)
        }

class IGAE_swin_t(nn.Module):
    """
    IGAE (Identity, Gender and Age Estimation) model based on swin transformer (Swin_T).
    """

    def __init__(self, n_identity_classes, n_gender_classes, n_age_classes):
        """
        Args:
            n_identity_classes (int): number of identity classes
            n_gender_classes (int): number of gender classes
            n_age_classes (int): number of age classes
        """
        super(IGAE_swin_t, self).__init__()
        swin_t = models.swin_t
        weights = models.Swin_T_Weights.DEFAULT
        model = swin_t(weights=weights)    # For transfer learning
        head = model.head
        model.head = Identity()
        self.backbone = model
        last_channel = head.in_features

        # Create separate classifiers for our outputs
        self.identity = nn.Linear(in_features=last_channel, out_features=n_identity_classes)
        self.gender = nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        self.age = nn.Linear(in_features=last_channel, out_features=n_age_classes)

    def forward(self, x):
        x = self.backbone(x)

        return {
            'identity': self.identity(x),
            'gender': self.gender(x),
            'age': self.age(x)
        }


class IGAE_maxvit_t(nn.Module):
    """
    IGAE (Identity, Gender and Age Estimation) model based on maxvit transformer (MaxViT_T).
    """

    def __init__(self, n_identity_classes, n_gender_classes, n_age_classes):
        """
        Args:
            n_identity_classes (int): number of identity classes
            n_gender_classes (int): number of gender classes
            n_age_classes (int): number of age classes
        """
        super(IGAE_maxvit_t, self).__init__()
        maxvit_t = models.maxvit_t
        weights = models.MaxVit_T_Weights.DEFAULT
        model = maxvit_t(weights=weights)    # For transfer learning
        classifier = model.classifier
        model.classifier = Identity()
        self.backbone = model
        last_channel = classifier[5].in_features

        # # Create separate classifiers for our outputs
        # self.identity = nn.Linear(in_features=last_channel, out_features=n_identity_classes)
        # self.gender = nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        # self.age = nn.Linear(in_features=last_channel, out_features=n_age_classes)

        # Create separate classifiers for our outputs
        self.identity = copy.deepcopy(classifier)
        self.identity[5] = nn.Linear(in_features=last_channel, out_features=n_identity_classes)
        self.gender = copy.deepcopy(classifier)
        self.gender[5] = nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        self.age = copy.deepcopy(classifier)
        self.age[5] = nn.Linear(in_features=last_channel, out_features=n_age_classes)

    def forward(self, x):
        x = self.backbone(x)

        return {
            'identity': self.identity(x),
            'gender': self.gender(x),
            'age': self.age(x)
        }


# This is only for exploring the internal components of IGAE_vit_b_16.
class IGAE_vit_b_16_mtl(nn.Module):
    """
    IGAE (Identity, Gender and Age Estimation) multi-task learning (mtl) model based on vision transformer (ViT_B_16).
    """

    def __init__(self, n_identity_classes, n_gender_classes, n_age_classes):
        """
        Args:
            n_identity_classes (int): number of identity classes
            n_gender_classes (int): number of gender classes
            n_age_classes (int): number of age classes
        """
        super(IGAE_vit_b_16_mtl, self).__init__()
        vit_b_16 = models.vit_b_16
        weights = models.ViT_B_16_Weights.DEFAULT
        model = vit_b_16(weights=weights)    # For transfer learning

        head = model.heads.head
        model.heads.head = Identity()
        last_channel = head.in_features

        self.ln = model.encoder.ln
        # ln = model.encoder.ln
        # self.ln_i = copy.deepcopy(ln)
        # self.ln_g = copy.deepcopy(ln)
        # self.ln_a = copy.deepcopy(ln)
        model.encoder.ln = Identity()

        layer_new = model.encoder.layers[11]
        self.layer_identity = copy.deepcopy(layer_new)
        self.layer_gender = copy.deepcopy(layer_new)
        self.layer_age = copy.deepcopy(layer_new)

        self.backbone = model  # Without the last normalization layer (ln).

        # Create separate classifiers for our outputs
        self.identity = nn.Linear(in_features=last_channel, out_features=n_identity_classes)
        self.gender = nn.Linear(in_features=last_channel, out_features=n_gender_classes)
        self.age = nn.Linear(in_features=last_channel, out_features=n_age_classes)

    def forward(self, x):

        n, c, h, w = x.shape  # n is batch size

        # (n, c, h, w) -> (n, hidden_dim, n_h, n_w). NB: hidden_dim = embedding dimension = 768
        x = self.backbone.conv_proj(x)

        # (n, hidden_dim, n_h, n_w) -> (n, hidden_dim, (n_h * n_w))
        x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[2])

        # (n, hidden_dim, (n_h * n_w)) -> (n, (n_h * n_w), hidden_dim)
        # The self attention layer expects inputs in the format (N, S, E) where S is the source sequence length, N is
        # the batch size, E is the embedding dimension.
        x = x.permute(0, 2, 1)

        # Expand the class token to the full batch
        batch_class_token = self.backbone.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.backbone.encoder(x)  # Position embedding, pos_embedding (n, 197, 768 ), is added to the input in the
        # encoder class.

        # Separate transformer layer for each task - identity, gender and age learning
        x_i = x_g = x_a = x   # For identity, gender and age
        # x_i = self.ln_i(self.layer_identity(x_i))  # Identity
        # x_g = self.ln_g(self.layer_gender(x_g))  # Gender
        # x_a = self.ln_a(self.layer_age(x_a))  # Age

        x_i = self.ln(self.layer_identity(x_i))  # Identity
        x_g = self.ln(self.layer_gender(x_g))  # Gender
        x_a = self.ln(self.layer_age(x_a))  # Age

        # Take features corresponding to 14x14 patches (196) with 768 channels.
        feature_maps_i, feature_maps_g, feature_maps_a = x_i[:, 1:], x_g[:, 1:], x_a[:, 1:]

        # Classifier "token" as used by standard language architectures
        x_i, x_g, x_a = x_i[:, 0], x_g[:, 0], x_a[:, 0]

        return {
            'identity': self.identity(x_i),
            'gender': self.gender(x_g),
            'age': self.age(x_a)
        }


# Check the model
if __name__ == '__main__':
    input_ = torch.FloatTensor(4, 3, 224, 224)

    # # ResNet50 based model
    # model = IGAE_ResNet50(100, 2, 6)
    # print(model)
    # outputs = model(input_)
    # print('Output keys:', outputs.keys())
    # print('Shape of identity output: ', outputs['identity'].shape)
    #
    # # DenseNet121 based model
    # model = IGAE_DenseNet121(100, 2, 6)
    # print(model)
    # outputs = model(input_)
    # print('Output keys:', outputs.keys())
    # print('Shape of identity output: ', outputs['identity'].shape)
    #
    # # ConvNext-T based model
    # model = IGAE_ConvNext_t(100, 2, 6)
    # print(model)
    # outputs = model(input_)
    # print('Output keys:', outputs.keys())
    # print('Shape of identity output: ', outputs['identity'].shape)

    # # vit_b_16 based model
    # model = IGAE_vit_b_16(100, 2, 6)
    # print(model)
    # outputs = model(input_)
    # print('Output keys:', outputs.keys())
    # print('Shape of identity output: ', outputs['identity'].shape)

    # swin_t based model
    model = IGAE_swin_t(100, 2, 6)
    print(model)
    outputs = model(input_)
    print('Output keys:', outputs.keys())
    print('Shape of identity output: ', outputs['identity'].shape)

    # # maxvit_t based model
    # model = IGAE_maxvit_t(100, 2, 6)
    # print(model)
    # outputs = model(input_)
    # print('Output keys:', outputs.keys())
    # print('Shape of identity output: ', outputs['identity'].shape)

    print('Done!')
