import torch
import torch.nn as nn

from model import Model


class MRF_NN(torch.nn.Module):
    """
    Class to initialize belief propagation model with neural parametrization of psi
    """
    def __init__(self, tags, sentence=None):
        super(MRF_NN, self).__init__()
        self.model = Model(tags)
        self.sentence = sentence

    def forward(self, pos, labs, W, psi_2):
        """
        :param pos: pos tags parameters
        :param labs: dependency label parameters
        :param W: weight matrix
        :param psi_2: message parameters
        :return: application of model to current sentence
        """
        num_labels, num_pos, n = len(labs), len(pos), len(pos[0])
        pos2 = pos.repeat((1, num_labels)).view(-1, n).repeat(num_pos, 1)
        pos1 = pos.repeat((1, num_pos * num_labels)).view(-1, n)
        labels = labs.repeat((num_pos * num_pos, 1))

        psi_1 = torch.cat([pos1, pos2, labels], 1).reshape((num_pos, num_pos, num_labels, n * 3))
        tanh = nn.Tanh()
        psi = torch.tensordot(tanh(torch.tensordot(psi_1, W, 1)), psi_2, 1)
        del pos2, pos1, labels, psi_1

        val = -self.model.log_prob(self.sentence.T, self.sentence.pos, self.sentence.m, psi)
        return val


class MRF_Lin(torch.nn.Module):
    """
    Class to initialize belief propagation model with linear parametrization of psi
    """
    def __init__(self, tags, sentence=None):
        super(MRF_Lin, self).__init__()
        self.model = Model(tags)
        self.sentence = sentence

    def forward(self, psi):
        """
        :param psi: psi parameters
        :return: application of model to current sentence
        """
        val = -self.model.log_prob(self.sentence.T, self.sentence.pos, self.sentence.m, psi)
        return val
