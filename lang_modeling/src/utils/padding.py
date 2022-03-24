import torch
from torch.autograd import Variable


def pad(sequences, max_length, pad_value=0):
    """Pads a list of sequences.
    Args:
        sequences: A list of sequences to be padded.
        max_length: The length to pad to.
        pad_value: The value used for padding.
    Returns:
        A list of padded sequences.
    """
    out = []
    for sequence in sequences:
        padded = sequence + [0]*(max_length - len(sequence))
        out.append(padded)
    return out


def collate_annotations(batch):
    """Function used to collate data returned by CoNLLDataset."""
    # Get inputs, targets, and lengths.
    inputs, targets = zip(*batch)
    lengths = [len(x) for x in inputs]
    # Sort by length.
    sort = sorted(zip(inputs, targets, lengths),
                  key=lambda x: x[2],
                  reverse=True)
    inputs, targets, lengths = zip(*sort)
    # Pad.
    max_length = max(lengths)
    inputs = pad(inputs, max_length)
    targets = pad(targets, max_length)
    # Transpose.
    inputs = list(map(list, zip(*inputs)))
    targets = list(map(list, zip(*targets)))
    # Convert to PyTorch variables.
    inputs = Variable(torch.LongTensor(inputs))
    targets = Variable(torch.LongTensor(targets))
    lengths = Variable(torch.LongTensor(lengths))
    if torch.cuda.is_available():
        inputs = inputs.cuda()
        targets = targets.cuda()
        lengths = lengths.cuda()
    return inputs, targets, lengths

