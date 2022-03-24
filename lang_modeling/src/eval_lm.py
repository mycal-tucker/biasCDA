import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from lang_modeling.src.conll_dataset import CoNLLDataset, collate_annotations
import numpy as np

def do_eval():
    language_model = torch.load(model_path)
    # Load eval dataset
    dataset = CoNLLDataset(fname=dataset_path, target='lm', token_vocab=vocab_path)

    if torch.cuda.is_available():
        language_model = language_model.cuda()
    weight = torch.ones(len(dataset.token_vocab))
    weight[0] = 0
    if torch.cuda.is_available():
        weight = weight.cuda()
    loss_function = torch.nn.NLLLoss(weight)

    data_loader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_annotations)

    losses = []
    for batch in data_loader:
        inputs, targets, lengths = batch
        outputs, _ = language_model(inputs, lengths=lengths)

        outputs = outputs.view(-1, len(dataset.token_vocab))
        targets = targets.view(-1)

        loss = loss_function(outputs, targets)
        losses.append(loss.item())

    print("Mean loss", np.mean(losses))


def inference(sentences):
    language_model = torch.load(model_path)
    dataset = CoNLLDataset(fname=dataset_path, target='lm', token_vocab=vocab_path)
    for sentence in sentences:
        # Convert words to id tensor.
        ids = [[dataset.token_vocab.word2id(x)] for x in sentence]
        ids = Variable(torch.LongTensor(ids))
        if torch.cuda.is_available():
            ids = ids.cuda()
        # Get model output.
        output, _ = language_model(ids)
        _, preds = torch.max(output, dim=2)
        if torch.cuda.is_available():
            preds = preds.cpu()
        preds = preds.data.view(-1).numpy()
        pred_words = [dataset.token_vocab.id2word(x) for x in preds]
        for word, pred_word in zip(sentence, pred_words):
            print('%s - %s' % (word, pred_word))
        print()


if __name__ == '__main__':
    vocab_path = 'data/vocab.pk'
    model_path = 'data/language_model_standard.pt'
    # model_path = 'data/language_model_reinflected.pt'
    # dataset_path = 'data/fr-test.conllu'
    dataset_path = 'data/french_reinflected_test.conllu'
    inference([["L'", "homme", "travaille", "à", "l'", "hôpital", "comme"],
              ["La", "femme", "travaille", "à", "l'", "hôpital", "comme"]])
    do_eval()