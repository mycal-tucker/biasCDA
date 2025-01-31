import argparse

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from lang_modeling.src.conll_dataset import CoNLLDataset, collate_annotations
from lang_modeling.src.language_model import LanguageModel

FLAGS = None


def main(_):
    # Load configuration.
    with open(FLAGS.config, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize CoNLL dataset.
    dataset = CoNLLDataset(fname=config['data']['train'], target='lm', token_vocab=config['data'].get('vocab'))
    # And save the vocab dataset
    dataset.token_vocab.save('data/vocab.pk')

    # Initialize model.
    language_model = LanguageModel(
        vocab_size=len(dataset.token_vocab),
        embedding_dim=config['model']['embedding_dim'],
        hidden_size=config['model']['hidden_size'],
        num_layers=config['model']['num_layers'])
    if torch.cuda.is_available():
        language_model = language_model.cuda()

    # Initialize loss function. NOTE: Manually setting weight of padding to 0.
    weight = torch.ones(len(dataset.token_vocab))
    weight[0] = 0
    if torch.cuda.is_available():
        weight = weight.cuda()
    loss_function = torch.nn.NLLLoss(weight)
    optimizer = torch.optim.Adam(language_model.parameters())

    # Main training loop.
    data_loader = DataLoader(
        dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        collate_fn=collate_annotations)
    dev_data = DataLoader(
        CoNLLDataset(fname=config['data']['dev'], target='lm', token_vocab=config['data'].get('vocab')),
        batch_size=config['training']['batch_size'],
        shuffle=False,
        collate_fn=collate_annotations)
    best_dev_loss = 100
    losses = []
    i = 0
    for epoch in range(config['training']['num_epochs']):
        print("Epoch", epoch)
        for batch in data_loader:
            inputs, targets, lengths = batch
            optimizer.zero_grad()
            outputs, _ = language_model(inputs, lengths=lengths)

            outputs = outputs.view(-1, len(dataset.token_vocab))
            targets = targets.view(-1)

            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())
            if (i % 100) == 0:
                average_loss = np.mean(losses)
                losses = []
                print('Iteration %i - Loss: %0.6f' % (i, average_loss))
            if (i % 1000) == 0:
                torch.save(language_model, config['data']['checkpoint'])
            i += 1
        # Run eval after every!? epoch
        with torch.no_grad():
            dev_losses = []
            for batch in dev_data:
                inputs, targets, lengths = batch
                optimizer.zero_grad()
                outputs, _ = language_model(inputs, lengths=lengths)

                outputs = outputs.view(-1, len(dataset.token_vocab))
                targets = targets.view(-1)
                loss = loss_function(outputs, targets)
                dev_losses.append(loss.item())
            dev_loss = np.mean(dev_losses)
            print("Dev loss", dev_loss)
            # Early stopping?
            # if dev_loss > best_dev_loss:
            #     break
            if dev_loss < best_dev_loss:
                best_dev_loss = dev_loss
    torch.save(language_model, config['data']['checkpoint'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True,
                        help='Path to configuration file.')
    FLAGS, _ = parser.parse_known_args()

    main(_)

