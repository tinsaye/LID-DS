import os
import math
import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

from algorithms.decision_engines.base_decision_engine import BaseDecisionEngine


class Transformer(BaseDecisionEngine):
    """

    LSTM decision engine

    Main idea:
        training:
            train prediction of next systemcall given a feature vector
        prediction:
            convert logits return value of model for every syscall to possibilties with softmax
            check predicted possibility of actual syscall and return 1-pred_pos as anomaly score


    """

    def __init__(self,
                 ngram_length: int,
                 embedding_size: int,
                 time_delta=0,
                 thread_change_flag=0,
                 return_value=0,
                 epochs=300,
                 architecture=None,
                 batch_size=2,
                 model_path='Models/',
                 force_train=False):
        """

        Args:
            ngram_length:       count of embedded syscalls
            embedding_size:     size of one embedded syscall
            extra_param:        amount of used extra parameters
            epochs:             set training epochs of LSTM
            architecture:       type of LSTM architecture
            batch_size:         set maximum batch_size
            model_path:         path to save trained Net to
            force_train:        force training of Net

        """
        self._ngram_length = ngram_length
        self._embedding_size = embedding_size
        # input dim:
        #   time_delta and return value per syscall,
        #   thread change flag per ngram
        self._input_dim = (self._ngram_length
                           * (self._embedding_size+return_value+time_delta)
                           + thread_change_flag)
        self._batch_size = batch_size
        self._epochs = epochs
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self._model_path = model_path \
            + f'n{self._ngram_length}-e{self._embedding_size}-r{bool(return_value)}' \
            + f'tcf{bool(thread_change_flag)}-td{bool(time_delta)}-ep{self._epochs}' \
            + f'b{self._batch_size}'
        self._training_data = {
            'x': [],
            'y': []
        }
        self._validation_data = {
            'x': [],
            'y': []
        }
        self._state = 'build_training_data'
        self._architecture = architecture
        self._transformer = None
        self._batch_indices = []
        self._batch_indices_val = []
        self._current_batch = []
        self._current_batch_val = []
        self._batch_counter = 0
        self._batch_counter_val = 0
        self._hidden = None
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cpu")

    def train_on(self, feature_list: list):
        """

        create training data and keep track of batch indices
        batch indices are later used for creation of batches

        Args:
            feature_list (int): list of prepared features for DE

        """
        if self._transformer is None:
            x = np.array(feature_list[:-1])
            y = np.array(feature_list[1:])
            self._training_data['x'].append(x)
            self._training_data['y'].append(y)
            self._current_batch.append(self._batch_counter)
            self._batch_counter += 1
            if len(self._current_batch) == self._batch_size:
                self._batch_indices.append(self._current_batch)
                self._current_batch = []
        else:
            pass

    def val_on(self, feature_list: list):
        """

        create validation data and keep track of batch indices
        batch indices are later used for creation of batches

        Args:
            feature_list (int): list of prepared features for DE

        """
        if self._transformer is None:
            x = np.array(feature_list[:-1])
            y = np.array(feature_list[1:])
            self._validation_data['x'].append(x)
            self._validation_data['y'].append(y)
            self._current_batch_val.append(self._batch_counter_val)
            self._batch_counter_val += 1
            if len(self._current_batch_val) == self._batch_size:
                self._batch_indices_val.append(self._current_batch_val)
                self._current_batch_val = []
        else:
            pass

    def _create_train_data(self, val: bool):
        if not val:
            x_tensors = Variable(torch.Tensor(self._training_data['x'])).to(self._device)
            y_tensors = Variable(torch.Tensor(self._training_data['y'])).to(self._device)
            y_tensors = y_tensors.long()
            x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Training Shape x: {x_tensors_final.shape} y: {y_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors_final, y_tensors)
        else:
            x_tensors = Variable(torch.Tensor(self._validation_data['x'])).to(self._device)
            y_tensors = Variable(torch.Tensor(self._validation_data['y'])).to(self._device)
            y_tensors = y_tensors.long()
            x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Validation Shape x: {x_tensors_final.shape} y: {y_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors_final, y_tensors)

    def fit(self):
        """
        """

        self._transformer = TransformerNet(
            num_tokens=4,
            dim_model=8,
            num_heads=2,
            num_encoder_layers=3,
            num_decoder_layers=4,
            dropout_p=0.1
        )
        opt = torch.optim.SGD(self._transformer.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()
        train_loss_list = []
        print("Training model")
        train_dataset = self._create_train_data(val=False)
        train_dataloader = DataLoader(train_dataset, batch_sampler=self._batch_indices)
        for epoch in range(self._epochs):
            print("-"*25, f"Epoch {epoch + 1}", "-"*25)
            train_loss = 0
            for batch in train_dataloader:
                print(batch)
                X, y = batch[0], batch[1]
                print(X, y)
                # X, y = torch.tensor(X).to(self._device), torch.tensor(y).to(self._device)
                # shift target by one
                y_input = y[:, :-1]
                y_expected = y[:, 1:]

                # get mask to mask out next words
                sequence_length = y_input.size(1)
                tgt_mask = self._transformer.get_tgt_mask(sequence_length).to(self._device)

                # standard training except passing y_input and tgt_mask
                pred = self._transformer(X, y_input, tgt_mask)

                # permute to have batch size first
                pred = pred.permute(1, 2, 0)
                loss = loss_fn(pred, y_expected)

                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.detach().item()
            train_loss_list += [train_loss]

        return train_loss_list

    def predict(self, feature_list: list) -> float:
        """

        remove label from feature_list and feed feature_list and hidden state into model.
        model returns probabilities of every syscall seen in training + index 0 for unknown syscall
        index of actual syscall gives predicted_prob
        1 - predicted_prob is anomaly score

        Returns:
            float: anomaly score

        """
        pass

    def new_recording(self, val: bool = False):
        """

        while creation of dataset:
            cut batch after recording end
        while fitting and detecting
            reset hidden state

        """
        if self._transformer is None and self._state == 'build_training_data':
            if not val:
                if len(self._current_batch) > 0:
                    self._batch_indices.append(self._current_batch)
                    self._current_batch = []
            else:
                if len(self._current_batch_val) > 0:
                    self._batch_indices_val.append(self._current_batch_val)
                    self._current_batch_val = []
        elif self._state == 'fitting':
            self._hidden = None
        else:
            pass


class TransformerNet(nn.Module):
    def __init__(
        self,
        num_tokens,
        dim_model,
        num_heads,
        num_encoder_layers,
        num_decoder_layers,
        dropout_p,
    ):
        super().__init__()

          # INFO
        self.model_type = "Transformer"
        self.dim_model = dim_model

        # LAYERS
        self.positional_encoder = PositionalEncoding(
            dim_model=dim_model, dropout_p=dropout_p, max_len=5000
        )
        self.embedding = nn.Embedding(num_tokens, dim_model)
        self.transformer = nn.Transformer(
            d_model=dim_model,
            nhead=num_heads,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout_p,
        )
        self.out = nn.Linear(dim_model, num_tokens)

    def forward(
        self,
        src,
        tgt,
        tgt_mask=None,
        src_pad_mask=None,
        tgt_pad_mask=None
    ):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # we permute to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src, tgt)
        out = self.out(transformer_out)

        return out

    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1)  # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf'))  # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0))  # Convert ones to 0

        # EX for size=5:
        # [[0., -inf, -inf, -inf, -inf],
        #  [0.,   0., -inf, -inf, -inf],
        #  [0.,   0.,   0., -inf, -inf],
        #  [0.,   0.,   0.,   0., -inf],
        #  [0.,   0.,   0.,   0.,   0.]]

        return mask

    def create_pad_mask(self, matrix: torch.tensor, pad_token: int) -> torch.tensor:
        # If matrix = [1,2,3,0,0,0] where pad_token=0, the result mask is
        # [False, False, False, True, True, True]
        return (matrix == pad_token)


class PositionalEncoding(nn.Module):
    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        # Modified version from: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
        # max_len determines how far the position can have an effect on a token (window)

        # Info
        self.dropout = nn.Dropout(dropout_p)

        # Encoding - From formula
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0,
                                      max_len,
                                      dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float()
                                  * (-math.log(10000.0)) / dim_model)  # 1000^(2i/dim_model)

        # PE(pos, 2i) = sin(pos/1000^(2i/dim_model))
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)

        # PE(pos, 2i + 1) = cos(pos/1000^(2i/dim_model))
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)

        # Saving buffer (same as parameter without gradients needed)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding: torch.tensor) -> torch.tensor:
        # Residual connection + pos encoding
        return self.dropout(token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class SyscallFeatureDataSet(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        if len(self.X) != len(self.Y):
            raise Exception("The length of X does not match length of Y")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        _y = self.Y[index]
        return _x, _y
