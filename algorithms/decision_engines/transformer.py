from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

import os
import math
import numpy as np


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

# from timeit import default_timer as timer


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Start of sentence and end of sentence
SOS, EOS = -1, -2
UNK_IDX, PAD_IDX = 1, 3


class TransformerDE(BuildingBlock):
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
                 input_vector: BuildingBlock,
                 distinct_syscalls: int,
                 epochs=300,
                 batch_size=1,
                 force_train=False,
                 model_path='Models/'):
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
        super().__init__()
        self._input_vector = input_vector
        self._dependency_list = [input_vector]
        # self._ngram_length = ngram_length
        # self._embedding_size = embedding_size
        # input dim:
        #   time_delta and return value per syscall,
        #   thread change flag per ngram
        self._distinct_syscalls = distinct_syscalls
        self._batch_size = batch_size
        self._epochs = epochs
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self._model_path = model_path \
            + f'-ep{self._epochs}' \
            + f'b{self._batch_size}'
        self._training_data = []
        self._validation_data = []
        self._state = 'build_training_data'
        self._transformer = None
        self._batch_indices = []
        self._batch_indices_val = []
        self._current_batch = []
        self._current_batch_val = []
        self._batch_counter = 0
        self._batch_counter_val = 0
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cpu")
        NUM_HEAD = 8
        NUM_TOKENS = 4
        NUM_DECODER_LAYERS = 3
        NUM_ENCODER_LAYERS = 3
        DIM_MODEL = 8
        DROPOUT = 0.1
        self.transformer = Transformer(NUM_TOKENS,
                                       NUM_HEAD,
                                       DIM_MODEL,
                                       NUM_ENCODER_LAYERS,
                                       NUM_DECODER_LAYERS,
                                       DROPOUT).to(DEVICE)

    def depends_on(self):
        return self._dependency_list

    def train_on(self, syscall: Syscall, dependencies: dict):
        """

        create training data and keep track of batch indices
        batch indices are later used for creation of batches

        Args:
            feature_list (int): list of prepared features for DE

        """
        feature_list = None
        if self._input_vector.get_id() in dependencies:
            feature_list = dependencies[self._input_vector.get_id()]
        if self._transformer is None and feature_list is not None:
            x = np.array(feature_list)
            # x = [SOS] + x + [EOS]
            # self._training_data.append([SOS] + x + [EOS])
            self._training_data.append(x)
            self._current_batch.append(self._batch_counter)
            self._batch_counter += 1
            if len(self._current_batch) == self._batch_size:
                self._batch_indices.append(self._current_batch)
                self._current_batch = []
        else:
            pass

    def val_on(self, syscall: Syscall, dependencies: dict):
        """

        create validation data and keep track of batch indices
        batch indices are later used for creation of batches

        Args:
            feature_list (int): list of prepared features for DE

        """
        feature_list = None
        if self._input_vector.get_id() in dependencies:
            feature_list = dependencies[self._input_vector.get_id()]
        if self._transformer is None and feature_list is not None:
            x = np.array(feature_list)
            self._validation_data.append(x)
            self._current_batch_val.append(self._batch_counter_val)
            self._batch_counter_val += 1
            if len(self._current_batch_val) == self._batch_size:
                self._batch_indices_val.append(self._current_batch_val)
                self._current_batch_val = []
        else:
            pass

    def _create_train_data(self, val: bool):
        if not val:
            x_tensors = Variable(torch.Tensor(self._training_data)).to(self._device)
            x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Training Shape x: {x_tensors_final.shape}")
            return SyscallFeatureDataSet(x_tensors_final)
        else:
            x_tensors = Variable(torch.Tensor(self._validation_data)).to(self._device)
            x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Validation Shape x: {x_tensors_final.shape}")
            return SyscallFeatureDataSet(x_tensors_final)

    def fit(self):
        """
        """
        if self._state == 'build_training_data':
            self._state = 'fitting'
        train_dataset = self._create_train_data(val=False)
        val_dataset = self._create_train_data(val=True)
        # for custom batches
        train_dataloader = DataLoader(train_dataset, batch_sampler=self._batch_indices)

        # val_dataloader = DataLoader(val_dataset, batch_sampler=self._batch_indices_val)
        loss_fn = nn.CrossEntropyLoss()
        learning_rate = 0.001
        optimizer = torch.optim.Adam(self.transformer.parameters(),
                                     lr=learning_rate,
                                     betas=(0.9, 0.98),
                                     eps=1e-9)
        for epoch in range(1, self._epochs+1):
            # start_time = timer()
            train_loss = self.train_loop(self.transformer, optimizer, loss_fn, train_dataloader)
            print(f"Training Loss: {train_loss:.4f}")
            # end_time = timer()
            # val_loss = self.evaluate(self.transformer)
            # print((f"Epoch: {epoch},
            # Train loss: {train_loss:.3f},
            # Val loss: {val_loss:.3f},
            # "f"Epoch time = {(end_time - start_time):.3f}s"))

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

    def calculate(self, syscall: Syscall, dependencies: dict):
        """

        remove label from feature_list and feed feature_list and hidden state into model.
        model returns probabilities of every syscall seen in training + index 0 for unknown syscall
        index of actual syscall gives predicted_prob
        Returns:
            float: anomaly score

        """

        feature_list = None
        if self._input_vector.get_id() in dependencies:
            feature_list = dependencies[self._input_vector.get_id()]
        if feature_list is not None:
            # x_tensor = Variable(torch.Tensor(np.array([feature_list[1:]])))
            # x_tensor_final = torch.reshape(x_tensor, (x_tensor.shape[0], 1, x_tensor.shape[1]))
            # actual_syscall = feature_list[0]
            anomaly_score = 0
            # prediction_logits, self._hidden = self._lstm(x_tensor_final,
            # self._hidden)
            # softmax = nn.Softmax(dim=0)
            # predicted_prob = float(softmax(prediction_logits[0])[actual_syscall])
            # anomaly_score = 1 - predicted_prob
            dependencies[self.get_id()] = anomaly_score

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

    def train_loop(self, model, opt, loss_fn, dataloader):
        """
        """

        model.train()
        total_loss = 0

        for batch in dataloader:
            print(batch)
            X = batch
            y = batch
            # X, y = batch[:, 0], batch[:, 1]
            X, y = torch.tensor(X).to(DEVICE), torch.tensor(y).to(DEVICE)
            # print(X)
            print(y)

            # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
            y_input = y[:, :-1]
            y_expected = y[:, 1:]
            print(y_input)
            print(y_expected)

            # Get mask to mask out the next words
            sequence_length = y_input.size(1)
            tgt_mask = model.get_tgt_mask(sequence_length).to(DEVICE)

            # Standard training except we pass in y_input and tgt_mask
            pred = model(X, y_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)
            loss = loss_fn(pred, y_expected)

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += loss.detach().item()

        return total_loss / len(dataloader)

    def validation_loop(self, model, loss_fn, dataloader):
        """
        """

        model.eval()
        total_loss = 0

        with torch.no_grad():
            for X,y in dataloader:
                X, y = batch[:, 0], batch[:, 1]
                # print(y)
                # X, y = torch.tensor(X, dtype=torch.long, device=DEVICE),
                # torch.tensor(y, dtype=torch.long, device=DEVICE)

                # Now we shift the tgt by one so with the <SOS> we predict the token at pos 1
                y_input = y[:, :-1]
                y_expected = y[:, 1:]

                # Get mask to mask out the next words
                sequence_length = y_input.size(1)
                tgt_mask = model.get_tgt_mask(sequence_length).to(DEVICE)

                # Standard training except we pass in y_input and src_mask
                pred = model(X, y_input, tgt_mask)

                # Permute pred to have batch size first again
                pred = pred.permute(1, 2, 0)
                loss = loss_fn(pred, y_expected)
                total_loss += loss.detach().item()

        return total_loss / len(dataloader)


class SyscallFeatureDataSet(Dataset):

    def __init__(self, X):
        self.X = X

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        _x = self.X[index]
        return _x


class Transformer(nn.Module):
    """
    Model from "A detailed guide to Pytorch's nn.Transformer() module.", by
    Daniel Melchor: https://medium.com/@danielmelchor/a-detailed-guide-to-pytorchs-nn-transformer-module-c80afbc9ffb1
    """

    # Constructor
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

    def forward(self, src, tgt, tgt_mask=None, src_pad_mask=None, tgt_pad_mask=None):
        # Src size must be (batch_size, src sequence length)
        # Tgt size must be (batch_size, tgt sequence length)

        # Embedding + positional encoding - Out size = (batch_size, sequence length, dim_model)
        src = self.embedding(src) * math.sqrt(self.dim_model)
        tgt = self.embedding(tgt) * math.sqrt(self.dim_model)
        src = self.positional_encoder(src)
        tgt = self.positional_encoder(tgt)

        # We could use the parameter batch_first=True, but our KDL version doesn't support it yet, so we permute
        # to obtain size (sequence length, batch_size, dim_model),
        src = src.permute(1, 0, 2)
        tgt = tgt.permute(1, 0, 2)

        # Transformer blocks - Out size = (sequence length, batch_size, num_tokens)
        transformer_out = self.transformer(src,
                                           tgt,
                                           tgt_mask=tgt_mask,
                                           src_key_padding_mask=src_pad_mask,
                                           tgt_key_padding_mask=tgt_pad_mask)
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
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)  # 0, 1, 2, 3, 4, 5
        division_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        # 1000^(2i/dim_model)

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
