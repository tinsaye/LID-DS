from algorithms.building_block import BuildingBlock
from dataloader.syscall import Syscall

import os
import math
import numpy as np

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

from timeit import default_timer as timer

from torch.nn.utils.rnn import pad_sequence
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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
                 ngram_length: int,
                 embedding_size: int,
                 epochs=300,
                 batch_size=2,
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
        self._ngram_length = ngram_length
        self._embedding_size = embedding_size
        # input dim:
        #   time_delta and return value per syscall,
        #   thread change flag per ngram
        self._batch_size = batch_size
        self._epochs = epochs
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        self._model_path = model_path \
            + f'-ep{self._epochs}' \
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
        self._transformer = None
        self._batch_indices = []
        self._batch_indices_val = []
        self._current_batch = []
        self._current_batch_val = []
        self._batch_counter = 0
        self._batch_counter_val = 0
        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._device = torch.device("cpu")

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
            return SyscallFeatureDataSet(x_tensors_final, y_tensors), y_tensors
        else:
            x_tensors = Variable(torch.Tensor(self._validation_data['x'])).to(self._device)
            y_tensors = Variable(torch.Tensor(self._validation_data['y'])).to(self._device)
            y_tensors = y_tensors.long()
            x_tensors_final = torch.reshape(x_tensors, (x_tensors.shape[0], 1, x_tensors.shape[1]))
            print(f"Validation Shape x: {x_tensors_final.shape} y: {y_tensors.shape}")
            return SyscallFeatureDataSet(x_tensors_final, y_tensors), y_tensors

    def fit(self):
        """
        """
        if self._state == 'build_training_data':
            self._state = 'fitting'
        train_dataset, y_tensors = self._create_train_data(val=False)
        val_dataset, y_tensors_val = self._create_train_data(val=True)
        # for custom batches
        train_dataloader = DataLoader(train_dataset, batch_sampler=self._batch_indices)
        for data in train_dataloader:
            print(data[0])
            break
        # val_dataloader = DataLoader(val_dataset, batch_sampler=self._batch_indices_val)
        # learning_rate = 0.001
        # optimizer = torch.optim.Adam(self.transformer.parameters(),
                                     # lr=learning_rate,
                                     # betas=(0.9, 0.98),
                                     # eps=1e-9)
        # for epoch in range(1, self._epochs+1):
            # start_time = timer()
            # train_loss = self.train_epoch(self._transformer, optimizer)
            # end_time = timer()
            # val_loss = self.evaluate(self.transformer)
            # print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))

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
            x_tensor = Variable(torch.Tensor(np.array([feature_list[1:]])))
            x_tensor_final = torch.reshape(x_tensor, (x_tensor.shape[0], 1, x_tensor.shape[1]))
            actual_syscall = feature_list[0]
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

    def train_epoch(model, optimizer):
        model.train()
        losses = 0
        SRC_LANGUAGE = de
        TGT_LANGUAGE = en
        train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

        for src, tgt in train_dataloader:
            print(src)
            print(len(src))
            print(tgt)
            print(len(tgt))
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask, src_padding_mask, tgt_padding_mask, src_padding_mask)

            optimizer.zero_grad()

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            loss.backward()

            optimizer.step()
            losses += loss.item()

        return losses / len(train_dataloader)

    def evaluate(model):
        model.eval()
        losses = 0

        val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
        loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

        for src, tgt in val_dataloader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:-1, :]

            src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

            logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

            tgt_out = tgt[1:, :]
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
            losses += loss.item()

        return losses / len(val_dataloader)


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


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None,
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


# helper Module that adds positional encoding to the token embedding
# to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])
