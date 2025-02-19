# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Deepspeech2 ASR Model"""
import paddle
from paddle import nn
from src_deepspeech2x.models.ds2.rnn import RNNStack

from paddlespeech.s2t.models.ds2.conv import ConvStack
from paddlespeech.s2t.modules.ctc import CTCDecoder
from paddlespeech.s2t.utils import layer_tools
from paddlespeech.s2t.utils.checkpoint import Checkpoint
from paddlespeech.s2t.utils.log import Log
logger = Log(__name__).getlog()

__all__ = ['DeepSpeech2Model', 'DeepSpeech2InferModel']


class CRNNEncoder(nn.Layer):
    def __init__(self,
                 feat_size,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=1024,
                 use_gru=False,
                 share_rnn_weights=True):
        super().__init__()
        self.rnn_size = rnn_size
        self.feat_size = feat_size  # 161 for linear
        self.dict_size = dict_size

        self.conv = ConvStack(feat_size, num_conv_layers)

        i_size = self.conv.output_height  # H after conv stack
        self.rnn = RNNStack(
            i_size=i_size,
            h_size=rnn_size,
            num_stacks=num_rnn_layers,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)

    @property
    def output_size(self):
        return self.rnn_size * 2

    def forward(self, audio, audio_len):
        """Compute Encoder outputs

        Args:
            audio (Tensor): [B, Tmax, D]
            text (Tensor): [B, Umax]
            audio_len (Tensor): [B]
            text_len (Tensor): [B]
        Returns:
            x (Tensor): encoder outputs, [B, T, D]
            x_lens (Tensor): encoder length, [B]
        """
        # [B, T, D]  -> [B, D, T]
        audio = audio.transpose([0, 2, 1])
        # [B, D, T] -> [B, C=1, D, T]
        x = audio.unsqueeze(1)
        x_lens = audio_len

        # convolution group
        x, x_lens = self.conv(x, x_lens)
        x_val = x.numpy()

        # convert data from convolution feature map to sequence of vectors
        #B, C, D, T = paddle.shape(x)  # not work under jit
        x = x.transpose([0, 3, 1, 2])  #[B, T, C, D]
        #x = x.reshape([B, T, C * D])  #[B, T, C*D]  # not work under jit
        x = x.reshape([0, 0, -1])  #[B, T, C*D]

        # remove padding part
        x, x_lens = self.rnn(x, x_lens)  #[B, T, D]
        return x, x_lens


class DeepSpeech2Model(nn.Layer):
    """The DeepSpeech2 network structure.

    :param audio_data: Audio spectrogram data layer.
    :type audio_data: Variable
    :param text_data: Transcription text data layer.
    :type text_data: Variable
    :param audio_len: Valid sequence length data layer.
    :type audio_len: Variable
    :param masks: Masks data layer to reset padding.
    :type masks: Variable
    :param dict_size: Dictionary size for tokenized transcription.
    :type dict_size: int
    :param num_conv_layers: Number of stacking convolution layers.
    :type num_conv_layers: int
    :param num_rnn_layers: Number of stacking RNN layers.
    :type num_rnn_layers: int
    :param rnn_size: RNN layer size (dimension of RNN cells).
    :type rnn_size: int
    :param use_gru: Use gru if set True. Use simple rnn if set False.
    :type use_gru: bool
    :param share_rnn_weights: Whether to share input-hidden weights between
                              forward and backward direction RNNs.
                              It is only available when use_gru=False.
    :type share_weights: bool
    :return: A tuple of an output unnormalized log probability layer (
             before softmax) and a ctc cost layer.
    :rtype: tuple of LayerOutput
    """

    def __init__(self,
                 feat_size,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=1024,
                 use_gru=False,
                 share_rnn_weights=True,
                 blank_id=0):
        super().__init__()
        self.encoder = CRNNEncoder(
            feat_size=feat_size,
            dict_size=dict_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_size,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights)
        assert (self.encoder.output_size == rnn_size * 2)

        self.decoder = CTCDecoder(
            odim=dict_size,  # <blank> is in  vocab
            enc_n_units=self.encoder.output_size,
            blank_id=blank_id,  # first token is <blank>
            dropout_rate=0.0,
            reduction=True,  # sum
            batch_average=True)  # sum / batch_size

    def forward(self, audio, audio_len, text, text_len):
        """Compute Model loss

        Args:
            audio (Tensor): [B, T, D]
            audio_len (Tensor): [B]
            text (Tensor): [B, U]
            text_len (Tensor): [B]

        Returns:
            loss (Tensor): [1]
        """
        eouts, eouts_len = self.encoder(audio, audio_len)
        loss = self.decoder(eouts, eouts_len, text, text_len)
        return loss

    @paddle.no_grad()
    def decode(self, audio, audio_len):
        # decoders only accept string encoded in utf-8

        # Make sure the decoder has been initialized
        eouts, eouts_len = self.encoder(audio, audio_len)
        probs = self.decoder.softmax(eouts)
        batch_size = probs.shape[0]
        self.decoder.reset_decoder(batch_size=batch_size)
        self.decoder.next(probs, eouts_len)
        trans_best, trans_beam = self.decoder.decode()
        return trans_best

    @classmethod
    def from_pretrained(cls, dataloader, config, checkpoint_path):
        """Build a DeepSpeech2Model model from a pretrained model.
        Parameters
        ----------
        dataloader: paddle.io.DataLoader

        config: yacs.config.CfgNode
            model configs

        checkpoint_path: Path or str
            the path of pretrained model checkpoint, without extension name

        Returns
        -------
        DeepSpeech2Model
            The model built from pretrained result.
        """
        model = cls(feat_size=dataloader.collate_fn.feature_size,
                    dict_size=len(dataloader.collate_fn.vocab_list),
                    num_conv_layers=config.num_conv_layers,
                    num_rnn_layers=config.num_rnn_layers,
                    rnn_size=config.rnn_layer_size,
                    use_gru=config.use_gru,
                    share_rnn_weights=config.share_rnn_weights)
        infos = Checkpoint().load_parameters(
            model, checkpoint_path=checkpoint_path)
        logger.info(f"checkpoint info: {infos}")
        layer_tools.summary(model)
        return model

    @classmethod
    def from_config(cls, config):
        """Build a DeepSpeec2Model from config
        Parameters

        config: yacs.config.CfgNode
            config
        Returns
        -------
        DeepSpeech2Model
            The model built from config.
        """
        model = cls(feat_size=config.feat_size,
                    dict_size=config.dict_size,
                    num_conv_layers=config.num_conv_layers,
                    num_rnn_layers=config.num_rnn_layers,
                    rnn_size=config.rnn_layer_size,
                    use_gru=config.use_gru,
                    share_rnn_weights=config.share_rnn_weights,
                    blank_id=config.blank_id)
        return model


class DeepSpeech2InferModel(DeepSpeech2Model):
    def __init__(self,
                 feat_size,
                 dict_size,
                 num_conv_layers=2,
                 num_rnn_layers=3,
                 rnn_size=1024,
                 use_gru=False,
                 share_rnn_weights=True,
                 blank_id=0):
        super().__init__(
            feat_size=feat_size,
            dict_size=dict_size,
            num_conv_layers=num_conv_layers,
            num_rnn_layers=num_rnn_layers,
            rnn_size=rnn_size,
            use_gru=use_gru,
            share_rnn_weights=share_rnn_weights,
            blank_id=blank_id)

    def forward(self, audio, audio_len):
        """export model function

        Args:
            audio (Tensor): [B, T, D]
            audio_len (Tensor): [B]

        Returns:
            probs: probs after softmax
        """
        eouts, eouts_len = self.encoder(audio, audio_len)
        probs = self.decoder.softmax(eouts)
        return probs, eouts_len

    def export(self):
        static_model = paddle.jit.to_static(
            self,
            input_spec=[
                paddle.static.InputSpec(
                    shape=[None, None, self.encoder.feat_size],
                    dtype='float32'),  # audio, [B,T,D]
                paddle.static.InputSpec(shape=[None],
                                        dtype='int64'),  # audio_length, [B]
            ])
        return static_model
