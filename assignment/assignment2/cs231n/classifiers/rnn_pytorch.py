import numpy as np
import torch
from ..rnn_layers_pytorch import *


class CaptioningRNN:
    """
    CaptioningRNN 使用循环神经网络，根据图像特征生成图像描述。

    RNN 接收维度为 D 的输入向量，词汇表大小为 V，处理长度为 T 的序列；
    RNN 隐藏状态的维度为 H，词向量的维度为 W，小批量大小为 N。

    注意：CaptioningRNN 不使用任何正则化。
    """

    def __init__(
        self,
        word_to_idx,
        input_dim=512,
        wordvec_dim=128,
        hidden_dim=128,
        cell_type="rnn",
        dtype=torch.float32,
    ):
        """
        创建一个新的 CaptioningRNN 实例。

        输入：
        - word_to_idx：表示词汇表的字典，包含 V 个条目；它将每个字符串映射到
          范围 [0, V) 内唯一的整数。
        - input_dim：输入图像特征向量的维度 D。
        - wordvec_dim：词向量的维度 W。
        - hidden_dim：RNN 隐藏状态的维度 H。
        - cell_type：使用的循环单元类型，可以是 'rnn' 或 'lstm'。
        - dtype：使用的数据类型；训练时使用 float32，数值梯度检查时使用
          float64。
        """
        if cell_type not in {"rnn", "lstm"}:
            raise ValueError('Invalid cell_type "%s"' % cell_type)

        self.cell_type = cell_type
        self.dtype = dtype
        self.word_to_idx = word_to_idx
        self.idx_to_word = {i: w for w, i in word_to_idx.items()}
        self.params = {}

        vocab_size = len(word_to_idx)

        self._null = word_to_idx["<NULL>"]
        self._start = word_to_idx.get("<START>", None)
        self._end = word_to_idx.get("<END>", None)

        # 初始化词向量
        self.params["W_embed"] = torch.randn(vocab_size, wordvec_dim)
        self.params["W_embed"] /= 100

        # 初始化从 CNN 图像特征到隐藏状态的投影参数
        self.params["W_proj"] = torch.randn(input_dim, hidden_dim)
        self.params["W_proj"] /= np.sqrt(input_dim)
        self.params["b_proj"] = torch.zeros(hidden_dim)

        # 初始化 RNN 参数
        dim_mul = {"lstm": 4, "rnn": 1}[cell_type]
        self.params["Wx"] = torch.randn(wordvec_dim, dim_mul * hidden_dim)
        self.params["Wx"] /= np.sqrt(wordvec_dim)
        self.params["Wh"] = torch.randn(hidden_dim, dim_mul * hidden_dim)
        self.params["Wh"] /= np.sqrt(hidden_dim)
        self.params["b"] = torch.zeros(dim_mul * hidden_dim)

        # 初始化从隐藏状态映射到词汇表分数的输出层参数
        self.params["W_vocab"] = torch.randn(hidden_dim, vocab_size)
        self.params["W_vocab"] /= np.sqrt(hidden_dim)
        self.params["b_vocab"] = torch.zeros(vocab_size)

        # 将参数转换成指定的数据类型
        for k, v in self.params.items():
            self.params[k] = v.to(self.dtype)

    def loss(self, features, captions):
        """
        计算 RNN 在训练阶段的损失。输入图像特征及其真实描述，然后使用
        RNN（或 LSTM）计算损失；所有参数的梯度由 PyTorch 自动计算。

        输入：
        - features：输入图像特征，形状为 (N, D)。
        - captions：真实描述，是形状为 (N, T + 1) 的整数数组，其中每个元素
          都满足 0 <= y[i, t] < V。

        返回：
        - loss：标量损失。
        """
        # 将 captions 切分为两部分：captions_in 包含除最后一个词之外的所有词，
        # 作为 RNN 的输入；captions_out 包含除第一个词之外的所有词，是期望 RNN
        # 生成的目标。二者错开一个位置，因为 RNN 接收第 t 个词后应预测第 t+1 个词。
        # captions_in 的第一个元素是 START 标记，captions_out 的第一个元素是描述正文
        # 中的第一个词。
        captions_in = captions[:, :-1]
        captions_out = captions[:, 1:]

        # 标记需要计入损失的位置；忽略 <NULL> 填充位置
        mask = captions_out != self._null

        # 从图像特征仿射变换到初始隐藏状态所使用的权重和偏置
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]

        # 词嵌入矩阵
        W_embed = self.params["W_embed"]

        # RNN 的输入到隐藏状态权重、隐藏状态到隐藏状态权重和偏置
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]

        # 从隐藏状态变换到词汇表分数所使用的权重和偏置
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        loss = 0.0
        ############################################################################
        # TODO：实现 CaptioningRNN 的前向传播。                                    #
        # 前向传播需要完成以下步骤：                                                #
        # (1) 对图像特征进行仿射变换，计算初始隐藏状态，结果形状应为 (N, H)。      #
        # (2) 使用词嵌入层将 captions_in 中的单词索引转换成向量，得到形状为        #
        #     (N, T, W) 的数组。                                                   #
        # (3) 根据 self.cell_type，使用普通 RNN 或 LSTM 处理输入词向量序列，       #
        #     得到所有时间步的隐藏状态，形状为 (N, T, H)。                         #
        # (4) 对隐藏状态进行时间仿射变换，计算每个时间步上词汇表中所有单词的       #
        #     分数，得到形状为 (N, T, V) 的数组。                                  #
        # (5) 使用时间 softmax 和 captions_out 计算损失，并通过上面的 mask 忽略    #
        #     输出词为 <NULL> 的位置。                                             #
        #                                                                          #       
        # 请确保实现不依赖输入张量的具体数据类型。                                 #
        #                                                                          #
        # 不需要考虑对权重或梯度进行正则化。                                       #
        #                                                                          #
        # 也不需要实现反向传播。                                                    #
        ############################################################################
        hidden_features=affine_forward(features,W_proj,b_proj)
        captions_in_embedding=word_embedding_forward(captions_in,W_embed)
        if self.cell_type=="rnn":
            h=rnn_forward(captions_in_embedding,hidden_features,Wx,Wh,b)
        elif self.cell_type=="lstm":
            h=lstm_forward(captions_in_embedding,hidden_features,Wx,Wh,b)
        scores=temporal_affine_forward(h,W_vocab,b_vocab)
        loss=temporal_softmax_loss(scores,captions_out,mask)
        ############################################################################
        #                              你的代码结束                                #
        ############################################################################

        return loss

    def sample(self, features, max_length=30):
        """
        在测试阶段执行模型的前向传播，为输入的图像特征采样生成描述。

        在每个时间步，首先嵌入当前单词，然后将词向量和前一个隐藏状态传入 RNN，
        得到下一个隐藏状态；再用隐藏状态计算词汇表中所有单词的分数，并选择分数
        最高的单词作为下一个词。初始隐藏状态由输入图像特征经过仿射变换得到，
        初始单词为 <START> 标记。

        对于 LSTM，还需要记录细胞状态；此时初始细胞状态应为零。

        输入：
        - features：形状为 (N, D) 的输入图像特征数组。
        - max_length：生成描述的最大长度 T。

        返回：
        - captions：形状为 (N, max_length) 的采样描述数组，其中每个元素都是
          范围 [0, V) 内的整数。captions 的第一个元素应是采样得到的第一个词，
          而不是 <START> 标记。
        """
        N = features.shape[0]
        captions = self._null * torch.ones((N, max_length), dtype=torch.long)

        # 取出模型参数
        W_proj, b_proj = self.params["W_proj"], self.params["b_proj"]
        W_embed = self.params["W_embed"]
        Wx, Wh, b = self.params["Wx"], self.params["Wh"], self.params["b"]
        W_vocab, b_vocab = self.params["W_vocab"], self.params["b_vocab"]

        ###########################################################################
        # TODO：实现模型在测试阶段的采样。首先对输入图像特征应用学习到的仿射变换，#
        # 初始化 RNN 的隐藏状态。输入 RNN 的第一个词应为 <START> 标记，它的值    #
        # 保存在 self._start 中。在每个时间步需要完成：                           #
        # (1) 使用学习到的词嵌入对前一个词进行嵌入。                              #
        # (2) 使用前一个隐藏状态和当前词向量执行一步 RNN，得到下一个隐藏状态。    #
        # (3) 对下一个隐藏状态应用学习到的仿射变换，得到词汇表中所有单词的分数。  #
        # (4) 选择分数最高的词作为下一个词，并将其单词索引写入 captions 中的      #
        #     对应位置。                                                          #
        #                                                                         #
        # 为了简化实现，采样到 <END> 标记后不必停止生成，但你也可以选择停止。      #
        #                                                                         #
        # 提示：这里不能使用 rnn_forward 或 lstm_forward；需要在循环中调用         #
        # rnn_step_forward 或 lstm_step_forward。                                 #
        #                                                                         #
        # 注意：此函数仍然以小批量方式工作。如果使用 LSTM，需要将初始细胞状态     #
        # 初始化为零。                                                            #
        ###########################################################################
        current_word=torch.full(
                                (N,),
                                self._start,
                                dtype=torch.long,
                                device=features.device,
        )
        h_prev=affine_forward(features,W_proj,b_proj)
        for t in range(max_length):
            current_word_embedding=word_embedding_forward(current_word,W_embed)
            h_now=rnn_step_forward(current_word_embedding,h_prev,Wx,Wh,b)
            scores=affine_forward(h_now,W_vocab,b_vocab)
            next_word=torch.argmax(scores,dim=1)
            current_word=next_word
            captions[:,t]=current_word
            h_prev=h_now
            
        ############################################################################
        #                              你的代码结束                                #
        ############################################################################
        return captions
