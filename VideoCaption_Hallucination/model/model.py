# coding=utf-8
"""
    @project: zero-shot-video-to-text-main
    @Author：no-zjc
    @file： model.py
    @date：2023/10/30 16:33
"""


import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer


# 加载预训练的语言模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# 自定义设置模型的参数





class CustomGPT2Model(nn.Module):
    def __init__(self, gpt2_model, external_model):
        super(CustomGPT2Model, self).__init__()
        self.gpt2_model = gpt2_model
        self.external_model = external_model
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, input_ids, attention_mask, external_input):
        # 使用预训练语言模型进行正向传播
        gpt_outputs = self.gpt2_model(input_ids=input_ids, attention_mask=attention_mask)
        gpt_hidden_states = gpt_outputs.last_hidden_state

        # 处理外部模型的输出和注意力掩码
        external_outputs = self.external_model(external_input)
        external_hidden_states = external_outputs.hidden_states
        external_attention_mask = external_outputs.attention_mask

        # 在预训练模型的输出和外部模型的输出之间进行交叉注意力操作
        combined_hidden_states, _ = self.cross_attention(query=gpt_hidden_states, key=external_hidden_states, value=external_hidden_states, attn_mask=external_attention_mask)

        # 将交叉注意力的输出与预训练模型的输出相结合，并返回最终结果
        combined_hidden_states = combined_hidden_states + gpt_hidden_states
        return combined_hidden_states

# 创建带有交叉注意力层的自定义语言模型
custom_model = CustomGPT2Model(model, external_model)


