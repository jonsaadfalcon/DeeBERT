

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss

from .modeling_roberta import RobertaEmbeddings
from .modeling_highway_bert_ner import BertModel, BertPreTrainedModel, entropy, HighwayException
from .configuration_roberta import RobertaConfig


ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-pytorch_model.bin",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-pytorch_model.bin",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-pytorch_model.bin",
}


class RobertaModel(BertModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    Examples::
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)
        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaModel, self).__init__(config)

        self.embeddings = RobertaEmbeddings(config)
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value



def calculate_loss(given_logits, given_mask, given_labels, number_of_labels):

    #print("Logits and labels shapes")
    #print(given_logits.shape)
    #print(given_mask.shape)
    #print(given_labels.shape)

    loss = None
    if given_labels is not None:
        loss_fct = nn.CrossEntropyLoss()
        # Only keep active parts of the loss
        if given_mask is not None:
            active_loss = given_mask.view(-1) == 1
            active_logits = given_logits.view(-1, number_of_labels)
            active_labels = torch.where(
                active_loss, given_labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(given_labels)
            )
            loss = loss_fct(active_logits, active_labels)
        else:
            loss = loss_fct(given_logits.view(-1, number_of_labels), given_labels.view(-1))

    return loss


class RobertaForTokenClassification(BertPreTrainedModel):

    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(RobertaForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels
        self.num_layers = config.num_hidden_layers

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_layer=-1, train_highway=False):

        exit_layer = self.num_layers
        try:
            outputs = self.roberta(input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds)

            #pooled_output = outputs[1]

            #pooled_output = self.dropout(pooled_output)
            #logits = self.classifier(pooled_output)
            #outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
            #print("total outputs")
            #print(len(outputs))
            #print("shape of each output component")
            #for i in range(0, len(outputs)):
            #    print(type(outputs[i]))
            #    print(len(outputs[i]))

            #print("Shape of outputs[2]")
            #print(len(outputs[2][0]))

            dropout_output = self.dropout(outputs[0])
            logits = self.classifier(dropout_output)
            outputs = (logits,) + outputs[2:]


        except HighwayException as e:
            outputs = e.message
            exit_layer = e.exit_layer
            logits = outputs[0]

        if not self.training:
            original_entropy = entropy(logits, attention_mask, 0, "original")
            highway_entropy = []
            highway_logits_all = []
        if labels is not None:
            
            loss = calculate_loss(logits, attention_mask, labels, self.config.num_labels)

            # work with highway exits
            highway_losses = []
            for highway_exit in outputs[-1]:
                
                highway_logits = highway_exit[0]
                if not self.training:
                    highway_logits_all.append(highway_logits)
                    highway_entropy.append(highway_exit[2])
                
                highway_loss = calculate_loss(highway_logits, attention_mask, labels, self.config.num_labels)

                highway_losses.append(highway_loss)

            if train_highway:
                outputs = (sum(highway_losses[:-1]),) + outputs
                # exclude the final highway, of course
            else:
                outputs = (loss,) + outputs
        if not self.training:
            outputs = outputs + ((original_entropy, highway_entropy), exit_layer)
            if output_layer >= 0:
                outputs = (outputs[0],) + \
                          (highway_logits_all[output_layer],) + \
                          outputs[2:]  ## use the highway of the last layer

        return outputs  # (loss), logits, (hidden_states), (attentions), entropy

