class BertTaggerForMultiSpanQA(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits, ) + outputs[:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs


class RobertaTaggerForMultiSpanQA(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits, ) + outputs[:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)
            outputs = (loss,) + outputs

        return outputs


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class TaggerPlusForMultiSpanQA(BertPreTrainedModel):
    def __init__(self, config, structure_lambda, span_lambda):
        super().__init__(config, structure_lambda, span_lambda)
        self.structure_lambda = structure_lambda
        self.span_lambda = span_lambda
        self.label2id= config.label2id
        self.num_labels = config.num_labels
        self.max_spans = 21
        self.max_pred_spans = 30
        self.H = config.hidden_size

        self.dense = nn.Linear(self.H, self.H)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(self.H, config.num_labels)
        self.num_span_outputs = nn.Sequential(nn.Linear(self.H, 64),nn.ReLU(),nn.Linear(64, 1))
        self.structure_outputs = nn.Sequential(nn.Linear(self.H, 128),nn.ReLU(),nn.Linear(128, 6))

        config.num_attention_heads=6 # for span encoder
        intermediate_size=1024
        self.span_encoder = BertLayer(config)

        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        num_span=None,
        structure=None,
    ):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )

        sequence_output = outputs[0]
        pooled_output = outputs[1]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        # gather and pool span hidden
        B = pooled_output.size(0) # batch_size
        pred_spans = torch.zeros((B, self.max_pred_spans, self.H)).to(logits)
        pred_spans[:,0,:] = pooled_output # init the cls token use the bert cls token
        span_mask = torch.zeros((B, self.max_pred_spans)).to(logits)
        pred_labels = torch.argmax(logits, dim=-1)

        for b in range(B):
            s_pred_labels = pred_labels[b]
            s_sequence_output = sequence_output[b]
            indexes = [[]]
            flag=False
            for i in range(len(s_pred_labels)):
                if s_pred_labels[i] == self.label2id['B']: # B
                    indexes.append([i])
                    flag=True
                if s_pred_labels[i] == self.label2id['I'] and flag: # I
                    indexes[-1].append(i)
                if s_pred_labels[i] == self.label2id['O']: # O
                    flag=False
            indexes = indexes[:self.max_pred_spans]

            for i,index in enumerate(indexes):
                if i == 0:
                    span_mask[b,i] = 1
                    continue
                s_span = s_sequence_output[index[0]:index[-1]+1,:]
                s_span = torch.mean(s_span, dim=0) # mean pooling
                pred_spans[b,i,:] = s_span
                span_mask[b,i] = 1

        # encode span
        span_mask = span_mask[:,None,None,:] # extend for attention
        span_x = self.span_encoder(pred_spans, span_mask)[0]
        pooled_span_cls = span_x[:,0]
        pooled_span_cls = torch.tanh(self.dense(pooled_span_cls))

        num_span_logits = self.num_span_outputs(pooled_span_cls)
        structure_logits = self.structure_outputs(pooled_span_cls)

        outputs = (logits, num_span_logits, ) + outputs[:]
        if labels is not None: # for train
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            active_loss = attention_mask.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)
            active_labels = torch.where(
                active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
            )
            loss = loss_fct(active_logits, active_labels)

            # num_span regression
            loss_mse = MSELoss()
            num_span=num_span.type(torch.float) / self.max_spans
            num_span_loss = loss_mse(num_span_logits.view(-1), num_span.view(-1))
            num_span_loss *= self.span_lambda
            # structure classification
            loss_focal = FocalLoss(gamma=0.5)
            structure_loss = loss_focal(structure_logits.view(-1, 6), structure.view(-1))
            structure_loss *= self.structure_lambda
            loss = loss + num_span_loss + structure_loss

            outputs = (loss, ) + outputs

        return outputs
