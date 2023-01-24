import pdb 
import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F
import itertools 

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from memory import BigState

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class T5Model(nn.Module):
    def __init__(self, name_or_path):
        super(T5Model, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(name_or_path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(name_or_path)
        # self.q_value_head = nn.Linear(self.model.config.d_model, 1)

    def convert_state_to_input(self, state: BigState) -> str:
        """ convert State into formatted string for T5"""
        def sanitize_str(inStr):
            out = inStr.replace("\n", " ")
            out = out.replace("\t", " ")
            return out

        task_desc = state.description
        cur_obs = state.obs
        look = state.look 
        inv = state.inventory
        prev_action = state.prev_action
        prev_obs = state.prev_obs
        task_desc, cur_obs, look, inv, prev_action, prev_obs = map(sanitize_str, [task_desc, cur_obs, look, inv, prev_action, prev_obs])
        out_str = task_desc + ' </s> ' + cur_obs + ' ' + inv + ' ' + look + ' </s> <extra_id_0>' + ' </s> ' + prev_action + ' </s> ' + prev_obs + ' </s>'
        return out_str

    def forward(self, state_batch, act_batch):
        # state = BigState(*zip(*state_batch))
        # This is number of admissible commands in each element of the batch
        act_sizes = [len(a) for a in act_batch]

        # Combine next actions into one long list
        act_batch = list(itertools.chain.from_iterable(act_batch))

        inputs = [self.convert_state_to_input(state) for state in state_batch]
        # input is single state 
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)
        # valid acts is list of possible actions
        valid_acts = self.tokenizer(act_batch, return_tensors="pt", padding=True, truncation=True).to(device) 
        valid_acts = valid_acts.input_ids
        # repeat input for each valid action
        inputs = {k: v.repeat(len(valid_acts), 1) for k, v in inputs.items()}
        outputs = self.model(**inputs, labels=valid_acts, output_hidden_states=True)

        generations = self.model.generate(inputs['input_ids'], num_beams=1, num_return_sequences=1, early_stopping=True)
        # decode generations
        generations = self.tokenizer.batch_decode(generations, skip_special_tokens=True)
        pdb.set_trace()

        # bsz, seq_len, hidden_dim
        hidden_reps = outputs.decoder_hidden_states[-1] 
        # use mean pool
        pooled_rep = hidden_reps.mean(dim=1)
        # bsz, 1
        # q_values = self.q_value_head(pooled_rep) 

        # bsz, seq_len, vocab_size
        logits = outputs.logits
        bsz, seq_len, vocab_size = logits.shape
        logits = logits.reshape(bsz * seq_len, vocab_size)
        valid_acts = valid_acts.reshape(bsz * seq_len)
        # compute loss 
        loss = F.cross_entropy(logits, valid_acts, reduction="none")
        loss = loss.reshape(bsz, seq_len)
        loss = loss.sum(dim=1, keepdim=True)
        # re-rank by loss (still a DAG) 
        # lowest loss wins 
        # q_values = q_values / loss

        # just use loss 
        q_values = 1/loss 
        q_values = q_values.squeeze(-1)
        return q_values.split(act_sizes)

    def act(self, input, possible_actions, sample=False):
        act_values = self(input, possible_actions)
        # TODO: Constrain actions to possible actions
        # detokenize
        # TODO: sample adds random action 
        if sample:
            act_probs = [F.softmax(vals, dim=0) for vals in act_values]
            act_idxs = [torch.multinomial(probs, num_samples=1).item() \
                        for probs in act_probs]
        else:
            act_idxs = [vals.argmax(dim=0).item() for vals in act_values]

        return act_idxs, act_values
        