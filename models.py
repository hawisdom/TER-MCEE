import torch
import torch.nn as nn
from data_process import event_roles
torch.set_printoptions(profile="full")

class Event_Type(nn.Module):
    def __init__(self, args,role_nums):
        super(Event_Type, self).__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)

        in_dim = 2*args.hidden_size+args.event_type_embedding_dim

        self.layer_norm = nn.LayerNorm(in_dim)

        last_hidden_size = in_dim
        layers = [nn.Linear(last_hidden_size, args.final_hidden_size), nn.LeakyReLU()]
        for _ in range(args.num_mlps - 1):
            layers += [nn.Linear(args.final_hidden_size, args.final_hidden_size), nn.LeakyReLU()]
        self.fcs = nn.Sequential(*layers)
        self.fc_final = nn.Linear(args.final_hidden_size,role_nums)

    def forward(self,token_feature,event_type_features):
        event_type_features = event_type_features.repeat(token_feature.shape[0], 1)
        token_event_input = torch.cat([token_feature, event_type_features], dim=1)

        token_event_input = self.layer_norm(token_event_input)

        out = self.fcs(token_event_input)
        logit = self.fc_final(out)

        return logit


class TER_MCEE(nn.Module):
    def __init__(self,args,pos_tag_num,dep_tag_num,sen_pos_tag_num,word_pos_tag_num,event_id_tag_num):
        super(TER_MCEE, self).__init__()
        self.args = args

        num_embeddings, embed_dim = args.token_embedding.shape
        self.embed = nn.Embedding(num_embeddings, embed_dim)
        self.embed.weight = nn.Parameter(args.token_embedding)

        self.sen_pos_embed = nn.Embedding(sen_pos_tag_num, args.sen_pos_embedding_dim)
        self.word_pos_embed = nn.Embedding(word_pos_tag_num, args.word_pos_embedding_dim)
        self.pos_embed = nn.Embedding(pos_tag_num, args.pos_embedding_dim)
        self.dep_embed = nn.Embedding(dep_tag_num, args.dep_embedding_dim)
        self.event_type_embed = nn.Embedding(args.event_type_nums, args.event_type_embedding_dim)
        self.event_embed = nn.Embedding(event_id_tag_num,args.event_embedding_dim)

        self.dropout = nn.Dropout(args.dropout)
        in_dim = 2*args.token_embedding_dim + 2*args.pos_embedding_dim + 2*args.dep_embedding_dim+args.sen_pos_embedding_dim+args.word_pos_embedding_dim+args.event_embedding_dim

        self.token_bilstm = nn.LSTM(input_size=in_dim, hidden_size=args.hidden_size,
                              bidirectional=True, batch_first=True, num_layers=args.num_layers)

        self.eTypes = []
        for e_id,(etype,roles) in enumerate(event_roles.items()):
            self.eTypes.append(Event_Type(args,len(roles)).to(args.device))

        self.event_type_ids = torch.tensor([0,1,2,3,4]).to(args.device)

    def forward(self,token_ids,pos_ids,dep_ids,sen_pos_ids,word_pos_ids,parent_ids,ppos_ids,pdep_ids,event_ids):
        token_feature = self.embed(token_ids)  # (N,T,D)
        token_feature = self.dropout(token_feature)
        pos_feature = self.pos_embed(pos_ids)
        pos_feature = self.dropout(pos_feature)
        dep_feature = self.dep_embed(dep_ids)
        dep_feature = self.dropout(dep_feature)
        sen_pos_feature = self.sen_pos_embed(sen_pos_ids)
        sen_pos_feature = self.dropout(sen_pos_feature)
        word_pos_feature = self.word_pos_embed(word_pos_ids)
        word_pos_feature = self.dropout(word_pos_feature)
        parent_feature = self.embed(parent_ids)  # (N,T,D)
        parent_feature = self.dropout(parent_feature)
        ppos_feature = self.pos_embed(ppos_ids)
        ppos_feature = self.dropout(ppos_feature)
        pdep_feature = self.dep_embed(pdep_ids)
        pdep_feature = self.dropout(pdep_feature)
        event_id_feature = self.event_embed(event_ids)
        event_id_feature = self.dropout(event_id_feature)
        #
        event_type_feature = self.event_type_embed(self.event_type_ids)
        event_type_feature = self.dropout(event_type_feature)
        event_type_feature_list = event_type_feature.chunk(self.args.event_type_nums, 0)

        all_token_features = torch.cat([token_feature,pos_feature,dep_feature,sen_pos_feature,word_pos_feature,parent_feature,ppos_feature,pdep_feature,event_id_feature],
                                        dim=1)

        token_out_bilstm, _ = self.token_bilstm(all_token_features.unsqueeze(0))
        token_out_bilstm = self.dropout(token_out_bilstm).squeeze(0)

        logits = []
        for i, _ in enumerate(self.eTypes):
            logits.append(self.eTypes[i](token_out_bilstm,event_type_feature_list[i]))

        return logits
