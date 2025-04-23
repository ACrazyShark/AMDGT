import dgl.nn.pytorch
import torch
import torch.nn as nn
from model import gt_net_drug, gt_net_disease
import torch.nn.functional as F

device = torch.device('cuda')

# 方法一
# class AMNTDDA(nn.Module):
#     def __init__(self, args):
#         super(AMNTDDA, self).__init__()
#         self.args = args

#         # 定义每个相似性矩阵的学习权重
#         self.weight_drf = nn.Parameter(torch.ones(1))  # 药物指纹相似性矩阵的权重
#         self.weight_drg = nn.Parameter(torch.ones(1))  # 药物GIP相似性矩阵的权重
#         self.weight_drp = nn.Parameter(torch.ones(1))  # 药物-蛋白质关联矩阵的权重


#         self.drug_linear = nn.Linear(300, args.hgt_in_dim)
#         self.protein_linear = nn.Linear(320, args.hgt_in_dim)
#         self.gt_drug = gt_net_drug.GraphTransformer(device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim,
#                                                     args.gt_head, args.dropout)
#         self.gt_disease = gt_net_disease.GraphTransformer(device, args.gt_layer, args.disease_number, args.gt_out_dim,
#                                                     args.gt_out_dim, args.gt_head, args.dropout)

#         self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim/args.hgt_head), args.hgt_head, 3, 3, args.dropout) ## hgt 修改
#         self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, args.hgt_head_dim, args.hgt_head, 3, 3, args.dropout) ## hgt 修改
#         self.hgt = nn.ModuleList()
#         for l in range(args.hgt_layer-1):
#             self.hgt.append(self.hgt_dgl)
#         self.hgt.append(self.hgt_dgl_last)

#         encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
#         self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
#         self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

#         self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
#         self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

#         # self.mlp = nn.Sequential(
#         #     nn.Linear(args.gt_out_dim * 2, 1024),
#         #     # nn.Linear(256, 1024),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.4),
#         #     nn.Linear(1024, 1024),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.4),
#         #     nn.Linear(1024, 256),
#         #     nn.ReLU(),
#         #     nn.Dropout(0.4),
#         #     nn.Linear(256, 2)
#         # )

#         # 替换RBF部分为更强大的架构
#         self.input_dim = 2 * args.gt_out_dim * 2 + 2 * args.gt_out_dim + 1
        
#         self.relation_net = nn.Sequential(
#             nn.Linear(self.input_dim, 512),  # 输入维度改为计算值
#             nn.ReLU(),
#             nn.Dropout(args.dropout),
#             nn.Linear(512, 256),
#             nn.ReLU(),
#             nn.Dropout(args.dropout),
#             nn.Linear(256, 2)
#         )
#         # 保持gamma作为可学习参数
#         self.gamma = nn.Parameter(torch.tensor(1.0))

#     # AMDGT $$
#     def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample):
#         dr_sim = self.gt_drug(drdr_graph)
#         di_sim = self.gt_disease(didi_graph)

#         drug_feature = self.drug_linear(drug_feature)
#         protein_feature = self.protein_linear(protein_feature)

#         feature_dict = {
#             'drug': drug_feature,
#             'disease': disease_feature,
#             'protein': protein_feature
#         }

#         drdipr_graph.ndata['h'] = feature_dict
#         g = dgl.to_homogeneous(drdipr_graph, ndata='h')
#         feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)

#         for layer in self.hgt:
#             hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
#             feature = hgt_out

#         dr_hgt = hgt_out[:self.args.drug_number, :]
#         di_hgt = hgt_out[self.args.drug_number:self.args.disease_number+self.args.drug_number, :]

#         dr = torch.stack((dr_sim, dr_hgt), dim=1)
#         di = torch.stack((di_sim, di_hgt), dim=1)

#         dr = self.drug_trans(dr)
#         di = self.disease_trans(di)

#         # mlp
#         # dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
#         # di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

#         # drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])
#         # output = self.mlp(drdi_embedding)

#         # RBF核函数：exp(-γ * ||h1 - h2||²)
#         dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
#         di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)
#         dr_sample = dr[sample[:, 0]]
#         di_sample = di[sample[:, 1]]
        
#         diff = dr_sample - di_sample
#         distance = torch.norm(diff, p=2, dim=1, keepdim=True)
#         rbf_score = torch.exp(-F.softplus(self.gamma) * distance**2)
        
#         # 组合多种特征
#         combined = torch.cat([
#             dr_sample,
#             di_sample,
#             diff,
#             rbf_score
#         ], dim=1)
        
#         # 通过关系网络得到最终输出
#         output = self.relation_net(combined)


#         return dr, output




# 方法二
class AMNTDDA(nn.Module):
    def __init__(self, args):
        super(AMNTDDA, self).__init__()
        self.args = args

        # 定义每个相似性矩阵的学习权重
        self.weight_drf = nn.Parameter(torch.ones(1))  # 药物指纹相似性矩阵的权重
        self.weight_drg = nn.Parameter(torch.ones(1))  # 药物GIP相似性矩阵的权重
        self.weight_drp = nn.Parameter(torch.ones(1))  # 药物-蛋白质关联矩阵的权重


        self.drug_linear = nn.Linear(300, args.hgt_in_dim)
        self.protein_linear = nn.Linear(320, args.hgt_in_dim)
        self.gt_drug = gt_net_drug.GraphTransformer(device, args.gt_layer, args.drug_number, args.gt_out_dim, args.gt_out_dim,
                                                    args.gt_head, args.dropout)
        self.gt_disease = gt_net_disease.GraphTransformer(device, args.gt_layer, args.disease_number, args.gt_out_dim,
                                                    args.gt_out_dim, args.gt_head, args.dropout)

        self.hgt_dgl = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, int(args.hgt_in_dim/args.hgt_head), args.hgt_head, 3, 3, args.dropout) ## hgt 修改
        self.hgt_dgl_last = dgl.nn.pytorch.conv.HGTConv(args.hgt_in_dim, args.hgt_head_dim, args.hgt_head, 3, 3, args.dropout) ## hgt 修改
        self.hgt = nn.ModuleList()
        for l in range(args.hgt_layer-1):
            self.hgt.append(self.hgt_dgl)
        self.hgt.append(self.hgt_dgl_last)

        encoder_layer = nn.TransformerEncoderLayer(d_model=args.gt_out_dim, nhead=args.tr_head)
        self.drug_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)
        self.disease_trans = nn.TransformerEncoder(encoder_layer, num_layers=args.tr_layer)

        self.drug_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)
        self.disease_tr = nn.Transformer(d_model=args.gt_out_dim, nhead=args.tr_head, num_encoder_layers=3, num_decoder_layers=3, batch_first=True)

        # 替换RBF部分为更强大的架构
        self.input_dim = 2 * args.gt_out_dim * 2 + 2 * args.gt_out_dim + 1
        
        self.relation_net = nn.Sequential(
            nn.Linear(3, 32),  # 输入：diff_mean, diff_max, rbf_score
            nn.ReLU(),
            nn.Linear(32, 2)
        )
        # 保持gamma作为可学习参数
        self.gamma = nn.Parameter(torch.tensor(1.0))

    # AMDGT $$
    def forward(self, drdr_graph, didi_graph, drdipr_graph, drug_feature, disease_feature, protein_feature, sample):
        dr_sim = self.gt_drug(drdr_graph)
        di_sim = self.gt_disease(didi_graph)

        drug_feature = self.drug_linear(drug_feature)
        protein_feature = self.protein_linear(protein_feature)

        feature_dict = {
            'drug': drug_feature,
            'disease': disease_feature,
            'protein': protein_feature
        }

        drdipr_graph.ndata['h'] = feature_dict
        g = dgl.to_homogeneous(drdipr_graph, ndata='h')
        feature = torch.cat((drug_feature, disease_feature, protein_feature), dim=0)

        for layer in self.hgt:
            hgt_out = layer(g, feature, g.ndata['_TYPE'], g.edata['_TYPE'], presorted=True)
            feature = hgt_out

        dr_hgt = hgt_out[:self.args.drug_number, :]
        di_hgt = hgt_out[self.args.drug_number:self.args.disease_number+self.args.drug_number, :]

        dr = torch.stack((dr_sim, dr_hgt), dim=1)
        di = torch.stack((di_sim, di_hgt), dim=1)

        dr = self.drug_trans(dr)
        di = self.disease_trans(di)

        # mlp
        # dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
        # di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)

        # drdi_embedding = torch.mul(dr[sample[:, 0]], di[sample[:, 1]])
        # output = self.mlp(drdi_embedding)

        # RBF核函数：exp(-γ * ||h1 - h2||²)
        dr = dr.view(self.args.drug_number, 2 * self.args.gt_out_dim)
        di = di.view(self.args.disease_number, 2 * self.args.gt_out_dim)
        dr_sample = dr[sample[:, 0]]
        di_sample = di[sample[:, 1]]
        
        diff = dr_sample - di_sample
        distance = torch.norm(diff, p=2, dim=1, keepdim=True)
        rbf_score = torch.exp(-F.softplus(self.gamma) * distance**2)
        
        diff_mean = diff.mean(dim=1, keepdim=True)  # 差异特征的均值
        diff_max = diff.max(dim=1, keepdim=True)[0]  # 差异特征的最大值
        combined = torch.cat([diff_mean, diff_max, rbf_score], dim=1)
        output = self.relation_net(combined)
        return dr, output
