import torch
import torch.nn as nn
import torch.nn.functional as F

import configs
from models.transformer import Transformer
from models.graph_wavenet import GraphWaveNet
from models.DGCN_BLOCK import dgcn_block
from models.DGCN_BLOCK import DGCN
import time
from utils.load_data import load_adj
from utils.log import clock, load_pkl
class StackedDilatedConv(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, stride=2, dilation=1, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=2, padding=2)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=4, padding=4)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, stride=2, dilation=8, padding=8)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        h = self.conv1(x)
        h = F.gelu(h)
        h = self.pool1(h)
        h = self.conv2(h)
        h = F.gelu(h)
        h = self.pool2(h)
        h = self.conv3(h)
        h = F.gelu(h)
        h = self.pool3(h)
        h = self.conv4(h)
        h = F.gelu(h)
        h = self.pool4(h)
        return h


class DynamicGraphConv(nn.Module):
    def __init__(self, num_nodes, input_dim, output_dim, dropout, support_len=3, order=2):
        super().__init__()
        self.node_vec1 = nn.Parameter(torch.randn(num_nodes, 10))
        self.node_vec2 = nn.Parameter(torch.randn(10, num_nodes))
        input_dim = (order * support_len + 1) * input_dim
        self.linear = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.order = order

    def conv(self, x, adj_mx):
        return torch.einsum("bnh,nm->bnh", (x, adj_mx)).contiguous()

    def forward(self, x, supports):
        outputs = [x]
        new_supports = []
        new_supports.extend(supports)
        adaptive = torch.softmax(torch.relu(torch.mm(self.node_vec1, self.node_vec2)), dim=1)
        new_supports.append(adaptive)
        for adj_mx in new_supports:
            adj_mx = adj_mx.to(x.device)
            x1 = self.conv(x, adj_mx)
            outputs.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.conv(x1, adj_mx)
                outputs.append(x2)
                x1 = x2
        outputs = torch.cat(outputs, dim=2)
        outputs = self.linear(outputs)
        outputs = self.dropout(outputs)
        return outputs

devive = torch.device("cuda" if torch.cuda.is_available() else "cpu")
'''class FeatureFusionModule(nn.Module):
    def __init__(self,num_fue,num_nodes):
        super(FeatureFusionModule,self).__init__()
        self.num_fue = num_fue
        self.num_nodes = num_nodes

        #卷积层降为24时间步
        self.conv1 = nn.Conv2d(in_channels=num_fue, out_channels=64, kernel_size=(1, 3), padding=(0, 1))
        self.pool = nn.MaxPool2d(kernel_size=(1, 4), stride=(1, 4))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1, 3), padding=(0, 1))
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=3, kernel_size=(1, 3), padding=(0, 1))

    def forward(self, fused_features):
        fused_features = torch.cat((trend_seasonality_hidden_expanded, short_x), dim=-1) #([8, 3, 307, 140])
        x = self.conv1(fused_features) #([8, 64, 307, 140])
        x = nn.ReLU()(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)#[8, 32, 307, 35])
        x = self.conv3(x)



        #x = x.view(x.size(0), -1) #([8, 343840])

        self.fc = nn.Linear(35, 24).cuda()
        output = self.fc(x)



        #output = output.view(x.size(0), self.num_fue, self.num_nodes, 24)
        #output = output.premute(0, 1, 2, 3)
        return output
'''





class LSTTN(nn.Module):
    def __init__(self, cfg, **model_args):
        super().__init__()
        transformer_args = model_args["TRANSFORMER"]
        lsttn_args = model_args["LSTTN"]
        dgcn_args = model_args["DGCN"]
        self.dataset_name = cfg.DATASET_NAME
        self.transformer1 = Transformer(mode="pretrain", **transformer_args)
        self.transformer = Transformer(mode="inference", **transformer_args)
        self.load_pretrained_transformer()
        self.num_nodes = lsttn_args["num_nodes"]
        self.pre_len = lsttn_args["pre_len"]
        self.long_trend_hidden_dim = lsttn_args["long_trend_hidden_dim"]
        self.seasonality_hidden_dim = lsttn_args["seasonality_hidden_dim"]
        self.mlp_hidden_dim = lsttn_args["mlp_hidden_dim"]
        self.dropout = lsttn_args["dropout"]
        dgcn = DGCN(**dgcn_args.GWNET)
        self.transformer_hidden_dim = self.transformer.encoder.d_model
        self.supports = lsttn_args["supports"]
        # 定义长期趋势以及周期提取块
        self.long_term_trend_extractor = StackedDilatedConv(self.transformer_hidden_dim, self.long_trend_hidden_dim)
        self.weekly_seasonality_extractor = DynamicGraphConv(
            self.num_nodes, self.transformer_hidden_dim, self.seasonality_hidden_dim, self.dropout
        )
        self.daily_seasonality_extractor = DynamicGraphConv(
            self.num_nodes, self.transformer_hidden_dim, self.seasonality_hidden_dim, self.dropout
        )
        # 定义长期趋势与周期融合块
        self.trend_seasonality_mlp = nn.Sequential(
            nn.Linear(self.long_trend_hidden_dim + self.seasonality_hidden_dim * 2, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
        )

        # 定义预测模块
        self.prediction_layer = dgcn
        self.mlp1 = nn.Sequential(
            nn.Linear(140,64),
            nn.ReLU(),
            nn.Linear(64,32),
            nn.ReLU(),
            nn.Linear(32, 24),
        )




        '''new_dim = (self.mlp_hidden_dim + dgcn_args["GWNET"]["out_dim"])//2
        self.mlp = nn.Sequential(
            nn.Linear(new_dim , self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.pre_len),
        )'''


    def load_pretrained_transformer(self):
        if self.dataset_name == "METRLA":
            state_dict = torch.load("../pretrained_transformers/METR-LA.pt")
        elif self.dataset_name == "PEMSBAY":
            state_dict = torch.load("../pretrained_transformers/PEMS-BAY.pt")
        elif self.dataset_name == "PEMS04":
            state_dict = torch.load("/home/zhu/桌面/Zhu/LSTTN-main1/pretrained_transformers/PEMS04.pt")
        elif self.dataset_name == "PEMS08":
            state_dict = torch.load("/home/zhu/桌面/Zhu/T-DGCN-08/pretrained_transformers/PEMS08.pt")
        else:
            assert NameError, "Unknown dataset"
        self.transformer.load_state_dict(state_dict["model_state_dict"])
        for param in self.transformer.parameters():
            param.requires_grad = False

    def forward(self, short_x, long_x):
        long_x = long_x[..., [0]]
        long_x = torch.permute(long_x, (0, 2, 3, 1))
        batch_size, num_nodes, _, long_seq_len = long_x.size()
        long_repr = self.transformer(long_x)
        batch_size, num_nodes, num_subseq, _ = long_repr.size()
        long_trend_hidden = self.long_term_trend_extractor(
            torch.reshape(long_repr, (-1, num_subseq, self.transformer_hidden_dim)).transpose(1, 2)
        )[:, :, -1]
        long_trend_hidden = torch.reshape(long_trend_hidden, (batch_size, num_nodes, -1))
        last_week_repr = long_repr[:, :, -7 * 24 - 1, :]
        last_day_repr = long_repr[:, :, -25, :]

        # 融合长期趋势和周期性特征
        weekly_hidden = self.weekly_seasonality_extractor(last_week_repr, self.supports)
        daily_hidden = self.daily_seasonality_extractor(last_day_repr, self.supports)
        trend_seasonality_hidden = torch.cat((long_trend_hidden, weekly_hidden, daily_hidden), dim=-1)
        trend_seasonality_hidden = self.trend_seasonality_mlp(trend_seasonality_hidden)
        short_x = torch.transpose(short_x, 1, 3)
        _, num_fue, _, _ = short_x.size()
        trend_seasonality_hidden_expanded = trend_seasonality_hidden.unsqueeze(-1)
        trend_seasonality_hidden_expanded = trend_seasonality_hidden_expanded.repeat(1, 1, 1, num_fue)
        trend_seasonality_hidden_expanded = trend_seasonality_hidden_expanded.permute(0, 3, 1, 2)
        fused_features = torch.cat([short_x, trend_seasonality_hidden_expanded], dim=-1)
        #创建特征融合模块
        '''fusion_module = FeatureFusionModule(num_fue, num_nodes).cuda()
        #进行特征融合
        fused_output = fusion_module(fused_features)
        output1, _, _ = self.prediction_layer(fused_output, self.supports)'''

        fused_features = torch.cat([short_x, trend_seasonality_hidden_expanded], dim=-1)

        x_1 = fused_features
        batch_size, num_fue, num_nodes, seq_length = x_1.shape
        x_1 = x_1.view(-1, seq_length)
        x_1 = self.mlp1(x_1)
        x_1 = x_1.view(batch_size, num_fue, num_nodes,-1)


        # 预测
        output1, _, _ = self.prediction_layer(x_1, self.supports)
        #output1, _, _ = self.prediction_layer(fused_output, self.supports)

        #output = self.mlp(output1)
        return output1
#消融
class LSTTN_NoSeasonality(LSTTN):
    """去掉周期性模块的LSTTN消融版本"""
    def __init__(self, cfg, **model_args):
        super().__init__(cfg, **model_args)
        # 重写去周期性后的 MLP
        self.trend_seasonality_mlp = nn.Sequential(
            nn.Linear(self.long_trend_hidden_dim, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
        )

    def forward(self, short_x, long_x):
        long_x = long_x[..., [0]]
        long_x = torch.permute(long_x, (0, 2, 3, 1))
        batch_size, num_nodes, _, long_seq_len = long_x.size()
        long_repr = self.transformer(long_x)
        batch_size, num_nodes, num_subseq, _ = long_repr.size()
        long_trend_hidden = self.long_term_trend_extractor(
            torch.reshape(long_repr, (-1, num_subseq, self.transformer_hidden_dim)).transpose(1, 2)
        )[:, :, -1]
        long_trend_hidden = torch.reshape(long_trend_hidden, (batch_size, num_nodes, -1))

        # 不使用周期性特征
        trend_seasonality_hidden = long_trend_hidden
        trend_seasonality_hidden = self.trend_seasonality_mlp(trend_seasonality_hidden)

        short_x = torch.transpose(short_x, 1, 3)
        _, num_fue, _, _ = short_x.size()
        trend_seasonality_hidden_expanded = trend_seasonality_hidden.unsqueeze(-1)
        trend_seasonality_hidden_expanded = trend_seasonality_hidden_expanded.repeat(1, 1, 1, num_fue)
        trend_seasonality_hidden_expanded = trend_seasonality_hidden_expanded.permute(0, 3, 1, 2)
        fused_features = torch.cat([short_x, trend_seasonality_hidden_expanded], dim=-1)
        x_1 = fused_features
        batch_size, num_fue, num_nodes, seq_length = x_1.shape
        x_1 = x_1.view(-1, seq_length)
        x_1 = self.mlp1(x_1)
        x_1 = x_1.view(batch_size, num_fue, num_nodes,-1)

        output1, _, _ = self.prediction_layer(x_1, self.supports)
        return output1

class LSTTN_NoLongTrend(LSTTN):
    """去掉长期趋势模块的LSTTN消融版本"""
    def __init__(self, cfg, **model_args):
        super().__init__(cfg, **model_args)
        # 重写去周期性后的 MLP
        self.trend_seasonality_mlp = nn.Sequential(
            nn.Linear(self.seasonality_hidden_dim * 2, self.mlp_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.mlp_hidden_dim, self.mlp_hidden_dim),
        )
    def forward(self, short_x, long_x):
        long_x = long_x[..., [0]]
        long_x = torch.permute(long_x, (0, 2, 3, 1))
        batch_size, num_nodes, _, long_seq_len = long_x.size()
        long_repr = self.transformer(long_x)
        batch_size, num_nodes, num_subseq, _ = long_repr.size()

        # 不使用长期趋势特征
        last_week_repr = long_repr[:, :, -7 * 24 - 1, :]
        last_day_repr = long_repr[:, :, -25, :]

        weekly_hidden = self.weekly_seasonality_extractor(last_week_repr, self.supports)
        daily_hidden = self.daily_seasonality_extractor(last_day_repr, self.supports)
        trend_seasonality_hidden = torch.cat((weekly_hidden, daily_hidden), dim=-1)
        trend_seasonality_hidden = self.trend_seasonality_mlp(trend_seasonality_hidden)

        short_x = torch.transpose(short_x, 1, 3)
        _, num_fue, _, _ = short_x.size()
        trend_seasonality_hidden_expanded = trend_seasonality_hidden.unsqueeze(-1)
        trend_seasonality_hidden_expanded = trend_seasonality_hidden_expanded.repeat(1, 1, 1, num_fue)
        trend_seasonality_hidden_expanded = trend_seasonality_hidden_expanded.permute(0, 3, 1, 2)
        fused_features = torch.cat([short_x, trend_seasonality_hidden_expanded], dim=-1)
        x_1 = fused_features
        batch_size, num_fue, num_nodes, seq_length = x_1.shape
        x_1 = x_1.view(-1, seq_length)
        x_1 = self.mlp1(x_1)
        x_1 = x_1.view(batch_size, num_fue, num_nodes, -1)

        output1, _, _ = self.prediction_layer(x_1, self.supports)
        return output1
#消融


if __name__ == "__main__":
    from torchinfo import summary
    from configs.PEMS08.forecasting import CFG


    model = LSTTN(CFG, **CFG.MODEL.PARAM)

    x_short = torch.randn((32, 12, 207, 3))
    x_long = torch.randn((32, 4032, 207, 3))
    # batch_size, long_seq_len, num_nodes, num_features,

    transformer = model.transformer

    x_long = x_long[..., [0]]
    x_long = torch.permute(x_long, (0, 2, 3, 1))
    start = time.time()
    long_repr = transformer(x_long)
    end_transformer = time.time()
    print(end_transformer - start)

    x_short = x_short[..., [0]]
    x_short = torch.permute(x_short, (0, 2, 3, 1))
    start = time.time()
    short_repr = transformer(x_short)
    end_transformer = time.time()
    print(end_transformer - start)
