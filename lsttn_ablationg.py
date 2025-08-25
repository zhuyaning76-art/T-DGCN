import torch
import torch.nn as nn
from models.lsttn import LSTTN

class LSTTN_NoSeasonality(LSTTN):
    """去掉周期性模块的LSTTN消融版本"""
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

        output1, _, _ = self.prediction_layer(fused_features, self.supports)
        output = self.mlp(output1)
        return output

class LSTTN_NoLongTrend(LSTTN):
    """去掉长期趋势模块的LSTTN消融版本"""
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

        output1, _, _ = self.prediction_layer(fused_features, self.supports)
        output = self.mlp(output1)
        return output