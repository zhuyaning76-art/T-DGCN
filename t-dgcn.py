import time
import math
import torch
import torch.nn as nn
from easytorch.core.data_loader import build_data_loader
from easytorch.utils.dist import master_only
from easytorch.device import to_device
from sklearn.metrics import mean_absolute_error
from easytorch import Runner, Config
from torch.utils.data import DataLoader

from models.lsttn import LSTTN
from utils.log import load_pkl
from utils.datasets import MTSDataset
from utils.load_data import max_min_scale, standard_scale
from utils.metrics import masked_mae, masked_rmse, masked_mape, all_metrics
from models.lsttn import LSTTN_NoSeasonality, LSTTN_NoLongTrend

class LSTTNRunner(Runner):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.cfg = cfg
        self.clip = 3
        dataset_name = cfg["DATASET_NAME"]
        # scaler
        if dataset_name in ["PEMS04", "PEMS08"]:
            _min = load_pkl("/home/zhu/桌面/Zhu/T-DGCN-08/datasets/PEMS08/min.pkl")
            _max = load_pkl("/home/zhu/桌面/Zhu/T-DGCN-08/datasets/PEMS08/max.pkl")
            self.scaler = max_min_scale
            self.scaler_args = {"min": _min, "max": _max}
        elif dataset_name in ["PEMS-BAY", "METR-LA"]:
            mean = load_pkl("datasets/" + dataset_name + "/mean.pkl")
            std = load_pkl("datasets/" + dataset_name + "/std.pkl")
            self.scaler = standard_scale
            self.scaler_args = {"mean": mean, "std": std}
        self.loss = masked_mae
        # self.loss = masked_mae_loss

        self.dataset_name = cfg["DATASET_NAME"]
        self.output_seq_len = 12
        self.cl_len = self.output_seq_len
        self.if_cl = True
        self.best_val_loss = float('inf')#初始化为无穷大
        self.best_model_path = "best_model.pth"#模型保存路径
    def init_training(self, cfg):
        super().init_training(cfg)
        self.register_epoch_meter("train_loss", "train", "{:.4f}")
        self.register_epoch_meter("train_MAPE", "train", "{:.4f}")
        self.register_epoch_meter("train_RMSE", "train", "{:.4f}")

    def init_validation(self, cfg: dict):
        super().init_validation(cfg)
        self.register_epoch_meter("val_loss", "val", "{:.4f}")
        self.register_epoch_meter("val_MAPE", "val", "{:.4f}")
        self.register_epoch_meter("val_RMSE", "val", "{:.4f}")
        #创建数据加载验证
        '''self.val_data_loader = self.build_val_data_loader(cfg)
    def build_val_data_loader(self, cfg: Config) -> DataLoader:
        dataset = self.build_val_data_loader(cfg)
        return build_data_loader(dataset, cfg["VAL"]["DATA"])'''

    @staticmethod
    def define_model(cfg: dict) -> nn.Module:
        name = cfg["MODEL"]["NAME"]
        if name == "LSTTN":
            return LSTTN(cfg, **cfg.MODEL.PARAM)
        elif name == "LSTTN_NoSeasonality":
            return LSTTN_NoSeasonality(cfg, **cfg.MODEL.PARAM)
        elif name == "LSTTN_NoLongTrend":
            return LSTTN_NoLongTrend(cfg, **cfg.MODEL.PARAM)
        else:
            raise ValueError(f"Unknown model name {name}")
    def build_train_dataset(self, cfg: dict):
        raw_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TRAIN"]["DATA"]["DIR"] + "/train_index.pkl"
        seq_len = cfg["TRAIN"]["DATA"]["SEQ_LEN"]
        batch_size = cfg["TRAIN"]["DATA"]["BATCH_SIZE"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, throw=False, pretrain=False)

        warmup_epochs = cfg["TRAIN"]["WARMUP_EPOCHS"]
        cl_epochs = cfg["TRAIN"]["CL_EPOCHS"]
        self.init_lr = cfg["TRAIN"]["OPTIM"]["PARAM"]["lr"]
        self.itera_per_epoch = math.ceil(len(dataset) / batch_size)
        self.warmup_steps = self.itera_per_epoch * warmup_epochs
        self.cl_steps = self.itera_per_epoch * cl_epochs
        print("cl_steps:{0}".format(self.cl_steps))
        print("warmup_steps:{0}".format(self.warmup_steps))

        return dataset

    @staticmethod
    def build_val_dataset(cfg: dict):
        raw_file_path = cfg["VAL"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["VAL"]["DATA"]["DIR"] + "/valid_index.pkl"
        seq_len = cfg["VAL"]["DATA"]["SEQ_LEN"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, pretrain=False)
        print("val len: {0}".format(len(dataset)))
        return dataset

    @staticmethod
    def build_test_dataset(cfg: dict):
        raw_file_path = cfg["TEST"]["DATA"]["DIR"] + "/data.pkl"
        index_file_path = cfg["TEST"]["DATA"]["DIR"] + "/test_index.pkl"
        seq_len = cfg["TEST"]["DATA"]["SEQ_LEN"]
        dataset = MTSDataset(raw_file_path, index_file_path, seq_len, pretrain=False)
        print("test len: {0}".format(len(dataset)))
        return dataset

    def train_iters(self, epoch, iter_index, data):
        iter_num = (epoch - 1) * self.itera_per_epoch + iter_index

        y, x_short, x_long = data
        y = to_device(y)
        x_short = to_device(x_short)
        x_long = to_device(x_long)

        output = self.model(x_short, x_long)
        output = output.transpose(1, 2)
        #print(output.shape)

        if iter_num < self.warmup_steps:
            self.cl_len = self.output_seq_len
        elif iter_num == self.warmup_steps:
            self.cl_len = 1
            for param_group in self.optim.param_groups:
                param_group["lr"] = self.init_lr
            print("======== Start curriculum learning... reset the learning rate to {0}. ========".format(self.init_lr))
        else:
            if (iter_num - self.warmup_steps) % self.cl_steps == 0 and self.cl_len <= self.output_seq_len:
                self.cl_len += int(self.if_cl)

        if "max" in self.scaler_args.keys():
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            #print(predict.shape)
            real_val = self.scaler(y.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val = real_val[..., 0]
            #print(real_val.shape)
            mae_loss = self.loss(predict[:, : self.cl_len, :], real_val[:, : self.cl_len, :])
        else:
            # inverse transform for both predict and real value.
            predict = self.scaler(output, **self.scaler_args)
            real_val = self.scaler(y[:, :, :, 0], **self.scaler_args)
            mae_loss = self.loss(predict[:, : self.cl_len, :], real_val[:, : self.cl_len, :], 0)

        loss = mae_loss
        # metrics
        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)

        self.update_epoch_meter("train_loss", loss.item())
        self.update_epoch_meter("train_MAPE", mape.item())
        self.update_epoch_meter("train_RMSE", rmse.item())
        #加内容
        #在每个epoch结束后验证
        if iter_index == self.itera_per_epoch - 1:
            val_loss = self.validate1() #自定义函数验证
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.best_model_path) #保存最佳模型
                print(f"Best model saved with loss {self.best_val_loss:.4f}")

        return loss

    def val_iters(self, iter_index, data):
        y, x_short, x_long = data
        y = to_device(y)
        x_short = to_device(x_short)
        x_long = to_device(x_long)

        output = self.model(x_short, x_long)
        output = output.transpose(1, 2)

        if "max" in self.scaler_args.keys():
            predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val = self.scaler(y.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(-1)
            real_val = real_val[..., 0]
        else:
            predict = self.scaler(output, **self.scaler_args)
            real_val = self.scaler(y[:, :, :, 0], **self.scaler_args)

        loss = self.loss(predict, real_val, 0.0)
        mape = masked_mape(predict, real_val, 0.0)
        rmse = masked_rmse(predict, real_val, 0.0)

        self.update_epoch_meter("val_loss", loss.item())
        self.update_epoch_meter("val_MAPE", mape.item())
        self.update_epoch_meter("val_RMSE", rmse.item())

    def validate1(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for data in self.val_data_loader:
                y, x_short, x_long = data
                y = to_device(y)
                x_short = to_device(x_short)
                x_long = to_device(x_long)
                output = self.model(x_short, x_long)
                output = output.transpose(1, 2)
                if "max" in self.scaler_args.keys():
                    predict = self.scaler(output.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1,
                                                                                                              2).squeeze(
                        -1)
                    real_val = self.scaler(y.transpose(1, 2).unsqueeze(-1), **self.scaler_args).transpose(1, 2).squeeze(
                        -1)
                    real_val = real_val[..., 0]
                else:
                    predict = self.scaler(output, **self.scaler_args)
                    real_val = self.scaler(y[:, :, :, 0], **self.scaler_args)

                loss = self.loss(predict, real_val, 0.0)
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_data_loader)
        return avg_loss



    def on_training_end(self):
        self.test(self.cfg)
        super().on_training_end()

    @master_only
    def init_test(self, cfg: dict):
        self.register_epoch_meter("test_loss", "test", "{:.4f}")
        self.register_epoch_meter("test_MAPE", "test", "{:.4f}")
        self.register_epoch_meter("test_RMSE", "test", "{:.4f}")

        self.test_interval = cfg['TEST'].get('INTERVAL', 1)
        self.test_data_loader = self.build_test_data_loader(cfg)
        self.register_epoch_meter('test_time', 'test', '{:.2f} (s)', plt=False)

    def build_test_data_loader(self, cfg: dict):
        dataset = self.build_test_dataset(cfg)
        return build_data_loader(dataset, cfg['TEST']['DATA'])

    @torch.no_grad()
    @master_only
    def test(self, cfg: dict = None, train_epoch: int = None):
        if train_epoch is None:
            self.init_test(cfg)
        self.model.load_state_dict(torch.load('best_model.pth'))    #加载最佳模型
        test_start_time = time.time()
        self.model.eval()

        outputs = []
        y_list = []
        for iter_index, data in enumerate(self.test_data_loader):
            preds, testy = self.test_iters(iter_index, data)
            outputs.append(preds)
            y_list.append(testy)
        yhat = torch.cat(outputs, dim=0)
        y_list = torch.cat(y_list, dim=0)

        if "max" in self.scaler_args.keys():
            real_val = self.scaler(y_list.squeeze(-1), **self.scaler_args).transpose(1, 2)
            predict = self.scaler(yhat.unsqueeze(-1), **self.scaler_args).transpose(1, 2)
            real_val = real_val[..., 0]
            predict = predict[..., 0]
        else:
            real_val = self.scaler(y_list[:, :, :, 0], **self.scaler_args).transpose(1, 2)
            predict = self.scaler(yhat, **self.scaler_args).transpose(1, 2)

        mae_of_batches = []
        mape_of_batches = []
        rmse_of_batches = []

        for i in range(6):
            pred = predict[:, :, i]
            real = real_val[:, :, i]
            dataset_name = self.dataset_name
            if (
                dataset_name == "PEMS04" or dataset_name == "PEMS08"
            ):
                mae = mean_absolute_error(pred.cpu().numpy(), real.cpu().numpy())
                rmse = masked_rmse(pred, real, 0.0).item()
                mape = masked_mape(pred, real, 0.0).item()

            else:
                metrics = all_metrics(pred, real)
                mae_of_batches.append(metrics[0])
                mape_of_batches.append(metrics[1])
                rmse_of_batches.append(metrics[2])

            for j in range(2):
                log = "Evaluate best model on test data for horizon {:d}, Test MAE: {:.4f}, Test RMSE: {:.4f}, Test MAPE: {:.4f}"
                horizon = int(i * 2 + j + 1)
                log = log.format(horizon, mae, rmse, mape)
                mae_of_batches.append(mae)
                mape_of_batches.append(mape)
                rmse_of_batches.append(rmse)
                self.logger.info(log)


        import numpy as np

        log = "On average over 12 horizons, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}"
        self.logger.info(log.format(np.mean(mae_of_batches), np.mean(mape_of_batches), np.mean(rmse_of_batches)))

        test_end_time = time.time()
        print(test_end_time - test_start_time)
        self.update_epoch_meter("test_time", test_start_time - test_end_time)
        self.print_epoch_meters("test")
        if train_epoch is not None:
            self.plt_epoch_meters("test", train_epoch // self.test_interval)

    def test_iters(self, iter_index: int, data: torch.Tensor or tuple):
        y, x_short, x_long = data
        y = to_device(y)
        x_short = to_device(x_short)
        x_long = to_device(x_long)

        output = self.model(x_short, x_long)
        output = output.transpose(1, 2)
        return output, y


if __name__ == "__main__":
    from configs.PEMS08.forecasting import CFG

    runner = LSTTNRunner(CFG)
    test_loader = runner.build_test_data_loader(CFG)
    runner.test(CFG)

    test_start_time = time.time()
    for iter_index, data in enumerate(runner.test_data_loader):
        runner.test_iters(iter_index, data)
        test_end_time = time.time()
        print(test_end_time - test_start_time)
