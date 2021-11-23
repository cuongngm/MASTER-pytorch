import model.master as master_arch
import model.tablemaster as table_master_arch
from parse_config import ConfigParser
import argparse
import torch

args = argparse.ArgumentParser(description='MASTER PyTorch Distributed Training')
args.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
args.add_argument('-r', '--resume', default=None, type=str, help='path to latest checkpoint (default: None)')
args.add_argument('-d', '--device', default=None, type=str, help='indices of GPUs to be available (default: all)')
config = ConfigParser.from_args(args)
model = config.init_obj('model_arch', table_master_arch)
img = torch.randn(1, 3, 32, 256)
full_tgt = torch.zeros(30, dtype=torch.int64)
tgt = torch.LongTensor([3, 4, 5, 6, 7, 8, 10, 15, 2])
full_tgt[:len(tgt)] = tgt
full_tgt = full_tgt.unsqueeze(0)
print(full_tgt.size())
out = model(img, full_tgt)
print(out.size())
