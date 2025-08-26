import os
import re
import shutil

from torch.utils.data import Dataset
from utils.data_utils import expmap_to_euler_torch, expmap_to_quaternion, expmap2rotmat_torch, angle_separate
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import copy
from datetime import datetime


class TrainData(Dataset):
    def __init__(self, data_set, source_seq_len, target_seq_len, sample_size=10, sample_start=0, data_flip=False):
        self._data_set = data_set
        self._index_lst = list(data_set.keys())

        self._source_seq_len = source_seq_len
        self._target_seq_len = target_seq_len
        self._total_frames = self._source_seq_len + self._target_seq_len
        self._sample_start = sample_start
        self._sample_size = sample_size
        self._sample_idx = 0

        self.data_flip = data_flip

    def __len__(self):
        return len(self._index_lst) * self._sample_size

    def __getitem__(self, index):
        data = self._data_set[self._index_lst[index // self._sample_size]]

        if data.shape[0] - self._total_frames <= 0:
            self._sample_idx = 0
        else:
            self._sample_idx = np.random.randint(self._sample_start, data.shape[0] - self._total_frames)

        # Select the data around the sampled points
        data_sel = copy.deepcopy(data[self._sample_idx:self._sample_idx + self._total_frames, :])
        if self.data_flip:
            if torch.rand(1)[0] > 0.5:
                data_sel = data_sel[::-1]

        source_inputs = data_sel[0:self._source_seq_len].copy()
        target_outputs = data_sel[self._source_seq_len:].copy()

        return source_inputs, target_outputs

class Trainer:
    def __init__(self, model, learning_rate, step_size, factor, weight_decay, loss_type):
        super(Trainer, self).__init__()

        self.loss_type = loss_type
        self.mae_loss = nn.L1Loss().cuda()  # MAE
        self.mse_loss = nn.MSELoss().cuda()  # MSE
        self.optimizer = optim.Adam(model, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=factor)

    def get_loss(self, output, target):
        B, T, D = output.shape

        outputs = output.unflatten(-1, (-1, 3))
        targets = target.unflatten(-1, (-1, 3))
        outputs = angle_separate(outputs)
        targets = angle_separate(targets)
        # outputs = expmap_to_quaternion(outputs.reshape(-1, 3))
        # targets = expmap_to_quaternion(targets.reshape(-1, 3))

        if self.loss_type == "L1":
            total_loss = self.mae_loss(outputs, targets)
        elif self.loss_type == "L2":
            total_loss = self.mse_loss(outputs, targets)
        else:
            raise NotImplementedError(self.loss_type)

        return total_loss


    def epoch(self, model, loader):
        model.train()
        train_loss = 0
        iters = 1
        for source, target in tqdm(loader, desc='epoch train'):
            if torch.cuda.is_available():
                source = source.cuda()
                target = target.cuda()

            # Discard the first joint, which represents a corrupted translation
            source = source[..., 3:]
            target = target[..., 3:]

            output, attn_output = model(source)
            batch_loss = self.get_loss(output, target)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)

            self.optimizer.zero_grad()
            batch_loss.backward()
            self.optimizer.step()

            train_loss += batch_loss.item()
            iters += 1

        train_loss /= iters
        self.scheduler.step()

        return self.optimizer.param_groups[0]['lr'], train_loss


def test(test_gen, model, eval_frame, actions):
    eval_msg = ""
    header = f"{'milliseconds':<18} | " + " | ".join(f"{ms[1]:^7d}" for ms in eval_frame) + f" | {'avg':^7} |"
    eval_msg = eval_msg + header
    avg_angle = np.zeros(len(eval_frame))
    model.eval()
    for act in actions:
        with torch.no_grad():
            source, target = test_gen[act]
            source = torch.tensor(source).float()
            target = torch.tensor(target).float()
            # Evaluate the model on the test batches
            if torch.cuda.is_available():
                source = source.cuda()
                target = target.cuda()

            # Discard the first joint, which represents a corrupted translation
            source = source[..., 3:]
            target = target[..., 3:]

            output, attn_output = model(source)

            # Convert from exponential map to Euler angles
            target_tst = expmap_to_euler_torch(target.transpose(0, 1))
            pred_target = expmap_to_euler_torch(output.transpose(0, 1))

            error = torch.pow(target_tst[..., :] - pred_target[..., :], 2)
            error = torch.sqrt(torch.sum(error, dim=-1))
            error = torch.mean(error, dim=1)
            error = error.cpu().detach().numpy()
            angle_error = np.array([error[i[0]] for i in eval_frame])
            avg_angle += angle_error

            value = f"{act:<18} | " + " | ".join(
                f'{float(error):^7.3f}' for error in angle_error) + f" | {float(angle_error.mean()):^7.3f} |"
            eval_msg = eval_msg + '\n' + value

    avg_angle = avg_angle / len(actions)
    avg_avg_angle = avg_angle.mean()
    value = f"{'avg':<18} | " + " | ".join(
        f'{float(avg_error):^7.3f}' for avg_error in avg_angle) + f" | {float(avg_avg_angle):^7.3f} |"
    eval_msg = eval_msg + '\n' + value

    return avg_angle, avg_avg_angle, eval_msg

def save_ckpt(work_dir, exp_name, state, epoch, err, num):
    files = os.listdir(work_dir)
    file_info = []

    for file in files:
        if file.startswith(f"{exp_name}_best_") and file.endswith(".pt"):
            match = re.search(rf'{exp_name}_best_([\d.]+)_(\d+)\.pt$', file)
            if match:
                last_err = float(match.group(1))
                last_epoch = int(match.group(2))
                file_path = os.path.join(work_dir, file)
                file_info.append((last_err, last_epoch, file_path))

    file_info.sort(reverse=True, key=lambda x: x[0])
    file_path = os.path.join(work_dir, f'{exp_name}_best_{err:.3f}_{epoch}.pt')
    if len(file_info) < num:
        torch.save(state, file_path)
    elif err <= file_info[0][0]:
        to_remove = file_info[0][2]
        os.remove(to_remove)
        torch.save(state, file_path)


def exp_create(exp_dir, exp_name, yml):
    now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    work_dir = os.path.join(exp_dir, now)
    os.makedirs(work_dir, exist_ok=True)
    yml_back_up = os.path.join(work_dir, exp_name + '_backup.yml')
    shutil.copy(yml, yml_back_up)
    code_back_up = os.path.join(work_dir, exp_name + '_backup.py')
    shutil.copy("./module/Model.py", code_back_up)
    run_back_up = os.path.join(work_dir, exp_name + f'_run_backup.py')
    shutil.copy(f"./utils/runtime.py", run_back_up)
    log = os.path.join(work_dir, exp_name + '_eval.log')
    log = open(log, 'a')
    return work_dir, log
