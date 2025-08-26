from torch.utils.data import Dataset
import numpy as np
import copy
import random


def find_indices_srnn(data, action, batch_size):
    """
    Find the same action indices as in SRNN.
    See https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """
    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    subject = 5
    subaction1 = 1
    subaction2 = 2

    T1 = data[(subject, action, subaction1, 'even')].shape[0]
    T2 = data[(subject, action, subaction2, 'even')].shape[0]
    prefix, suffix = 50, 100

    idx = []
    for k in range(int(batch_size / 2)):
        idx.append(rng.randint(16, T1 - prefix - suffix))
        idx.append(rng.randint(16, T2 - prefix - suffix))

    return idx


def get_batch_srnn(data, action, source_seq_len, target_seq_len, input_size, batch_size):
    """
    Get a random batch of data from the specified bucket, prepare for step.
    Args
      data: dictionary with k:v, k=((subject, action, subsequence, 'even')),
        v=nxd matrix with a sequence of poses
      action: the action to load data from
    Returns
      The tuple (encoder_inputs, decoder_inputs, decoder_outputs);
      the constructed batches have the proper format to call step(...) later.
    """
    frames = {}
    batch_size = int(batch_size)  # we always evaluate 8 seeds
    frames[action] = find_indices_srnn(data, action, batch_size)

    subject = 5  # we always evaluate on subject 5

    seeds = [(action, (i % 2) + 1, frames[action][i]) for i in range(batch_size)]

    encoder_inputs = np.zeros((batch_size, source_seq_len, input_size * 3 + 3), dtype=float)
    decoder_target = np.zeros((batch_size, target_seq_len, input_size * 3 + 3), dtype=float)

    # Reproducing SRNN's sequence subsequence selection as done in
    # https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L343
    for i in range(batch_size):
        _, subsequence, idx = seeds[i]
        idx = idx + 50
        data_sel = data[(subject, action, subsequence, 'even')]
        data_sel = data_sel[(idx - source_seq_len):(idx + target_seq_len), :]

        encoder_inputs[i, :, :] = data_sel[0:source_seq_len, :]
        decoder_target[i, 0:len(data_sel) - source_seq_len, :] = data_sel[source_seq_len:, :]
    return encoder_inputs, decoder_target


def get_batch_srnn_cmu(data, action, source_seq_len, target_seq_len, input_size, batch_size):

    # Todo, as the result is not stable, I enlarge the batch_size of testing
    batch_size = int(batch_size)  # we always evaluate 80 seeds
    encoder_inputs = np.zeros((batch_size, source_seq_len, input_size * 3 + 3), dtype=float)
    decoder_target = np.zeros((batch_size, target_seq_len, input_size * 3 + 3), dtype=float)
    data_sel = copy.deepcopy(data[('test', action, 1, 'even')])
    total_frames = source_seq_len + target_seq_len

    SEED = 1234567890
    rng = np.random.RandomState(SEED)
    for i in range(batch_size):
        n, _ = data_sel.shape
        if n - total_frames <= 0:
            idx = 0
        else:
            idx = rng.randint(0, n - total_frames)

        data_sel2 = data_sel[idx:(idx + total_frames), :]
        encoder_inputs[i, :, :] = data_sel2[0:source_seq_len, :]
        decoder_target[i, :, :] = data_sel2[source_seq_len:, :]
    return encoder_inputs, decoder_target
