import argparse
import pickle
import numpy as np
import os
import json
import torch
import torch.nn as nn
import utils
import torch.nn.utils.rnn as rnn_utils
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import glob
from pioneer import Pioneer

class CustomDataset(Dataset):
    def __init__(self, json_files):
        self.observations = []
        self.capacities= []
        # self.true_bandwidth = []  # When evaluate, you may need add this item
        self.video_quality = []
        self.audio_quality = []
        counter = 0
        for file_path in json_files:
            with open(file_path, 'r') as file:
                data = json.load(file)
                observation = torch.tensor(data['observations'], dtype=torch.float)
                capacity = torch.tensor(data['bandwidth_predictions'], dtype=torch.float)
                # true_bandwidth = torch.tensor(data['true_capacity'], dtype=torch.float)
                video_quality = torch.tensor(np.nan_to_num(data['video_quality']), dtype=torch.float)                
                audio_quality = torch.tensor(np.nan_to_num(data['audio_quality']), dtype=torch.float)         

                self.observations.append(observation)
                self.capacities.append(capacity)
                # self.true_bandwidth.append(true_bandwidth)
                self.video_quality.append(video_quality)
                self.audio_quality.append(audio_quality)

            counter += 1
            print(counter)


    def __len__(self):
        return len(self.capacities)

    def __getitem__(self, idx):
        return self.observations[idx], self.capacities[idx], self.video_quality[idx],self.audio_quality[idx]


def collate_fn(batch):
    observations, capacities,  video_quality, audio_quality = zip(*batch)

    min_length = min(obs.shape[0] for obs in observations)

    observations_truncated = [obs[:min_length] for obs in observations]
    capacities_truncated = [cap[:min_length] for cap in capacities]
    vq_truncated = [vq[:min_length] for vq in video_quality]
    aq_truncated = [aq[:min_length] for aq in audio_quality]

    observations_truncated = torch.stack(observations_truncated)
    capacities_truncated = torch.stack(capacities_truncated)
    vq_truncated = torch.stack(vq_truncated)
    aq_truncated = torch.stack(aq_truncated)

    return observations_truncated, capacities_truncated, vq_truncated, aq_truncated

def eval_policy(epoch,policy):

    data_files = glob.glob(os.path.join("./data", f'*.json'), recursive=True)
    for filename in tqdm(data_files, desc="Processing"):
        with open(filename,'r') as file:
            call_data = json.load(file)
        
        observations = np.expand_dims(np.asarray(call_data['observations'], dtype=np.float32), axis=0)
        bandwidth_predictions = np.asarray(call_data['bandwidth_predictions'], dtype=np.float32)
        true_capacity = np.asarray(call_data['true_capacity'], dtype=np.float32)

        test_model_predictions = []
        h = torch.zeros(1, 256).to(device)
        c = torch.zeros(1, 256).to(device)
        for t in range(observations.shape[1]):
            state = observations[0:1,t:t+1,:]
            action,_ , _ = policy.forward(state, h, c)
            # action=action.numpy().flatten()
            test_model_predictions.append(action[0,0,0])

        test_model_predictions = np.asarray(test_model_predictions, dtype=np.float32)

        fig = plt.figure(figsize=(8, 5))
        time_s = np.arange(0, observations.shape[1]*60,60)/1000
        plt.plot(time_s, test_model_predictions/1000, label='test_model', color='g')
        plt.plot(time_s, bandwidth_predictions/1000, label='BW Estimator '+call_data['policy_id'], linestyle='--',color='r',alpha=0.5)
        plt.plot(time_s, true_capacity/1000, label='True Capacity', color='k')
        plt.ylabel("Bandwidth (Kbps)")
        plt.xlabel("Call Duration (second)")
        plt.grid(True)
        plt.legend()
        figs_dir = f"./figs/evaluate/{epoch}"
        if not os.path.exists(figs_dir):
            os.makedirs(figs_dir)
        plt.savefig(os.path.join(figs_dir,os.path.basename(filename).replace(".json",".png")))
        plt.close()


if __name__ == "__main__":


    torch.multiprocessing.set_start_method('spawn')
    writer = SummaryWriter()  # TensorBoard writer

    # Define the model, loss function, and optimizer
    obs_dim= 150
    act_dim = 1
    hidden_size = 128
    batch_size = 25
    save_seq = 20
    eval_seq = 20
    seed = 0
    
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    kwargs = {
	    "state_dim": obs_dim,
        "action_dim": act_dim,
        "max_action": 1,
		"device" : device,
        "discount": 0.99,
        "tau": 0.005,
		"lmbda": 0.75,
		"phi"  : 0.05,

	}

    policy = Pioneer(**kwargs)

    replay_buffer = utils.ReplayBuffer(obs_dim, act_dim)


    # ------  We use pickle to accelerate data loading. This is not necessary ------
    with open('./dataset_testbed_sort.pkl', 'rb') as file:
        dataset = pickle.load(file)
        print("testbed dataset loaded!\n")
    
    with open('./dataset_emulated_train_sort.pkl','rb') as file:
        dataset += pickle.load(file)
        print("emulated dataset(train) loaded!\n")
        print(f"total length: {len(dataset)}")
    # ------------------------------------------------------------------------------
        
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Training loop
    num_epochs = 2000
    update_batch = 100

    
    for epoch in tqdm(range(num_epochs + 1), desc="Epochs"):
        counter = 0
        critic_loss_sum, actor_loss_sum, reward_sum = 0,0,0
        for batch_data in dataloader:
            sequences_padded, capacities,_ , _ = batch_data
            flag, seq_length, state_size = sequences_padded.shape
            if flag!= batch_size:
               continue
            counter += 1
            critic_loss,actor_loss,reward = policy.train(replay_buffer, batch_data, batch_size, writer)
            critic_loss_sum += critic_loss

            actor_loss_sum += actor_loss
            reward_sum += reward.sum().item()

        print(f'Epoch {epoch}: Critic Loss = {critic_loss_sum/counter}, Actor Loss = {actor_loss_sum/counter}')

        if epoch <= 20 :
            eval_policy(epoch,policy)
            policy.save(f"./checkpoints/checkpoint_epoch_{epoch}")
        else:
            if epoch % eval_seq == 0:
                eval_policy(epoch,policy)
            if epoch % save_seq == 0:
                policy.save(f"./checkpoints/checkpoint_epoch_{epoch}")

        writer.add_scalar('Epoch Training Loss - Critic', critic_loss_sum/counter, epoch)
        writer.add_scalar('Epoch Training Loss - Actor', actor_loss_sum/counter, epoch)
        writer.add_scalar('Epoch Reward',reward_sum/counter, epoch)

    writer.close()  # 关闭 TensorBoard writer
