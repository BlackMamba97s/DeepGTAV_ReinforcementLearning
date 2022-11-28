import time

from deepgtav.messages import Start, Stop, Scenario, Commands, frame2numpy
from deepgtav.client import Client

import argparse
from torch import optim
import torch
import numpy as np
from data_structures.buffer_istance import BufferIstance
from data_structures.model import RNNResnet
from utils import count_down
from utils.util import *
from .data_structures.data_container import DataContainer

class DummyModel:
    def run(self, action):
        return execute_agent_action(action)

class Model:
    def run(self, action):
        return execute_agent_action(action)





def training(agent_model): # Actor-Critic RL Method
    counter = 1
    net = RNNResnet().cuda()
    train_parameters = [param for param in net.parameters()
                        if param.requires_grad == True]
    optimizer = optim.SGD(train_parameters, lr=0.001, momentum=0.9)

    count_down(1)  # 2 sec cd

    # all needed for rnn sequences and data for prediction
    data_c = DataContainer() #setting parameters
    goodEpisode = 0  # round completed successfully
    badEpisode = 0  # otherwise

    while True:
        for i in range(data_c.learn_batch_num):  # each counter tic run a number of act_batch_num times
            message = client.recvMessage() # get message from deep GTA_V
            if no_prediction(data_c.datas): # never enter first time because we have no prediction
                data, x, actions, state = action_by_prediction(data_c)

            else: # first time enter, no prediction data, azione randomica.
                data, info, x = random_action_for_training(agent_model,data_c, message, x)

            #start the reward part
            reward = give_reward(data, data_c.previous_datas)
            lost = False
            data_c.vehicle_speed_list.append(data.car.Speed)
            data_c.wheel_speeds.append(data.car.WheelSpeed)
            if is_vehicle_stop(data_c): # check if vehicle is in control
                max_speed, mean_wheel_speed = max_vec_speed(data_c), mean_vec_speed(data_c)
                if is_vehicle_idle(max_speed,mean_wheel_speed):  # is the vehicle idle? colpire il muro o macchina
                    reward = -5
                    lost = True
                data_c.vehicle_speed_list.clear()
                data_c.wheel_speeds.clear()
            if distance(data.car.Position, data.endPosition) > 220: # too far from finish
                reward = -10
                lost = True
            if distance(data.car.Position, data.endPosition) < 20: # near or in the finish line
                reward = 25
                lost = True
            reward = reward / 10
            if train_data_ok(data_c):  # when the length of sequence reach the trainable standard
                buffer_istance = BufferIstance()
                buffer_istance.set_buffer_new_data(data_c,state,reward,down)
                buffer_istance.actions = actions.cpu().data.numpy() if isinstance(
                    actions, torch.Tensor) else actions
                if len(data_c.buffer) >= data_c.max_buff_num: #is buffer full?
                    data_c.buffer.pop()
                data_c.buffer.append(buffer_istance)
            if data_c.num_step_from_last_down >= data_c.max_num_step_a_episode: # too many steps and still not finish
                lost = True
            if lost: # means we lost the round
                #back_to_start_point()
                time.sleep(1)
                if reward >= 10:  # good episode
                    goodEpisode += 1
                    break
                if lost and reward < 0: # bad episode
                    badEpisode += 1
                    break
            if lost:
                data_c.num_step_from_last_down = 0
            if not lost:
                data_c.num_step_from_last_down += 1

        # loss and learn part
        if len(data_c.buffer) > data_c.least_buffer_for_learning:  # if the buffer got enough istances to be ok for learning
            down_index = -1
            for i, record in enumerate(data_c.buffer):
                if record.down == True:
                    down_index = i
                    break
            if down_index != -1 and down_index < data_c.learn_batch_num:
                data_c.learn_batch_num = down_index
            rewards = []
            for i, record in enumerate(data_c.buffer[:data_c.learn_batch_num]):
                for j, next_record in enumerate(data_c.buffer[i+1: i+data_c.n]):
                  # 使用 discount 计算 target reward
                    if next_record.down == False:
                        reward += data_c.gamma**(j+1) * next_record.reward
                    else:
                        break
                rewards.append(reward)
            p_policise, p_states, states = process_datas(data_c, data_c.learn_batch_num, net)
            _states = []
            for i, record in enumerate(data_c.buffer[: data_c.learn_batch_num]):
                state = states[i]
                for j, next_record in enumerate(data_c.buffer[i+1: i+data_c.n]):
                    if next_record.down == False:
                        state += data_c.gamma**(j+1) * states[i+j]
                    else:
                        break
                _states.append(state)
            rewards = torch.tensor([[torch.tensor(reward, dtype=dtype, device=device)]
                                    for reward in rewards], dtype=dtype, device=device)
            states = torch.stack(_states)
            yes = rewards + states
            state_loss = torch.mean(torch.pow(p_states - yes, 2))
            advantages = - p_states
            max_policy, _ = torch.max(p_policise, 1)
            min_policy, _ = torch.min(p_policise, 1)
            smaller_than_0_index = (max_policy < 0)
            smaller_than_0_index = smaller_than_0_index.float()
            max_policy = max_policy - smaller_than_0_index * \
                min_policy
            policise_loss = torch.mean(
                torch.log(max_policy) * advantages)
            loss = state_loss + policise_loss
            optimizer.zero_grad()  # reset
            loss.backward()  # calculate gradient
            torch.nn.utils.clip_grad_norm_(train_parameters, 0.5)  # cleanup del gradiente, normalizzato
            optimizer.step()  # discesa gradiente

        counter += 1
        if counter % 100 == 0 and counter != 0:
            # time to save all the state dict
            torch.save(
                net.state_dict(), "./saved_model/AC_resnet_{}_state_dict.pt".format(counter))
            print("model saved in {} iterations".format(counter))










    # We receive a message as a Python dictionary
    message = client.recvMessage()
    # The frame is a numpy array that can we pass through a CNN for example
    image = frame2numpy(message['frame'], (320, 160))
    # The frame is a numpy array and can be displayed using OpenCV or similar

    # We send the commands predicted by the agent back to DeepGTAV to control the vehicle
    client.sendMessage(Commands(commands[0], commands[1], commands[2]))


def process_datas(data_c, learn_batch_num, net):
    _p_xes = torch.cat([torch.tensor(np.array(x), dtype=dtype, device=device)
                        for record in data_c.buffer[:learn_batch_num] for x in record.p_xes],
                       dim=0)  # shape (learn_batch_num,len_rnn_seq) => (learn_batch_num * len_rnn_seq,)
    _p_infoes = torch.cat([torch.tensor(np.array(info), dtype=dtype, device=device)
                           for record in data_c.buffer[:learn_batch_num] for info in record.p_infoes],
                          dim=0)  # shape (learn_batch_num,len_rnn_seq) => (learn_batch_num * len_rnn_seq,)
    _xes = torch.cat([torch.tensor(np.array(x), dtype=dtype, device=device)
                      for record in data_c.buffer[:learn_batch_num + data_c.n] for x in record.xes],
                     dim=0)  # shape (learn_batch_num + n,len_rnn_seq) => ((learn_batch_num + n) * len_rnn_seq,)
    _infoes = torch.cat([torch.tensor(np.array(
        info), dtype=dtype, device=device) for record in data_c.buffer[:learn_batch_num + n] for info in record.infoes],
        dim=0)  # shape (learn_batch_num + n,len_rnn_seq) => ((learn_batch_num + n) * len_rnn_seq,)
    p_policise, p_states = net.forward(_p_xes, _p_infoes)
    _, states = net.forward(_xes, _infoes)
    return p_policise, p_states, states


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port.
    client = Client(ip=args.host, port=args.port)

    scenario = Scenario(drivingMode=-1)  # manual driving

    # Send the Start request to DeepGTAV. Dataset is set as default, we only receive frames at 10Hz (320, 160)
    client.sendMessage(Start(scenario=scenario))

    agent_model = DummyModel()


    # base_train(agent_model, client)
    training(agent_model)

    client.sendMessage(Stop())
    client.close()
