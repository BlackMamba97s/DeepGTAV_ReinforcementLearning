from deepgtav.messages import frame2numpy, Commands
import random
import torch
import time

def execute_agent_action(action):
    return [1.0, 0.0, 1.0]  # throttle, brake, steering



def get_image_of_agent_driving(message): # return the image of that frame got by deepGTAV by message
    return frame2numpy(message['frame'], (320, 160))


def get_data_from_input_executed():
    pass


def random_action():
    # da fare un attimo
    return execute_agent_action(None)


def gta_agent_control(agent, actions):
    agent.run("forward")

def choose_action_to_do():
    rand_num = random.random()
    if rand_num > 0.9:  # 10% chance of explore randomly
        actions = random_action()
        gta_agent_control(list(actions))  # the actual random action
    else:
        gta_agent_control()  # vediamo


def action_by_prediction(data_c, message, net, x):
    if len(data_c.previous_dats) >= data_c.len_rnn_seq:
        data_c.previous_datas.pop()  # pop first data, used as oldest
    data_c.previous_datas.append(data_c.data)  # put newest at the end
    if len(data_c.previous_xs) >= data_c.len_rnn_seq:
        data_c.previous_xs.pop()
    data_c.previous_xs.append(x)
    if len(data_c.previous_infos) >= data_c.len_rnn_seq:
        data_c.previous_infos.pop()
    data_c.previous_infos.append(data_c.info)
    with torch.no_grad():  # only prediction, no train
        actions, state = net.forward(
            torch.cat(data_c.previous_xs, dim=0), torch.cat(data_c.previous_infos, dim=0))
    choose_action_to_do()  # si spiega da se.
    time.sleep(0.1)  # uso le sleep per dare tempo di eseguire l'azione
    x = get_image_of_agent_driving(message)  # ricorda di processare
    if len(data_c.xs) >= data_c.len_rnn_seq:
        data_c.xs.pop()
    data_c.xs.append(x)
    data, info = get_data_from_input_executed()
    if len(data_c.datas) >= data_c.len_rnn_seq:
        data_c.datas.pop()
    data_c.datas.append(info)
    if len(data_c.infos) >= data_c.len_rnn_seq:
        data_c.infos.pop()
    data_c.infos.append(info)
    return data, x, actions, state


def random_action_for_training(agent_model, datas, infos, message, x, xs):
    actions = random_action()  # we get a random action to do
    gta_agent_control(agent_model, actions)  # we execute the action
    time.sleep(0.1)
    state = 0
    x = get_image_of_agent_driving(message)
    # devo ricordarmi di pre processare l'immagine, vediamo come poi
    data, info = get_data_from_input_executed()  # get game data after action is executed
    xs.append(x)
    datas.append(info)
    infos.append(info)
    return data, info, x



def give_reward(data, previous_datas):
    # distinguo tra : have datas from previous frame, otherwise
    if len(previous_datas) > 0:
        health_reward = 0
        if data.car.Health - previous_datas[0].car.Health < 0: # se ho dati dal previous frame allora non sono all'inizio e controllo se la macchina sia danneggiata
            health_reward = (data.car.Health -
                             previous_datas[0].car.Health) * 2
        reward = calculate_reward(data) + health_reward
    else: #otherwise
        reward = calculate_reward(data)
    return reward

def no_prediction(datas):
    return len(datas) > 0


def base_train(agent, client):
    while True:
        message = client.recvMessage()  # get message from deep GTA_V

        commands =agent.run(None)
        print(commands)
        client.sendMessage(Commands(commands[0], commands[1], commands[2]))


def is_vehicle_stop(data_c):
    return len(data_c.vehicle_speed_list) > data_c.num_step_to_check_trapped


def max_vec_speed(data_c):
    return max(data_c.vehicle_speed_list)


def mean_vec_speed(data_c):
    return sum(data_c.wheel_speeds) / \
                       len(data_c.wheel_speeds)


def is_vehicle_idle(max_speed, mean_wheel_speed):
    return max_speed < 1 and mean_wheel_speed > 6


def train_data_ok(data_c):
    return (len(data_c.datas) >= data_c.len_rnn_seq) \
    and (len(data_c.previous_datas) >= data_c.len_rnn_seq) \
    and (len(data_c.previous_xs) >= data_c.len_rnn_seq) \
    and (len(data_c.xs) >= data_c.len_rnn_seq) \
    and (len(data_c.infos) >= data_c.len_rnn_seq) \
    and (len(data_c.previous_infos) >= data_c.len_rnn_seq)

def calculate_reward(data):
    time_since = 150  # ms
    against_traffic = time_since > data.time_since_player_drove_against_traffic >= 0
    drove_on_pavement = time_since > data.time_since_player_drove_on_pavement >= 0
    hit_ped = time_since > data.time_since_player_hit_ped >= 0
    hit_vehicle = time_since > data.time_since_player_hit_vehicle >= 0
    num_near_by_vehicle = len(data.near_by_vehicles)
    num_near_by_peds = len(data.near_by_peds)
    diff_Speed = abs(data.car.Speed - data.car.WheelSpeed)
    # potrei aggiungere qualcosa inerente alla distanza dalla fine, ma come disse mirko potrebbe essere spinoso (strada con curve)
    reward =+ data.onRoad * 20\
        - against_traffic * 8 \
        - drove_on_pavement * 8\
        - hit_ped * 6\
        - hit_vehicle * 6\
        + num_near_by_peds * 4\
        - abs(data.car.Speed - 30)\
        - data.is_player_in_water * 20 \
        + num_near_by_vehicle * 4\
        - len(data.near_by_touching_peds) * 4\
        - len(data.near_by_touching_props) * 2  \
        - diff_Speed * 8\
        - len(data.near_by_touching_vehicles) * 2\
        + 800

    if reward < 1:
        reward = 1
    return reward / 200