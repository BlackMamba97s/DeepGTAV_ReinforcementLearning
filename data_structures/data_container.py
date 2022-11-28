

class DataContainer():
    def __init__(self):
        self.len_rnn_seq = 3  # length of the sequence that enter the rnn
        self.previous_datas = []  # list of data used in the prev prediction，length = len_rnn_seq ( se la sequenza è 3 è normale)
        self.previous_xs = []  # x is the image, previous
        self.previous_infos = []  # info used in the prediction of the previous frame (rivedi sempre len_rnn_seq)
        self.datas = []  # prediction for current frame
        self.xs = []
        self.infos = []

        self.num_step_from_last_down = 0  # step ==/== counter
        self.max_num_step_a_episode = 1500
        self.buffer = []  # record playback
        self.max_buff_cap = 500
        self.act_batch_num = 20  # executions before each training
        self.learn_batch_num = 24  # batch size x train
        self.num_step_to_check_trapped = 6  # if wheels are not spinning (trapped)
        self.vehicle_speed_list = []  #
        self.wheel_speeds = []  # list of wheel speed of num_step_to_check_trapped
        self.gamma = 0.97  # reward discount (discount formula), vista nel video
        self.n = 16  # calculate the discount and reward for n sequences
        self.least_buffer_for_learning = self.learn_batch_num + n

