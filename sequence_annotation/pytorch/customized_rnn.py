class ConcatRNN(nn.Module):
    def __init__(self,input_size,num_layers,rnn_cell_class,
                 rnns_as_output=True,rnn_setting=None):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn_setting = rnn_setting or {}
        self.rnn_cell_class = rnn_cell_class
        self.rnns_as_output = rnns_as_output
        in_num = input_size
        self.rnns = []
        for index in range(self.num_layers):
            hidden_size = self.hidden_size[index]
            rnn_cell = self.rnn_cell_class(input_size=in_num,hidden_size=hidden_size)
            rnn = RNN(rnn_cell,**rnn_setting)
            self.rnns.append(rnn)
            setattr(self, 'rnn_'+str(index), rnn)
            in_num += hidden_size*2
        if self.rnns_as_output:
            self.hidden_size = hidden_size*2*self.num_layers
        else:
            self.hidden_size = in_num
        self._build_layers()
        self.reset_parameters()
            
    def reset_parameters(self):
        for layer in self.children():
            layer.reset_parameters()

    def forward(self, x, lengths):
        #X shape : N,C,L
        x_ = x.transpose(1, 2)
        distribution_output = {}
        rnn_output = {}
        for name in self.rnns[0].output_names:
            if name not in rnn_output.keys():
                rnn_output[name] = []
        rnn_output['after_rnn'] = []
        for index in range(self.num_layers):
            rnn = self.rnns[index]
            pre_x = x_
            x_ = rnn(x_,lengths)
            if hasattr(rnn,'output_names'):
                for name,item in zip(rnn.output_names,x_):
                    rnn_output[name].append(item)
                    values = item.transpose(1,2)
                    distribution_output["rnn_"+str(index)+"_"+name] = values
                x_ = x_[rnn.output_names.index('new_h')]
            rnn_output['after_rnn'].append(x_)
            x_,lengths = Concat(dim=1)([pre_x.transpose(1,2),x_.transpose(1,2)],lengths)
            x_ = x_.transpose(1,2)
        for name,value in rnn_output.items():
            value = [item.transpose(1,2) for item in value]
            temp,lengths = Concat(dim=1)(value,lengths)
            rnn_output[name] = temp
        if self.rnns_as_output:
            x = rnn_output['after_rnn']
        else:
            x = x_.transpose(1, 2)
        distribution_output['rnn_result'] = rnn_output['after_rnn']
        return x,lengths,distribution_output

class GatedIndRnnCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self._gate_num = 1
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gate_weights_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.gate_weights_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.weights_i = nn.Parameter(torch.empty(hidden_size, input_size))
        self.i_gate_bias = nn.Parameter(torch.empty(hidden_size))
        self.f_gate_bias = nn.Parameter(torch.empty(hidden_size))
        self.input_bias = nn.Parameter(torch.empty(hidden_size))
        self.gate_function =  NoisyHardSigmoid()
        self.ln = nn.LayerNorm(hidden_size)
        self.recurrent_function = NoisyReLU()
        self.reset_parameters()
        self.output_names = ['new_h','pre_gate_f','pre_gate_i','gate_f','gate_i','values_i','pre_h']
    def reset_parameters(self):
        gate_bound_ih = (1/((self.input_size**0.5)))
        gate_bound_hh = (1/((self.input_size**0.5)))
        input_bound = (1/(self.input_size**0.5))
        uniform_(self.gate_weights_ih,-gate_bound_ih,gate_bound_ih)
        uniform_(self.gate_weights_hh,-gate_bound_hh,gate_bound_hh)
        uniform_(self.weights_i,-input_bound,input_bound)
        constant_(self.i_gate_bias,1)
        constant_(self.f_gate_bias,1)
        constant_(self.input_bias,0)
    def forward(self, input, state):
        #input shape should be (number,feature size)
        values_i = F.linear(input, self.weights_i,self.input_bias)
        pre_gate_ih = F.linear(input, self.gate_weights_ih)
        pre_gate_hh = F.linear(self.ln(state), self.gate_weights_hh)
        pre_gate_f = pre_gate_ih + self.f_gate_bias
        pre_gate_i = pre_gate_hh + self.i_gate_bias
        gate_f = self.gate_function(pre_gate_f,training=self.training)
        gate_i = self.gate_function(pre_gate_i,training=self.training)
        pre_h = state*gate_f+ values_i*gate_i
        new_h = self.recurrent_function(pre_h,self.training)
        return new_h,pre_gate_f,pre_gate_i,gate_f,gate_i,values_i,pre_h