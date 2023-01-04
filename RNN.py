import torch
import torch.nn as nn
# inspiration: https://www.youtube.com/watch?v=WEV61GmmPrk&list=PLiDmKRJhglti6HwdDP9kEItTlHMZCDPk_&index=4&t=184s
class RNN(nn.Module):
    # hidden : layer for the recurent part
    # output : the output of the rnn
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN,self).__init__()
        self.hidden_size = hidden_size
        self.inputToHidden = nn.Linear(input_size + hidden_size, hidden_size)
        self.inputToOutput = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1) #TODO google this, might need something else

    def forward(self,input_tensor, hidden_tensor):
        combined = torch.cat((input_tensor, hidden_tensor),1)
        hidden = self.inputToHidden(combined)
        output = self.inputToOutput(combined)
        output = self.softmax(output)
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1,self.hidden_size)





