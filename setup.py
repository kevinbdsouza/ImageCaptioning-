# Your code goes here
encoder = models.vgg16(pretrained=True).cuda()
modified_classifier = nn.Sequential(*list(encoder.classifier.children())[:-1])

encoder.train()
modified_classifier

# Your code goes here
encoder_hidden_size = 4096
hidden_size = 512

class conv(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(conv, self).__init__()
        self.hidden_size = hidden_size

        self.out = nn.Linear(input_size, hidden_size)

    def forward(self, input):
        output = self.out(input)
        return output

converter = conv(encoder_hidden_size, hidden_size).cuda() 
converter 


# Your code goes here
encoder_hidden_size = 4096
input_size = wordEncodingSize
hidden_size = 512
output_size = vocabularySize


class DecoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input,hidden,state):
        output = F.relu(input)
        output,(hidden,state) = self.lstm(output,(hidden,state))
        output = self.out(output)
        output = F.log_softmax(output.squeeze())
        return output.unsqueeze(0),hidden,state

decoder = DecoderLSTM(input_size, hidden_size, output_size).cuda() 
decoder 
