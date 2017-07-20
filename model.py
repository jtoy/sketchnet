import torch
import torch.nn as nn
#import torchvision.models as models
from resnet import *
from vgg import *
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.autograd import Variable
from torchvision import transforms
from utils import to_var

mini_transform = transforms.Compose([ 
    transforms.ToPILImage(),
    transforms.Scale(20),
    transforms.ToTensor() ])
class EncoderCNN(nn.Module):
    def __init__(self, embed_size,pretrained=True):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        #net = resnet152(pretrained=False)
        print("pretrained is "+str(pretrained))
        #net = resnet152(pretrained)
        net = VGG('VGG16')
        modules = list(net.children())[:-1]      # delete the last fc layer.
        self.net = nn.Sequential(*modules)
        self.linear = nn.Linear(net.fc.in_features,embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()
        
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
        
    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.net(images)
        features = Variable(features.data)
        features = features.view(features.size(0),-1)
        features = self.bn(self.linear(features))
        return features
        #count = images.size()[0]
        #mini_ts = torch.FloatTensor(count,3,20,20)
        #for ii,image in enumerate(images): 
        #    mini_ts[ii] = mini_transform(image.data.cpu())
        #mini_ts = to_var(mini_ts.view(count,-1),volatile=False)
        #mini_ts = mini_ts.view(count,-1)
        #return to_var(torch.cat([features.data,mini_ts.data],1),volatile=False)
    
   
class ImagineLayer(nn.Module):
    def __init__(self):
        super(ImagineLayer, self).__init__()
    def forward(self,x):
        return x

class ImageLayer(nn.Module):
    def __init__(self,size):
        super(ImageLayer, self).__init__()
        self.linear = nn.Linear(size  ,4800)
        self.init_weights()
    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)
    def forward(self,x):
        x = self.linear(x)
        x = x.view(x.size(0),3,40,40)
        print("new size"+ str(x.size()))
        return x

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers,vocab):
        """Set the hyper-parameters and build the layers."""
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.classifier = nn.Linear(hidden_size,4800)
        self.vocab = vocab
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights."""
        self.embed.weight.data.uniform_(-0.1, 0.1)
        self.linear.weight.data.uniform_(-0.1, 0.1)
        self.linear.bias.data.fill_(0)
        self.classifier.weight.data.uniform_(-0.1, 0.1)
        self.classifier.bias.data.fill_(0)
        
    def forward(self, features, captions, lengths):
        """Decode image feature vectors and generates captions."""
        embeddings = self.embed(captions)
        #print("forward sizes")
        print("embedding size:"+str(embeddings.size()))
        #print(features.unsqueeze(1).size())
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        packed = pack_padded_sequence(embeddings, lengths, batch_first=True) 
        print("packed size"+str(packed.data.size())
        rnn_features, _ = self.lstm(packed)
        #features, _ = self.lstm(embeddings)
        #unpacked,unpacked_len = pad_packed_sequence(hiddens)
        #outputs = self.linear(rnn_features[0])
        #print("unpacked size"+str(unpacked.size()))
        #print("unpacked len"+str(unpacked_len))
        print("rnn_features:"+str(rnn_features.data.size()))
        outputs = self.classifier(rnn_features[0])
        outputs = outputs.view(outputs.size(0),3,40,40)
        return outputs
    
    def sample(self, features,length=20, states=None):
        """Samples captions for given image features (Greedy search)."""
        sampled_ids = []
        inputs = features.unsqueeze(1)
        for i in range(length):                                      # maximum sampling length
            hiddens, states = self.lstm(inputs, states)          # (batch_size, 1, hidden_size)
            outputs = self.linear(hiddens.squeeze(1))            # (batch_size, vocab_size)
            predicted = outputs.max(1)[1]
            sampled_ids.append(predicted)
            inputs = self.embed(predicted)
        sampled_ids = torch.cat(sampled_ids, 1)                  # (batch_size, 20)
        return sampled_ids.squeeze()
