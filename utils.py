import json
import time
from torch.autograd import Variable 

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)

def calculate_image_norm(folder):
    pass

def add_log_entry(name,start_time, params):
    localtime = time.asctime( time.localtime(time.time()) )
    line = "experiment: " + name +" | started training at: " +localtime + " | params: " + json.dumps(params) +"\n" 
    if name == "test":
        print(line)
    else:
        with open("log.md", 'a') as f:
            f.write(line)

def add_log_results(name,run_time,loss):
    line = "finished experiment: " + name +" | run time: " +run_time + " | loss: " + loss +"\n"
    if name == "test":
        print(line)
    else:
        with open("log.md", 'a') as f:
            f.write(line)
