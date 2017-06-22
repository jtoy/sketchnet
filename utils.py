import json
import time

def calculate_image_norm(folder):
    pass

def add_log_entry(name,start_time, params):
    localtime = time.asctime( time.localtime(time.time()) )
    with open("log.md", 'a') as f:
        f.write("experiment: " + name +" | started training at: " +localtime + " | params: " + json.dumps(params) +"\n" )

def add_log_results(name,run_time,loss):
    with open("log.md", 'a') as f:
        f.write("finished experiment: " + name +" | run time: " +run_time + " | loss: " + loss +"\n")
