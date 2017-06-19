import argparse,os,glob,re
parser = argparse.ArgumentParser()
parser.add_argument('--images', type=str, default="./test_images" , help='path to images')
parser.add_argument('--name', type=str, help='model name')
#parser.add_argument('--model_path', type=str, default='./models/', help='path for trained encoder/decoder/vocab')
#parser.add_argument('--output_path', type=str, default='./models/', help='path for trained encoder/decoder/vocab')
parser.add_argument('--crop_size', type=int, default=224, help='size for center cropping images')
# Model parameters (should be same as paramters in train.py)
parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int , default=1 , help='number of layers in lstm')
parser.add_argument('--length', type=int, default=200, help='length of output')
args = parser.parse_args()
model_path = "./models/"+args.name
out_path = "./output/"+args.name
os.system("mkdir -p "+ out_path)
decoder = max(glob.iglob(os.path.join(model_path,'decoder*')), key=os.path.getctime)
encoder = max(glob.iglob(os.path.join(model_path,'encoder*')), key=os.path.getctime)
any_fails = False
for root, dirs, files in os.walk(os.path.abspath(args.images)):
  for file in files:
    if any_fails == False and (file.endswith(".jpg") or file.endswith(".png")):
      image_path = os.path.join(root, file)
      number = re.findall(r'\d+', file)[0]
      code_path = out_path + "/" + str(number) + "_code.txt"
      c = "python sample.py --length="+ str(args.length)+" --vocab_path="+model_path+"/vocab.pkl --image="+image_path+" --decoder="+decoder+ " --encoder="+encoder +" --num_layers="+str(args.num_layers)  +" --embed_size=" + str(args.embed_size)+ " > "+ code_path
      with open(code_path, 'r') as fin:
        print(fin.read())
      print(c)
      r = os.system(c)
      print(r)
      print("CODE")
      if r == 0:
        #if file is not empty
        fp =  os.path.abspath(os.path.join(out_path,str(number)+"_code.txt"))
        cp =  os.path.abspath("output/"+str(number)+"_cleaned.txt")
        op =  os.path.abspath(os.path.join(out_path,str(number)+".jpg"))
        #99% the last line will be incomplete 
        os.system("head -n-1 "+ fp +" > " + cp)
        c = "xvfb-run -a jruby ../jruby/generator/scaling_output.rb " + cp + " " + op
        print(c)
        print("SDASDD")
        os.system(c)
      else:
        pass
        #any_fails = True

#r = os.system("tar -cvf output.tar output && curl --upload-file ./output.tar https://transfer.sh/output.tar")
#print(r)
