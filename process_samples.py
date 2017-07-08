import argparse,os,glob,re,json
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from jinja2 import Environment, FileSystemLoader
def entries_to_remove(entries, the_dict):
    new_dict = the_dict.copy()
    for key in entries:
        if key in new_dict:
            del new_dict[key]
    return new_dict
def process(args):
    print(args)
    model_path = "./models/"+args.name
    out_path = "./output/"+args.name
    os.system("mkdir -p "+ out_path)
    print(model_path)
    if args.decoder is None:
        decoder = max(glob.iglob(os.path.join(model_path,'decoder*')), key=os.path.getctime)
    else:
        decoder = args.decoder
    if args.encoder is None:
        encoder = max(glob.iglob(os.path.join(model_path,'encoder*')), key=os.path.getctime)
    else:
        encoder = args.encoder
    with open(model_path+"/parameters.json", 'r') as f:
        parameters = json.load(f)
    if args.embed_size is not None:
        embed_size = str(parameters['embed_size'])
    else:
        embed_size = str(args.num_layers)
    if args.num_layers is not None:
        num_layers = str(parameters['num_layers'])
    else:
        num_layers = str(args.num_layers)
    if args.hidden_size is None:
       hidden_size = str(parameters['hidden_size'])
    else:
       hidden_size = str(args.hidden_size)
    any_fails = False
    results = []
    for root, dirs, files in os.walk(os.path.abspath(args.images)):
      for file in files:
        if any_fails == False and (file.endswith(".jpg") or file.endswith(".png")):
          image_path = os.path.join(root, file)
          number = re.findall(r'\d+', file)[0]
          code_path = out_path + "/" + str(number) + "_code.txt"
          c = "python sample.py --length="+ str(args.length)+" --vocab_path="+model_path+"/vocab.pkl --image="+image_path+" --decoder="+decoder+ " --encoder="+encoder +" --num_layers="+num_layers  +" --hidden_size="+hidden_size+ " --embed_size=" + embed_size+ " > "+ code_path
          print(c)
          r = os.system(c)
          print(r)
          print("code:")
          result = {}
          with open(code_path, 'r') as fin:
            print(fin.read())
          result['input'] = image_path
          if r == 0:
            #if file is not empty
            fp =  os.path.abspath(os.path.join(out_path,str(number)+"_code.txt"))
            cp =  os.path.abspath("output/"+str(number)+"_cleaned.txt")
            op =  os.path.abspath(os.path.join(out_path,str(number)+".jpg"))
            result['code'] = fp
            result['cleaned_code'] = cp
            result['output'] = op
            #99% the last line will be incomplete 
            os.system("head -n-1 "+ fp +" > " + cp)
            c = "xvfb-run -a jruby ../processing_data_generator/generator/output.rb " + cp + " " + op
            #c = "xvfb-run -a jruby ../processing_data_generator/generator/scaling_output.rb " + cp + " " + op
            print(c)
            os.system(c)
          else:
            pass
            #any_fails = True
          results.append(result)

    if args.s3:
        #TODO check if keys are null
        s3conn = S3Connection(os.environ.get('AWS_ACCESS_KEY_ID'), os.environ.get('AWS_SECRET_ACCESS_KEY'))
        bucket = s3conn.get_bucket('sketchnet')
        path = 'experiments/'+args.name+'/'
        for result in results:
            for key in ['input','output']:
                k = Key(bucket)
                k.key=path+key+os.path.basename(result[key])
                k.set_contents_from_filename(result[key], reduced_redundancy=True)
          
        env = Environment(loader=FileSystemLoader("./"))
        template = env.get_template('results.html')
        cleaned_params = entries_to_remove(['name','images','s3','transfer'], vars(args)) 
        cleaned_results = []
        for result in results:
            r = {}
            r['input']='input'+os.path.basename(result['input'])
            r['output']='output'+os.path.basename(result['output'])
            with open(result['code'], 'r') as f:
                r['code']=f.read()
            with open(result['cleaned_code'], 'r') as f:
                r['cleaned_code']=f.read()
            cleaned_results.append(r)
        rendered_template = template.render(experiment=args.name, parameters=cleaned_params,results=cleaned_results)
        k = Key(bucket)
        k.content_type = 'text/html'
        k.key=path+'index.html'
        k.set_contents_from_string(rendered_template, reduced_redundancy=True)
    if args.transfer: 
        r = os.system("tar -cvf output.tar output && curl --upload-file ./output.tar https://transfer.sh/output.tar")
        print(r)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, default="./test_images" , help='path to images')
    parser.add_argument('--name', type=str, help='model name')
    parser.add_argument('--encoder', type=str,  help='path for trained encoder')
    parser.add_argument('--decoder', type=str,  help='path for trained decoder')
    parser.add_argument('--crop_size', type=int, default=224, help='size for center cropping images')
    # Model parameters (should be same as paramters in train.py)
    parser.add_argument('--embed_size', type=int , default=256, help='dimension of word embedding vectors')
    parser.add_argument('--hidden_size', type=int , default=512, help='dimension of lstm hidden states')
    parser.add_argument('--num_layers', type=int , default=1 , help='number of layers in lstm')
    parser.add_argument('--length', type=int, default=200, help='length of output')
    parser.add_argument('--s3', action='store_false', help='save results to s3 bucket')
    parser.add_argument('--transfer', action='store_true', help='use transfer.sh to upload')
    args = parser.parse_args()
    process(args)
