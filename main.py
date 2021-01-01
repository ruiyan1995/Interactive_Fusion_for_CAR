import yaml
import os
import argparse
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--cfg', type=str)
parser.add_argument('--ckpt_suffix', type=str, default='')
args, unknown_args = parser.parse_known_args()

curPath = os.path.dirname(os.path.realpath(__file__))
yamlPath = os.path.join(curPath, 'configs', args.cfg + '.yaml')


f = open(yamlPath, 'r', encoding='utf-8')
cfg = f.read()
cfg_dict = yaml.load(cfg)


command = 'python train.py '


# load args from yaml
# arg_type: DATA, MODEL, SOLVER
for arg_type in cfg_dict.keys():
    for arg_name in cfg_dict[arg_type].keys():
        arg_value = cfg_dict[arg_type][arg_name]
        if arg_value == 'default':
            pass
        else:
            command += '--%s %s '%(arg_name, str(arg_value))

# define and check 'ckpt'
ckpt = os.path.join(curPath, 'ckpt', '%s/%s'%(args.cfg, args.ckpt_suffix))
utils.path_check(ckpt)
command += '--ckpt %s '%(ckpt)

# load extra unknow args
for i, arg in enumerate(unknown_args):
    if arg.startswith(('-', '--')):
        arg_name, arg_value = arg, unknown_args[i+1]
        command += '%s %s '%(arg_name, str(arg_value))

command += '>%s/log'%(ckpt)
print(command)

# execute the command
os.system(command)