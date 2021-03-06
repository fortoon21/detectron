from configs.ssd import ssd300_config, ssd512_config
from configs.fpnssd import fpnssd_config
from configs.retinanet import retinanet_config
from configs.refinedet import refinedet_config
from configs.s3fd import s3fd_config
from configs.resnet import resnet_config
from configs.chanet import chanet_config
from configs.fpnssd_v_caption import fpnssd_v_caption_config
from configs.resnet_nas import resnet_nas_config
from configs.resnet_type import resnet_type_config
from configs.resnet_num import resnet_num_config
from configs.resnet_sym import resnet_sym_config
from configs.resnet_alp import resnet_alp_config

def init_config(args):

    model_name = args.parse_args().model

    if model_name == 'ssd300':
        opt = ssd300_config(args)

    elif model_name == 'ssd512':
        opt = ssd512_config(args)

    elif model_name == 'fpnssd':
        opt = fpnssd_config(args)

    elif model_name == 'retinanet':
        opt = retinanet_config(args)

    elif model_name == 'refinedet':
        opt = refinedet_config(args)

    elif model_name == 's3fd':
        opt = s3fd_config(args)

    elif model_name == 'fpnssd_v_caption':
        opt = fpnssd_v_caption_config(args)

    elif model_name == 'resnet':
        opt = resnet_config(args)

    elif model_name == 'chanet':
        opt = chanet_config(args)

    elif model_name == 'resnet_nas':
        opt = resnet_nas_config(args)

    elif model_name=='resnet_type':
        opt=resnet_type_config(args)

    elif model_name=='resnet_num':
        opt=resnet_num_config(args)

    elif model_name=='resnet_alp':
        opt=resnet_alp_config(args)

    elif model_name=='resnet_sym':
        opt=resnet_sym_config(args)
    else:
        raise ValueError('[!] model not found!')

    return opt
