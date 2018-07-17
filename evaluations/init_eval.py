
def init_eval(opt, dataloader, model):

    if opt.dataset == 'wider':

        if opt.model == 's3fd':
            from evaluations.eval_wider_s3fd import Evaluator
            evaluator = Evaluator(opt, dataloader, model)
        else:
            raise ValueError('None supported')

    elif opt.dataset == 'v_caption_patch_hangul':

        if opt.model == 'resnet':
            from evaluations.eval_v_caption_patch_hangul_resnet import Evaluator
            evaluator = Evaluator(opt, dataloader, model)
        else:
            raise ValueError('None supported')

    else:
        raise ValueError('Not supported')

    return evaluator