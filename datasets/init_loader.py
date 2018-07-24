import os
import torch
import random
import torch.utils.data.dataloader as dataloader
from transforms.init_transform import init_transforms


def init_dataloader_train(opt):

    batch_size = opt.batch_size_train

    dataset_name = opt.dataset
    transform = init_transforms(opt, 'train')

    if dataset_name.startswith('voc'):

        opt.num_classes = 21
        from datasets.voc import VOCDetection
        voc_root = os.path.join(opt.data_root_dir, 'VOCdevkit')

        # voc 2007
        if dataset_name == 'voc07':
            image_root = os.path.join(voc_root, 'VOC2007/JPEGImages')
            list_file = os.path.join('data/voc/voc07_trainval.txt')

        # voc 2012
        elif dataset_name == 'voc12':
            image_root = os.path.join(voc_root, 'VOC2012/JPEGImages')
            list_file = os.path.join('data/voc/voc12_trainval.txt')

        # voc 0712
        elif dataset_name == 'voc0712':
            image_root = os.path.join(voc_root, 'VOC0712/JPEGImages')
            list_file = [os.path.join('data/voc/voc07_trainval.txt'),
                         os.path.join('data/voc/voc12_trainval.txt')]

        else:
            raise ValueError('Not a valid dataset name')

        dataset = VOCDetection(root=image_root,
                               list_file=list_file,
                               transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)

    elif dataset_name == 'wider':
        opt.num_classes = 2
        from datasets.wider import WiderDetection
        wider_root = os.path.join(opt.data_root_dir, 'WIDER/WIDER_train/images')
        list_file = 'data/wider/wider_train.txt'
        dataset = WiderDetection(root=wider_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,)

    elif dataset_name == 'fddb':
        opt.num_classes = 2
        from datasets.fddb import FDDBDetection
        from preprocess.fddb.preprocess import parse_fddb_annotation

        wider_root = os.path.join(opt.data_root_dir, 'FDDB/imgs')
        annotation_dir = os.path.join(opt.data_root_dir, 'FDDB/anno')
        list_file = os.path.join(opt.project_root, 'data/fddb/fddb_train.txt')
        os.system('rm %s' % list_file)
        parse_fddb_annotation(wider_root, annotation_dir, 'train', list_file, opt.fold)

        dataset = FDDBDetection(root=wider_root,
                                list_file=list_file,
                                mode='train',
                                transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)

    elif dataset_name == 'v_caption_detection':
        from datasets.v_caption_detection import V_Caption_Detection
        data_root = os.path.join(opt.data_root_dir, 'V.DO/caption/background_result')
        list_file = 'data/v_caption_detection/caption_BG_train.txt'
        dataset = V_Caption_Detection(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers)

    elif dataset_name == 'v_caption_patch':
        from datasets.v_caption_patch import V_Caption_Patch
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = 'data/v_caption_detection/bghangul_patch_train.txt'
        dataset = V_Caption_Patch(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_num':
        from datasets.v_caption_patch_num import V_Caption_Patch_Num
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = 'data/v_caption_detection/number_patch_train.txt'
        dataset = V_Caption_Patch_Num(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_alp':
        from datasets.v_caption_patch_alp import V_Caption_Patch_Alp
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = ['data/v_caption_detection/bgalphabet_patch_train.txt','data/v_caption_detection/alphabet_patch_train.txt','data/v_caption_detection/bgalphabet_patch_val.txt']
        dataset = V_Caption_Patch_Alp(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_sym':
        from datasets.v_caption_patch_sym import V_Caption_Patch_Sym
        data_root = os.path.join(opt.data_root_dir, '')
        list_file =['data/v_caption_detection/symbol_patch_train.txt','data/v_caption_detection/bgsymbol_patch_train.txt','data/v_caption_detection/bgsymbol_patch_val.txt']
        dataset = V_Caption_Patch_Sym(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_type':
        opt.num_classes = 4
        from datasets.v_caption_patch_type import V_Caption_Resnet_Type
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = ['data/v_caption_detection/bgalphabet_patch_train.txt','data/v_caption_detection/bghangul_patch_train.txt','data/v_caption_detection/bgnumber_patch_train.txt',
                     'data/v_caption_detection/bgsymbol_patch_train.txt']
        dataset = V_Caption_Resnet_Type(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    else:
        raise ValueError('Not a valid dataset')

    return loader


def init_dataloader_valid(opt):

    batch_size = opt.batch_size_valid

    dataset_name = opt.dataset
    transform = init_transforms(opt, 'valid')

    if dataset_name.startswith('voc'):

        opt.num_classes = 21
        from datasets.voc import VOCDetection
        voc_root = os.path.join(opt.data_root_dir, 'VOCdevkit')

        # voc 2007
        if dataset_name == 'voc07':
            image_root = os.path.join(voc_root, 'VOC2007/JPEGImages')
            list_file = os.path.join('data/voc/voc07_test.txt')

        # voc 2012
        elif dataset_name == 'voc12':
            image_root = os.path.join(voc_root, 'VOC2012/JPEGImages')
            list_file = os.path.join('data/voc/voc12_test.txt')

        # voc 0712
        elif dataset_name == 'voc0712':
            image_root = os.path.join(voc_root, 'VOC0712/JPEGImages')
            list_file = os.path.join('data/voc/voc07_test.txt')
        else:
            raise ValueError('Not a valid dataset name')

        dataset = VOCDetection(root=image_root,
                               list_file=list_file,
                               transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers)

    elif dataset_name == 'wider':
        opt.num_classes = 2
        from datasets.wider import WiderDetection
        wider_root = os.path.join(opt.data_root_dir, 'WIDER/WIDER_val/images')
        list_file = 'data/wider/wider_val.txt'
        dataset = WiderDetection(root=wider_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,)

    elif dataset_name == 'fddb':
        opt.num_classes = 2
        from datasets.fddb import FDDBDetection
        from preprocess.fddb.preprocess import parse_fddb_annotation
        wider_root = os.path.join(opt.data_root_dir, 'FDDB/imgs')
        annotation_dir = os.path.join(opt.data_root_dir, 'FDDB/anno')
        list_file = os.path.join(opt.project_root,'data/fddb/fddb_val.txt')
        os.system('rm %s' % list_file)
        parse_fddb_annotation(wider_root, annotation_dir, 'valid', list_file, opt.fold)

        dataset = FDDBDetection(root=wider_root,
                                list_file=list_file,
                                mode='valid',
                                transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers)

    elif dataset_name == 'v_caption_detection':
        from datasets.v_caption_detection import V_Caption_Detection
        data_root = os.path.join(opt.data_root_dir, 'V.DO/caption/background_result')
        list_file = 'data/v_caption_detection/caption_BG_val.txt'
        dataset = V_Caption_Detection(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       )

    elif dataset_name == 'v_caption_patch':
        from datasets.v_caption_patch import V_Caption_Patch
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = 'data/v_caption_detection/bghangul_patch_val.txt'
        dataset = V_Caption_Patch(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_num':
        from datasets.v_caption_patch_num import V_Caption_Patch_Num
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = 'data/v_caption_detection/number_patch_val.txt'
        dataset = V_Caption_Patch_Num(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_alp':
        from datasets.v_caption_patch_alp import V_Caption_Patch_Alp
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = ['data/v_caption_detection/alphabet_patch_val.txt']
        dataset = V_Caption_Patch_Alp(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_sym':
        from datasets.v_caption_patch_sym import V_Caption_Patch_Sym
        data_root = os.path.join(opt.data_root_dir, '')
        list_file =['data/v_caption_detection/symbol_patch_val.txt']
        dataset = V_Caption_Patch_Sym(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_nas':
        from datasets.v_caption_patch_nas import V_Caption_Patch_NAS
        data_root = os.path.join(opt.data_root_dir, '')


        list_file = ['data/v_caption_detection/bgalphabet_patch_val.txt','data/v_caption_detection/bghangul_patch_val.txt', 'data/v_caption_detection/bgnumber_patch_val.txt' ]
        dataset = V_Caption_Patch_NAS(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_type':
        opt.num_classes = 4
        from datasets.v_caption_patch_type import V_Caption_Resnet_Type
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = ['data/v_caption_detection/bgalphabet_patch_val.txt','data/v_caption_detection/bghangul_patch_val.txt','data/v_caption_detection/bgnumber_patch_val.txt',
                     'data/v_caption_detection/bgsymbol_patch_val.txt']
        dataset = V_Caption_Resnet_Type(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    else:
        raise ValueError('Not a valid dataset')

    return loader


def init_dataloader_test(opt):

    batch_size = opt.batch_size_test

    dataset_name = opt.dataset
    transform = init_transforms(opt, 'test')

    if dataset_name == 'wider':
        opt.num_classes = 2
        from datasets.wider import WiderDetection
        wider_root = os.path.join(opt.data_root_dir, 'WIDER/WIDER_val/images')
        list_file = 'data/wider/wider_val.txt'
        dataset = WiderDetection(root=wider_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,)

    elif dataset_name == 'fddb':
        opt.num_classes = 2
        from datasets.fddb import FDDBDetection
        from preprocess.fddb.preprocess import parse_fddb_annotation
        wider_root = os.path.join(opt.data_root_dir, 'FDDB/imgs')
        annotation_dir = os.path.join(opt.data_root_dir, 'FDDB/anno')
        list_file = os.path.join(opt.project_root,'data/fddb/fddb_val.txt')
        os.system('rm %s' % list_file)
        parse_fddb_annotation(wider_root, annotation_dir, 'valid', list_file, opt.fold)

        dataset = FDDBDetection(root=wider_root,
                                list_file=list_file,
                                mode='valid',
                                transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers)

    elif dataset_name == 'v_caption_patch_nas':
        from datasets.v_caption_patch import V_Caption_Patch
        data_root = os.path.join(opt.data_root_dir, 'V.DO/V_Caption')
        list_file = 'data/v_caption_patch_hangul/patch_val.txt'
        dataset = V_Caption_Patch(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)

    elif dataset_name == 'v_caption_patch_type':
        from datasets.v_caption_patch_type import V_Caption_Resnet_Type
        data_root = os.path.join(opt.data_root_dir, '')
        list_file = 'data/v_caption_patch_hangul/patch_val.txt'
        dataset = V_Caption_Resnet_Type(root=data_root,
                                 list_file=list_file,
                                 transform=transform)

        loader = dataloader.DataLoader(dataset=dataset,
                                       batch_size=batch_size,
                                       shuffle=False,
                                       num_workers=opt.num_workers,
                                       pin_memory=True)


    else:
        raise ValueError('Not a valid dataset')

    return loader


if __name__ == '__main__':
    import argparse

    args = argparse.ArgumentParser()
    args.add_argument('--data_root_dir', type=str, default='/home/myunggi/Project/Data/')
    args.add_argument('--batch_size_train', type=int, default=32)
    args.add_argument('--num_workers', type=int, default=8)
    args.add_argument('--dataset', type=str, default='voc0712')
    args.add_argument('--model', type=str, default='ssd')
    args.add_argument('--steps', type=str, default=(8, 16, 32, 64, 100, 300))
    args.add_argument('--box_sizes', type=str, default=(30, 60, 111, 162, 213, 264, 315))
    args.add_argument('--aspect_ratios', type=str, default=((2,), (2,3), (2,3), (2,3), (2,), (2,)))
    args.add_argument('--fm_sizes', type=str, default=(38, 19, 10, 5, 3, 1))
    args.add_argument('--img_size', type=int, default=512)

    opt = args.parse_args()

    train_loader = init_dataloader_train(opt)
    valid_loader = init_dataloader_valid(opt)

    print('training data size: %d' % len(train_loader.dataset))
    print('validation data size: %d' % len(valid_loader.dataset))

    for i, data in enumerate(train_loader):

        print(data)