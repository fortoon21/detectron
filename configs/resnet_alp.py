
def resnet_alp_config(args):

    args.add_argument('--resnet_model', type=str, default='resnet50')
    args.add_argument('--img_size', type=int, default=96)

    # v_caption_patch_hangul
    args.add_argument('--num_classes',  type=int, default=52)

    args.add_argument('--use_pretrained', type=bool, default=True)

    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=1e-4)
    args.add_argument('--accum_grad', type=int, default=2)

    opt = args.parse_args()

    return opt