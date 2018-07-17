
def fpnssd_v_caption_config(args):

    args.add_argument('--img_size', type=int, default=(1280, 720))
    args.add_argument('--num_classes', type=int, default=11)

    args.add_argument('--use_pretrained', type=bool, default=True)

    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=5e-4)
    args.add_argument('--lr_steps', type=str, default=[80, 100])
    args.add_argument('--lr_decay_rate', type=float, default=0.1)
    args.add_argument('--accum_grad', type=int, default=1)

    opt = args.parse_args()

    return opt