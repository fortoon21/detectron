
def ssd300_config(args):

    args.add_argument('--steps', type=str, default=(8, 16, 32, 64, 100, 300))
    args.add_argument('--box_sizes', type=str, default=(30, 60, 111, 162, 213, 264, 315))
    args.add_argument('--aspect_ratios', type=str, default=((2,), (2, 3), (2, 3), (2, 3), (2,), (2,)))
    args.add_argument('--fm_sizes', type=str, default=(38, 19, 10, 5, 3, 1))
    args.add_argument('--img_size', type=int, default=300)

    args.add_argument('--use_pretrained', type=bool, default=True)

    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=1e-4)
    args.add_argument('--lr_steps', type=str, default=[190, 250])
    args.add_argument('--lr_decay_rate', type=float, default=0.1)
    args.add_argument('--accum_grad', type=int, default=2)

    # fddb settings
    args.add_argument('--fold', type=int, default=9)

    opt = args.parse_args()

    return opt


def ssd512_config(args):

    args.add_argument('--steps', type=str, default=(8, 16, 32, 64, 128, 256, 512))
    args.add_argument('--box_sizes', type=str, default=(35.84, 76.8, 153.6, 230.4, 307.2, 384.0, 460.8, 537.6))
    args.add_argument('--aspect_ratios', type=str, default=((2,), (2, 3), (2, 3), (2, 3), (2, 3), (2,), (2,)))
    args.add_argument('--fm_sizes', type=str, default=(64, 32, 16, 8, 4, 2, 1))
    args.add_argument('--img_size', type=int, default=512)

    args.add_argument('--lr', type=float, default=1e-3)
    args.add_argument('--momentum', type=float, default=0.9)
    args.add_argument('--weight_decay', type=float, default=1e-4)
    args.add_argument('--lr_steps', type=str, default=[60, 120, 180])
    args.add_argument('--lr_decay_rate', type=float, default=0.1)
    args.add_argument('--accum_grad', type=int, default=2)

    opt = args.parse_args()

    return opt