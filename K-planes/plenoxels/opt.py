import argparse


def config_parser(cmd=None):

    p = argparse.ArgumentParser(description="")

    p.add_argument('--render-only', action='store_true')
    p.add_argument('--validate-only', action='store_true')
    p.add_argument('--spacetime-only', action='store_true')
    p.add_argument('--compress-only', action='store_true')
    p.add_argument('--decompress-only', action='store_true')
    p.add_argument('--config-path', type=str, required=True)
    p.add_argument('--log-dir', type=str, default=None)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--use-mask', type=int, default=1)
    # p.add_argument('--mask_weight', type=float, default=5e-14) # oringal
    p.add_argument('--mask_weight', type=float, default=5e-12) 
    # p.add_argument('--mask_weight', type=float, default=5e-13)
    # p.add_argument('--mask-weight', type=float, default=5e-17)
    p.add_argument('--reconstruct-mask', type=bool, default=True)
    p.add_argument('override', nargs=argparse.REMAINDER)


     # loader options
    p.add_argument("--s3im_weight", type=float, default=0.0)
    p.add_argument("--s3im_kernel", type=int, default=4)
    p.add_argument("--s3im_stride", type=int, default=4)
    p.add_argument("--s3im_repeat_time", type=int, default=10)
    p.add_argument("--s3im_patch_height", type=int, default=64)
    p.add_argument("--s3im_patch_width", type=int, default=64)

    if cmd is not None:
        return p.parse_args(cmd)
    else:
        return p.parse_args()
