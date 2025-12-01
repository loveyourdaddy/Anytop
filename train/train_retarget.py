# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model for motion retargeting with reconstruction loss.

python -m train.train_retarget --model_prefix retarget --overwrite
    --source_skeleton Alligator --target_skeletons Alligator

python -m train.train_retarget \
    --model_prefix retarget_alligator \
    --batch_size 16 \
    --overwrite

python -m train.train_retarget --model_prefix retarget_brownbear --overwrite --ml_platform_type TensorboardPlatform --batch_size 1 --device 4
"""
import sys
import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop_retarget import RetargetTrainLoop
from data_loaders.get_data_retarget import get_retarget_dataset_loader
from utils.model_util import create_model_and_diffusion_general_skeleton
from utils.ml_platforms import ClearmlPlatform, TensorboardPlatform, NoPlatform, WandBPlatform


def main():
    args = train_args()
    fixseed(args.seed)

    args.source_skeleton = "BrownBear" # Bear
    args.target_skeletons = [args.source_skeleton] #"bear" 
    args.save_interval = 1000  # 10000
    # args.batch_size = 1 # TODO

    # Setup save directory
    save_dir = args.save_dir
    if save_dir is None:
        prefix = "Retarget"
        if args.model_prefix is not None:
            prefix = args.model_prefix
        model_name = f'{prefix}_dataset_truebones_bs_{args.batch_size}_latentdim_{args.latent_dim}'
        mod_list = [m for m in os.listdir(os.path.join(os.getcwd(), 'save')) if m.startswith(model_name)]
        if len(mod_list) > 0 and not args.overwrite:
            model_name = f'{model_name}_{len(mod_list)}'
        save_dir = os.path.join(os.getcwd(), 'save', model_name)
        args.save_dir = save_dir

    # Setup ML platform for logging
    ml_platform_type = eval(args.ml_platform_type)
    ml_platform = ml_platform_type(save_dir=args.save_dir)
    ml_platform.report_args(args, name='Args')

    if save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(save_dir))
    elif not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Save arguments
    args_path = os.path.join(save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)

    dist_util.setup_dist(args.device)

    print("creating retargeting data loader...")
    data = get_retarget_dataset_loader(
        args,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        temporal_window=args.temporal_window,
        t5_name=args.t5_name,
        source_skeleton=args.source_skeleton,
        target_skeletons=args.target_skeletons,
    )

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    model.to(dist_util.dev())
    ml_platform.watch_model(model)

    print("Training retargeting model...")
    RetargetTrainLoop(args, ml_platform, model, diffusion, data).run_loop()
    ml_platform.close()


if __name__ == "__main__":
    main()
