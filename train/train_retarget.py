# This code is based on https://github.com/openai/guided-diffusion
"""
Usage:
Single character
python -m train.train_retarget \
    --overwrite \
    --source_skeleton BrownBear \
    --source_group quadropeds \
    --batch_size 1 \
    --lambda_geo 1.0 \
    --use_self_reconstruction\
    --use_cycle_loss

모든 quadropeds을 source로:
python -m train.train_retarget \
    --overwrite \
    --source_group quadropeds \
    --batch_size 1 \
    --use_self_reconstruction\
    --use_cycle_loss

Resume (resume_chekcpoint, save_dir must be specified):
python -m train.train_retarget \
    --source_group quadropeds \
    --batch_size 1 \
    --resume_checkpoint save/20260325_Retarget_1_latentdim_128_src_quadropeds_selfRecon_cycle/visualization/step000400000.pt \
    --save_dir save/20260325_Retarget_1_latentdim_128_src_quadropeds_selfRecon_cycle \
    --overwrite \
    --use_self_reconstruction \
    --use_cycle_loss
    
Options
    source_group: 리타게팅하려는 그룹. 무조건 명시되어야함
    use_self_reconstruction: source motion에 대해 reconstruction loss
    use_cross_reconstruction: source motion과 target motion의 이름이 동일할 때, target motion에 대해 reconstruction loss
    use_cycle_loss: source motion -> target motion -> source motion으로 cycle loss
    --lambda_root 2.0 \
    --lambda_self_recon 2.0 \
    --lambda_geo 1.0 \
--latent_dim 64
--source_skeleton Horse
--lambda_cycle 0.1
--resume_checkpoint save/20260218_Retarget_dataset_truebones_bs_4_latentdim_128/model000290000.pt 
--save_dir save/20260218_Retarget_dataset_truebones_bs_4_latentdim_128

Visualization
    output: 
        /home/inseo/Github/BVHView/render_bvhs.sh /home/inseo/Github/Anytop/save/20260325_Retarget_1_latentdim_128_src_quadropeds_selfRecon_cycle/visualizations/step000400001
        /home/inseo/Github/Anytop/save/20260325_Retarget_1_latentdim_128_src_BrownBear_selfRecon_cycle/visualizations/step000500001
    source: 
        /home/inseo/Github/BVHView/render_bvhs.sh /home/inseo/Github/Anytop/dataset/truebones/zoo/truebones_processed/bvhs/BrownBear
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
import time


def main():
    args = train_args()
    fixseed(args.seed)

    # Setup save directory
    save_dir = args.save_dir
    if save_dir is None:
        # naming convention
        prefix = "Retarget"
        if args.model_prefix is not None:
            # prefix = args.model_prefix
            prefix = "retarget_" + args.source_group
        model_name = f'{prefix}_{args.batch_size}_latentdim_{args.latent_dim}_src'
        
        # source skel
        if args.source_skeleton is not None: # args.source_skeleton: list 
            source_group = getattr(args, 'source_skeleton', None)
        elif args.source_group is not None:
            source_group = [getattr(args, 'source_group', None)]
        else: 
            raise ValueError("Must specify either source_group or source_skeletons")
        for animal in source_group:
            model_name += f'_{animal}'
            
        # options
        if args.use_self_reconstruction:
            model_name += '_selfRecon'
        if args.use_cross_reconstruction:
            model_name += '_crossRecon'
        if args.use_cycle_loss:
            model_name += '_cycle'
            
        # 중복체크
        mod_list = [m for m in os.listdir(os.path.join(os.getcwd(), 'save')) if m.startswith(model_name)]
        if len(mod_list) > 0 and not args.overwrite:
            model_name = f'{model_name}_{len(mod_list)}'
        # date
        curr_time = time.strftime("%Y%m%d", time.localtime())
        model_name = curr_time + '_' + model_name
        # path
        save_dir = os.path.join(os.getcwd(), 'save', model_name)
        args.save_dir = save_dir
    print("Save_dir to:", save_dir)

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

    # source_group이 none이거나, source_skeleton이 none이어야함
    source_group = getattr(args, 'source_group', None)
    source_skeletons = getattr(args, 'source_skeleton', None)

    # group
    data = get_retarget_dataset_loader(
        args,
        batch_size=args.batch_size,
        num_frames=args.num_frames,
        temporal_window=args.temporal_window,
        t5_name=args.t5_name,
        source_group=source_group,
        source_skeletons=source_skeletons,
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
