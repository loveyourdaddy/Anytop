'''
python -m sample.generate --model_path save/flying_model_dataset_truebones_bs_16_latentdim_128/model000229999.pt --object_type Parrot2 Bat --num_repetitions 3
python -m sample.retarget --model_path save/quadropeds_model_dataset_truebones_bs_16_latentdim_128/model000189999.pt\
    --source_motion dataset/truebones/zoo/truebones_processed/motions/Alligator_Alligator_BigMouth_11.npy\
    --source_type Alligator --num_repetitions 3

1. anytop 데이터 (dataset/truebones/zoo/truebones_processed/motions)
    ./dataset/truebones/zoo/truebones_processed/motions/Horse_Horse_Idle_428.npy
2. 우리 데이터    (dataset/truebones/zoo/truebones_processed/processed)
    dataset/truebones/zoo/truebones_processed/processed/Alligator/Alligator_BigMouth.npz
'''

# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion_general_skeleton, load_model
from utils import dist_util
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from data_loaders.tensors import truebones_batch_collate
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np, recover_from_bvh_rot_np
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from os.path import join as pjoin
from model.conditioners import T5Conditioner
import BVH
from InverseKinematics import animation_from_positions
from data_loaders.truebones.truebones_utils.get_opt import get_opt

def load_source_motion(source_path, source_type, cond_dict, n_frames=None):
    """
    Load source motion and preprocess it
    """
    # Load motion data
    motion_file = np.load(source_path)
    motion = np.load(source_path)  # [frames, joints, features]
    
    # Denormalize source motion
    mean = cond_dict[source_type]['mean'][None, :]
    std = cond_dict[source_type]['std'][None, :]
    motion = motion * std + mean
    # (Pdb) motion['q'].shape (136, 31, 6)
    # (Pdb) mean.shape (1, 25, 13)
    
    # Get source skeleton info
    source_parents = cond_dict[source_type]['parents']
    source_offsets = cond_dict[source_type]['offsets']
    
    # Optionally trim or pad to desired length
    if n_frames is not None:
        if motion.shape[0] > n_frames:
            motion = motion[:n_frames]
        elif motion.shape[0] < n_frames:
            # Pad with last frame
            pad_length = n_frames - motion.shape[0]
            motion = np.concatenate([motion, motion[-1:].repeat(pad_length, axis=0)], axis=0)
    
    # Extract positions from source motion
    source_positions = recover_from_bvh_ric_np(motion)
    
    return motion, source_positions, source_parents, source_offsets

def create_retargeting_condition(source_motion, source_positions, source_type, 
                                 target_types, cond_dict, t5_conditioner, 
                                 max_joints, feature_len, temporal_window):
    """
    Create conditioning for retargeting
    """
    bs = len(target_types)
    n_frames = source_motion.shape[0]
    
    # Create basic condition structure
    _, model_kwargs = create_condition(
        target_types, cond_dict, n_frames, temporal_window, 
        t5_conditioner=t5_conditioner, max_joints=max_joints, 
        feature_len=feature_len
    )
    
    # Add source motion as conditioning signal
    # Option 1: Use source positions as reference
    model_kwargs['y']['source_positions'] = torch.from_numpy(source_positions).float()
    model_kwargs['y']['source_type'] = source_type
    
    # Option 2: Use inpainting to guide certain joints/frames
    # Create inpainting mask for key frames or root motion
    inpainting_mask = torch.zeros(bs, max_joints, feature_len, n_frames, dtype=torch.bool)
    inpainted_motion = torch.zeros(bs, max_joints, feature_len, n_frames)
    
    # Example: Keep root motion from source
    for i, target_type in enumerate(target_types):
        target_mean = cond_dict[target_type]['mean'][None, :]
        target_std = cond_dict[target_type]['std'][None, :]
        
        # Normalize source motion for target
        # You may need to map source root to target root here
        normalized_motion = (source_motion - target_mean) / target_std
        inpainted_motion[i, 0, :, :] = torch.from_numpy(normalized_motion[:, 0, :].T)
        inpainting_mask[i, 0, :, :] = True  # Keep root motion
    
    model_kwargs['y']['inpainting_mask'] = inpainting_mask
    model_kwargs['y']['inpainted_motion'] = inpainted_motion
    
    return model_kwargs

def save_retargeted_motions(sample, model_kwargs, cond_dict, out_path, 
                           rep_i, fps, source_motion, source_positions, source_type):
    """
    Save retargeted motions and create comparison visualization
    """
    # save source motion, bvh, source video
    source_npy_name = f'source_{source_type}.npy'
    np.save(pjoin(out_path, source_npy_name), source_motion)
    
    source_mp4_name = f'source_{source_type}.mp4'
    plot_general_skeleton_3d_motion(
        pjoin(out_path, source_mp4_name), 
        model_kwargs['y']['parents'][0], source_positions,
        title=f'Source: {source_type}', fps=fps
    )
    
    source_bvh_name = f'source_{source_type}.bvh'
    out_anim, _1, _2 = animation_from_positions(
        positions=source_positions, 
        parents=model_kwargs['y']['parents'][0], 
        offsets=cond_dict[source_type]['offsets'], 
        iterations=150
    )
    if out_anim is not None:
        # save 
        BVH.save(pjoin(out_path, source_bvh_name), out_anim,
                cond_dict[source_type]['joints_names'])
    
    # Save retargeted motions
    for i, motion in enumerate(sample):
        n_joints = model_kwargs['y']["n_joints"][i].item()
        motion = motion[:n_joints]
        target_type = model_kwargs['y']["object_type"][i]
        parents = model_kwargs['y']["parents"][i]
        
        # Denormalize
        mean = cond_dict[target_type]['mean'][None, :]
        std = cond_dict[target_type]['std'][None, :]
        motion = motion.cpu().permute(2, 0, 1).numpy() * std + mean
        
        # Recover positions
        offsets = cond_dict[target_type]['offsets']
        global_positions = recover_from_bvh_ric_np(motion)
        out_anim, _1, _2 = animation_from_positions(
            positions=global_positions, parents=parents, 
            offsets=offsets, iterations=150
        )
        
        # Save with descriptive names
        name_pref = f'{source_type}_to_{target_type}_rep_{rep_i}'
        existing_files = [f for f in os.listdir(out_path) 
                         if f.startswith(name_pref) and f.endswith('.npy')]
        
        npy_name = f'{name_pref}_#{len(existing_files)}.npy'
        mp4_name = f'{name_pref}_#{len(existing_files)}.mp4'
        bvh_name = f'{name_pref}_#{len(existing_files)}.bvh'
        comparison_name = f'{name_pref}_comparison_#{len(existing_files)}.mp4'
        
        # Save motion
        np.save(pjoin(out_path, npy_name), motion)
        
        # Save visualization
        plot_general_skeleton_3d_motion(
            pjoin(out_path, mp4_name), parents, global_positions,
            title=f'{source_type} → {target_type}', fps=fps
        )
        
        # Save BVH
        if out_anim is not None:
            BVH.save(pjoin(out_path, bvh_name), out_anim, 
                    cond_dict[target_type]['joints_names'])
        
        # Create side-by-side comparison
        # plot_comparison(
        #     pjoin(out_path, comparison_name),
        #     source_positions, cond_dict[source_type]['parents'],
        #     global_positions, parents,
        #     source_type, target_type, fps
        # )
        
        print(f"Retargeted motion created: {npy_name}")
        
def main(args=None, cond_dict=None):
    if args is None:
        args = generate_args()
    
    # Add new arguments for retargeting
    if not hasattr(args, 'source_motion'):
        raise ValueError("source_motion path is required for retargeting")
    if not hasattr(args, 'source_type'):
        raise ValueError("source_type is required for retargeting")
    
    fixseed(args.seed)
    opt = get_opt(args.device)
    
    if cond_dict is None:
        if args.cond_path:
            cond_dict = np.load(args.cond_path, allow_pickle=True).item()
        else:
            cond_dict = np.load(opt.cond_file, allow_pickle=True).item()
    
    # Load source motion
    print(f"Loading source motion from {args.source_motion}")
    source_motion, source_positions, source_parents, source_offsets = load_source_motion(
        args.source_motion, args.source_type, cond_dict, n_frames=None
    )
    
    n_frames = source_motion.shape[0]
    fps = opt.fps
    max_joints = opt.max_joints
    
    # Set target 
    args.object_type = [args.source_type]
    
    # Setup
    dist_util.setup_dist(args.device)
    target_types = args.object_type  # Target skeleton types
    
    out_path = args.output_dir
    if out_path == '':
        name = os.path.basename(os.path.dirname(args.model_path))
        niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
        out_path = os.path.join(
            os.path.dirname(args.model_path),
            f'retarget_{args.source_type}_to_{target_types[0]}_{name}_{niter}_seed{args.seed}'
        )
    os.makedirs(out_path, exist_ok=True)
    
    args.batch_size = len(target_types)
    
    print("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion_general_skeleton(args)
    
    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model(model, state_dict)
    
    print("Loading T5 model")
    t5_conditioner = T5Conditioner(
        name=args.t5_name, finetune=False, 
        word_dropout=0.0, normalize_text=False, device='cuda'
    )
    
    model.to(dist_util.dev())
    model.eval()
    
    # Create retargeting condition
    model_kwargs = create_retargeting_condition(
        source_motion, source_positions, args.source_type,
        target_types, cond_dict, t5_conditioner,
        max_joints=opt.max_joints, feature_len=opt.feature_len,
        temporal_window=args.temporal_window
    )
    
    for rep_i in range(args.num_repetitions):
        print(f'### Retargeting [repetition #{rep_i}]')
        sample_fn = diffusion.p_sample_loop
        
        sample = sample_fn(
            model,
            (args.batch_size, max_joints, model.feature_len, n_frames),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            skip_timesteps=0,
            init_image=None,
            progress=True,
            dump_steps=None,
            noise=None,
            const_noise=False,
        )
        
        # Save results
        save_retargeted_motions(
            sample, model_kwargs, cond_dict, out_path, 
            rep_i, fps, source_motion, source_positions, args.source_type
        )
        
def encode_joints_names(joints_names, t5_conditioner): # joints names should be padded with None to be of max_len 
        names_tokens = t5_conditioner.tokenize(joints_names)
        embs = t5_conditioner(names_tokens)
        return embs
    
def create_condition(object_types, cond_dict, n_frames, temporal_window, t5_conditioner, max_joints, feature_len):
    batches = list()
    for object_type in object_types:
        batch=list()
         # motion, m_length, parents, joints_perm, inv_joints_perm, tpos_first_frame, offsets, self.temporal_mask_template, joints_graph_dist, joints_relations, object_type, joints_names
        parents = cond_dict[object_type]['parents']
        n_joints = len(parents)
        mean = cond_dict[object_type]['mean']
        std = cond_dict[object_type]['std']
        tpos_first_frame = cond_dict[object_type]['tpos_first_frame']
        tpos_first_frame =  (tpos_first_frame - mean) / (std + 1e-6)
        tpos_first_frame = np.nan_to_num(tpos_first_frame)
        joint_relations = cond_dict[object_type]['joint_relations']
        joints_graph_dist = cond_dict[object_type]['joints_graph_dist']
        offsets = cond_dict[object_type]['offsets']
        joints_names_embs = encode_joints_names(cond_dict[object_type]['joints_names'] , t5_conditioner).detach().cpu().numpy()
        batch.append(np.zeros((n_frames, n_joints, feature_len)))
        batch.append(n_frames)
        batch.append(parents)
        batch.append(tpos_first_frame)
        batch.append(offsets)
        batch.append(create_temporal_mask_for_window(temporal_window, n_frames))
        batch.append(joints_graph_dist)
        batch.append(joint_relations)
        batch.append(object_type)
        batch.append(joints_names_embs)
        batch.append(0)
        batch.append(mean)
        batch.append(std)
        batch.append(max_joints)
        batches.append(batch)
        
    return truebones_batch_collate(batches)


if __name__ == "__main__":
    main()
