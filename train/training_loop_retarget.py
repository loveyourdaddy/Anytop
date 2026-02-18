"""
Training loop for retargeting model
Based on AnyTop's TrainLoop implementation
"""
import BVH
from InverseKinematics import animation_from_positions
from data_loaders.truebones.truebones_utils.motion_process import recover_from_bvh_ric_np
from data_loaders.truebones.truebones_utils.plot_script import plot_general_skeleton_3d_motion
from pathlib import Path
import numpy as np
import functools
import os
import re
from os.path import join as pjoin
from typing import Optional
import torch
from torch.optim import AdamW
from diffusion import logger
from utils import dist_util
from diffusion.fp16_util import MixedPrecisionTrainer
from diffusion.resample import LossAwareSampler, create_named_schedule_sampler
from tqdm import tqdm
from utils.model_util import load_model


INITIAL_LOG_LOSS_SCALE = 20.0


class RetargetTrainLoop:
    """
    Training loop for retargeting model with reconstruction loss
    """

    def __init__(self, args, train_platform, model, diffusion, data):
        self.args = args
        self.train_platform = train_platform
        self.model = model
        self.diffusion = diffusion
        self.cond_mode = model.cond_mode
        self.data = data
        self.batch_size = args.batch_size
        self.microbatch = args.batch_size  # deprecating this option
        self.lr = args.lr
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.resume_checkpoint = args.resume_checkpoint
        self.use_fp16 = False  # deprecating this option
        self.fp16_scale_growth = 1e-3  # deprecating this option
        self.weight_decay = args.weight_decay
        self.lr_anneal_steps = args.lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.global_batch = self.batch_size
        self.num_steps = args.num_steps
        self.num_epochs = self.num_steps // len(self.data) + 1
        self.sync_cuda = torch.cuda.is_available()
        self.save_dir = args.save_dir
        self.overwrite = args.overwrite

        self.save_source_motions = args.save_source_motions

        # Load checkpoint if exists
        # self._load_and_sync_parameters()

        # Mixed precision trainer
        self.mp_trainer = MixedPrecisionTrainer(
            model=self.model,
            use_fp16=self.use_fp16,
            fp16_scale_growth=self.fp16_scale_growth,
        )

        # Optimizer
        self.opt = AdamW(
            self.mp_trainer.master_params,
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        # Learning rate scheduler
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt,
            step_size=10000,
            gamma=0.99
        )

        # if self.resume_step:
        #     self._load_optimizer_state()

        # Device
        self.device = torch.device("cuda")
        if torch.cuda.is_available() and dist_util.dev() != 'cpu':
            self.device = torch.device(dist_util.dev())

        # Schedule sampler
        self.schedule_sampler_type = 'uniform'
        self.schedule_sampler = create_named_schedule_sampler(
            self.schedule_sampler_type,
            diffusion
        )

        # DDP (currently not used)
        self.use_ddp = False
        self.ddp_model = self.model

        print(f"RetargetTrainLoop initialized:")
        print(f"  - Total steps: {self.num_steps}")
        print(f"  - Batch size: {self.batch_size}")
        print(f"  - Learning rate: {self.lr}")
        print(f"  - Save dir: {self.save_dir}")
        print(f"  - Device: {self.device}")

    def _load_and_sync_parameters(self):
        """Load model checkpoint if exists"""
        self.resume_checkpoint = self.find_resume_checkpoint() or self.resume_checkpoint

        if self.resume_checkpoint:
            self.resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            logger.log(f"loading model from checkpoint: {self.resume_checkpoint}...")

            state_dict = dist_util.load_state_dict(
                self.resume_checkpoint,
                map_location=dist_util.dev()
            )

            load_model(self.model, state_dict)

    def _load_optimizer_state(self):
        """Load optimizer state from checkpoint"""
        opt_checkpoint = self.find_resume_opt_checkpoint()
        if opt_checkpoint and os.path.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint,
                map_location=dist_util.dev()
            )

            tgt_wd = self.opt.param_groups[0]['weight_decay']
            print('target weight decay:', tgt_wd)
            self.opt.load_state_dict(state_dict)
            print('loaded weight decay (will be replaced):',
                  self.opt.param_groups[0]['weight_decay'])

            # Preserve the weight decay parameter
            for group in self.opt.param_groups:
                group['weight_decay'] = tgt_wd

    def run_loop(self):
        """Main training loop"""
        print('train steps:', self.num_steps)
        # torch.multiprocessing.set_start_method('spawn')
        self.epoch = 0

        while self.total_step() < self.num_steps:
            print(f'Starting a new epoch {self.epoch} at step {self.total_step()}')

            for batch_data in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break

                # Unpack retargeting batch
                # batch_data = (source_data, target_data, source_kwargs, target_kwargs, metadata)
                source_tuple, target_tuple, source_kwargs, target_kwargs, metadata = batch_data
                # print("metadata:", metadata)

                # Unpack tuples from truebones_batch_collate
                source_motion, source_cond = source_tuple
                target_motion, target_cond = target_tuple

                # Move target motion to device (this is what we want to reconstruct)
                target_motion = target_motion.to(self.device)

                # Use target_cond as base conditioning
                cond = target_cond

                # Move conditioning to device
                if 'y' in cond:
                    for key, val in cond['y'].items():
                        if torch.is_tensor(val):
                            cond['y'][key] = val.to(self.device)

                # Add source motion as conditioning
                source_motion = source_motion.to(self.device)  # ['motion']
                cond['y']['source_motion'] = source_motion
                cond['y']['source_type'] = metadata['source_types']

                # Run training step
                self.run_step(target_motion, cond)

                # Logging
                if self.total_step() % self.log_interval == 0:
                    print(f"Source: {metadata['source_types']}")
                    print(f"Target: {metadata['target_types']}")
                    print(f"Action: {metadata['action_names']}")

                    for k, v in logger.get_current().dumpkvs().items():
                        if k == 'loss':
                            print('step[{}]: loss[{:0.5f}]'.format(self.total_step(), v))
                        if k in ['step', 'samples'] or '_q' in k:
                            continue
                        else:
                            self.train_platform.report_scalar(
                                name=k,
                                value=v,
                                iteration=self.total_step(),
                                group_name='Loss'
                            )

                # Render source motion
                if self.epoch == 0 and self.save_source_motions:
                    save_all_source_motions(data_loader=self.data, save_dir=self.save_dir, device=self.device, fps=30, max_motions=None)

                # Save checkpoint
                if (self.total_step() % self.save_interval == 0 and self.total_step() != 0) or self.total_step() == self.num_steps - 1:
                    self.save()

                    # Visualize and save bvh
                    save_training_visualization(
                        model=self.model,
                        diffusion=self.diffusion,
                        data_loader=self.data,
                        save_dir=self.save_dir,
                        step=self.total_step(),
                        device=self.device,
                        max_samples=None,
                        fps=30
                    )

                # Integration test
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return

                self.step += 1

                if self.total_step() == self.num_steps:
                    break

            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break
            self.epoch += 1

    def total_step(self):
        """Get total training steps including resumed steps"""
        total_step = self.step
        if self.resume_step:
            total_step += self.resume_step + 1
        return total_step

    def evaluate(self):
        """Evaluation during training (not implemented)"""
        if not self.args.eval_during_training:
            return
        print('Evaluation during training not implemented for retargeting')

    def generate_during_training(self):
        """Generate samples during training (not implemented)"""
        if not self.args.gen_during_training:
            return
        print('Generation during training not implemented for retargeting')
        # TODO: Implement retargeting sample generation

    def run_step(self, batch, cond, epoch=-1):
        """Single training step"""
        self.forward_backward(batch, cond, epoch)
        self.mp_trainer.optimize(self.opt, self.lr_scheduler)
        self._anneal_lr()
        self.log_step()

    def forward_backward(self, batch, cond, epoch):
        """Forward and backward pass with reconstruction loss"""
        self.mp_trainer.zero_grad()

        for i in range(0, batch.shape[0], self.microbatch):
            # Eliminates the microbatch feature
            assert i == 0
            assert self.microbatch == self.batch_size
            micro = batch
            micro_cond = cond
            last_batch = (i + self.microbatch) >= batch.shape[0]

            # Sample timesteps
            t, weights = self.schedule_sampler.sample(micro.shape[0], dist_util.dev())

            # Compute losses
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.ddp_model,
                micro,  # [bs, joints, features, frames] - target motion
                t,  # [bs](int) sampled timesteps
                model_kwargs=micro_cond  # conditioning including source motion
            )

            if last_batch or not self.use_ddp:
                losses = compute_losses()
            else:
                with self.ddp_model.no_sync():
                    losses = compute_losses()

            # Update loss-aware sampler if used
            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(
                    t, losses["loss"].detach()
                )

            # Weighted loss
            loss = (losses["loss"] * weights).mean()

            # Log losses
            log_loss_dict(
                self.diffusion, t, {k: v * weights for k, v in losses.items()}
            )

            # Backward
            self.mp_trainer.backward(loss)

    def _anneal_lr(self):
        """Anneal learning rate"""
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        """Log training statistics"""
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def ckpt_file_name(self):
        """Generate checkpoint filename"""
        return f"model{(self.step+self.resume_step):09d}.pt"

    def save(self):
        """Save model and optimizer checkpoints"""
        def save_checkpoint():
            def del_clip(state_dict):
                # Do not save CLIP weights
                clip_weights = [
                    e for e in state_dict.keys() if e.startswith('clip_model.')
                ]
                for e in clip_weights:
                    del state_dict[e]

            state_dict = self.mp_trainer.master_params_to_state_dict(
                self.mp_trainer.master_params
            )
            del_clip(state_dict)

            logger.log(f"saving model...")
            filename = self.ckpt_file_name()
            filepath = pjoin(self.save_dir, filename)

            with open(filepath, "wb") as f:
                torch.save(state_dict, f)

        save_checkpoint()

        # Save optimizer state
        opt_filepath = pjoin(self.save_dir, f"opt{(self.total_step()):09d}.pt")
        with open(opt_filepath, "wb") as f:
            torch.save(self.opt.state_dict(), f)
        print(f"Checkpoint saved: {self.ckpt_file_name()}") # and optimizer state

    def find_resume_checkpoint(self) -> Optional[str]:
        """Find the latest checkpoint in save directory"""
        if not os.path.exists(self.args.save_dir):
            return None

        matches = {
            file: re.match(r'model(\d+).pt$', file)
            for file in os.listdir(self.args.save_dir)
        }
        models = {
            int(match.group(1)): file
            for file, match in matches.items() if match
        }

        return pjoin(self.args.save_dir, models[max(models)]) if models else None

    def find_resume_opt_checkpoint(self) -> Optional[str]:
        """Find the latest optimizer checkpoint in save directory"""
        if not os.path.exists(self.args.save_dir):
            return None

        matches = {
            file: re.match(r'opt(\d+).pt$', file)
            for file in os.listdir(self.args.save_dir)
        }
        models = {
            int(match.group(1)): file
            for file, match in matches.items() if match
        }

        return pjoin(self.args.save_dir, models[max(models)]) if models else None

    def _save_sources_from_batch(self, batch_data):
        """Save sources from current batch"""
        from pathlib import Path
        import numpy as np

        # Unpack
        source_tuple, target_tuple, _, _, metadata = batch_data
        source_motion, source_cond = source_tuple

        # Load cond_dict
        from data_loaders.truebones.truebones_utils.get_opt import get_opt
        opt = get_opt(self.device)
        cond_dict_full = np.load(opt.cond_file, allow_pickle=True).item()

        vis_dir = Path(self.save_dir) / 'visualizations' / 'all_sources'
        vis_dir.mkdir(parents=True, exist_ok=True)

        # Move to device
        source_motion = source_motion.to(self.device)
        batch_size = source_motion.shape[0]

        # Save each source
        for i in range(batch_size):
            skeleton_type = metadata['target_types'][i]
            action_name = metadata['action_names'][i]
            source_name = f"{skeleton_type}_{action_name}"
            src_path = vis_dir / source_name

            # Skip if exists
            if src_path.with_suffix('.mp4').exists():
                continue

            # Get motion and skeleton info
            source = source_motion[i]
            n_joints = source_cond['y']['n_joints'][i].item()
            source = source[:n_joints]

            parents = source_cond['y']['parents'][i].tolist()
            mean_full = source_cond['y']['mean'][i].cpu().numpy()
            std_full = source_cond['y']['std'][i].cpu().numpy()
            mean = mean_full[:n_joints]
            std = std_full[:n_joints]

            offsets = cond_dict_full[skeleton_type]['offsets']
            joints_names = cond_dict_full[skeleton_type]['joints_names']

            # Save
            save_visualization(
                source,
                parents,
                offsets,
                mean,
                std,
                joints_names,
                skeleton_type,
                str(src_path),
                fps=30,
                title=f'Source - {skeleton_type} - {action_name}'
            )
            print(f"  ✅ Saved source: {source_name}")


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def log_loss_dict(diffusion, ts, losses):
    """Log loss dictionary with quartile statistics"""
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular)
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)


### Motion saving and visualization functions ### 

# save function
def save_visualization(
    motion,
    parents,
    offsets,
    mean,
    std,
    joints_names,
    object_type,
    save_path,
    fps=30,
    title='Motion'
):
    """
    Save motion in multiple formats: NPY, MP4, BVH
    Based on AnyTop's generation code

    Args:
        motion: [joints, features, frames] - normalized motion
        parents: List of parent indices
        offsets: [joints, 3] - bone offsets
        mean: [joints, features] - normalization mean
        std: [joints, features] - normalization std
        joints_names: List of joint names
        object_type: Skeleton type name
        save_path: Base path (without extension)
        fps: Frames per second
        title: Video title
    """
    # Denormalize motion
    # motion: [joints, features, frames]
    motion_denorm = motion.cpu().permute(2, 0, 1).numpy()  # [frames, joints, features]
    motion_denorm = motion_denorm * std[None, :] + mean[None, :]

    # Recover 3D positions from rotation representation
    global_positions = recover_from_bvh_ric_np(motion_denorm)

    # Save MP4
    # mp4_path = save_path + '.mp4'
    # plot_general_skeleton_3d_motion(
    #     mp4_path,
    #     parents,
    #     global_positions,
    #     title=title,
    #     fps=fps
    # )
    # print(f"  🎥 Saved MP4: {mp4_path}")

    # # Save NPY
    # npy_path = save_path + '.npy'
    # np.save(npy_path, motion_denorm)
    # print(f"  💾 Saved NPY: {npy_path}")

    # Create BVH animation using inverse kinematics
    out_anim, _, _ = animation_from_positions(
        positions=global_positions,
        parents=parents,
        offsets=offsets,
        iterations=150
    )

    # Save BVH
    if out_anim is not None:
        bvh_path = save_path + '.bvh'
        BVH.save(bvh_path, out_anim, joints_names)
        print(f"  📁 Saved BVH: {bvh_path}")

    # return {
    #     'npy': npy_path,
    #     'mp4': mp4_path,
    #     'bvh': bvh_path if out_anim is not None else None,
    #     'positions': global_positions
    # }


# Source motions 
def save_all_source_motions(
    data_loader,
    save_dir,
    device,
    fps=30,
    max_motions=None
):
    """
    Save all source motions from the dataset for visualization.

    Creates MP4 videos for every source motion in the dataset.
    Useful for inspection and debugging.

    Args:
        data_loader: DataLoader containing motion pairs
        save_dir: Base directory to save files
        device: torch device
        fps: Frames per second for videos
        max_motions: Maximum number of motions to save (None = all)

    Returns:
        Dict with statistics about saved motions
    """
    print(f"\n{'='*80}")
    print(f"SAVING ALL SOURCE MOTIONS")
    print(f"{'='*80}")

    # Create save directory
    vis_dir = Path(save_dir) / 'visualizations' / 'all_sources'
    vis_dir.mkdir(parents=True, exist_ok=True)

    # Load skeleton metadata
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(device)
    cond_dict_full = np.load(opt.cond_file, allow_pickle=True).item()

    # Statistics
    stats = {
        'total_processed': 0,
        'newly_saved': 0,
        'skipped_existing': 0,
        'errors': 0,
        'skeletons': {},
    }

    print(f"\n📁 Save directory: {vis_dir}")
    print(f"🔄 Iterating through dataset...")
    print()

    # Iterate through dataset
    motion_count = 0

    for batch_idx, batch_data in enumerate(tqdm(data_loader, desc="Processing batches")):
        # try:
        # Unpack batch
        source_tuple, target_tuple, _, _, metadata = batch_data
        source_motion, source_cond = source_tuple

        # Move to device
        source_motion = source_motion.to(device)
        batch_size = source_motion.shape[0]

        # Process each sample in batch
        for i in range(batch_size):
            # Check max_motions limit
            if max_motions is not None and motion_count >= max_motions:
                print(f"\n⏹️  Reached max_motions limit ({max_motions})")
                break

            # Get metadata
            skeleton_type = metadata['target_types'][i]
            action_name = metadata['action_names'][i]

            # Track skeleton
            if skeleton_type not in stats['skeletons']:
                stats['skeletons'][skeleton_type] = 0
            stats['skeletons'][skeleton_type] += 1

            # Create filename
            source_name = f"{skeleton_type}_{action_name}"
            src_path = vis_dir / source_name

            # Check if already exists
            if src_path.with_suffix('.mp4').exists():
                stats['skipped_existing'] += 1
                continue

            # Get motion
            source = source_motion[i]
            n_joints = source_cond['y']['n_joints'][i].item()
            source = source[:n_joints]  # TODO: Trim to actual joints

            # Get skeleton info
            parents = source_cond['y']['parents'][i]
            parents = parents[:n_joints]  # TODO: Trim: 맞는지 확인 필요

            mean_full = source_cond['y']['mean'][i].cpu().numpy()
            std_full = source_cond['y']['std'][i].cpu().numpy()
            mean = mean_full[:n_joints]
            std = std_full[:n_joints]

            offsets = cond_dict_full[skeleton_type]['offsets']
            joints_names = cond_dict_full[skeleton_type]['joints_names']

            # Save motion
            save_visualization(
                source.cpu(),
                parents,
                offsets,
                mean,
                std,
                joints_names,
                skeleton_type,
                str(src_path),
                fps=fps,
                title=f'Source - {skeleton_type} - {action_name}'
            )
            stats['newly_saved'] += 1
            print(f"  ✅ Saved: {source_name}")

            stats['total_processed'] += 1
            motion_count += 1

        # Check max_motions limit
        if max_motions is not None and motion_count >= max_motions:
            break

    # Print summary
    print(f"\n{'='*80}")
    print(f"SUMMARY")
    print(f"{'='*80}")
    print(f"  📊 Total processed: {stats['total_processed']}")
    print(f"  ✅ Newly saved: {stats['newly_saved']}")
    print(f"  ⏭️  Skipped (existing): {stats['skipped_existing']}")
    print(f"  ❌ Errors: {stats['errors']}")
    print(f"\n  📁 Saved to: {vis_dir}")

    if stats['skeletons']:
        print(f"\n  Motions per skeleton:")
        for skel in sorted(stats['skeletons'].keys()):
            count = stats['skeletons'][skel]
            print(f"    {skel}: {count} motions")

    print(f"{'='*80}\n")

    return stats

# save generated 
def save_training_visualization(
    model,
    diffusion,
    data_loader,
    save_dir,
    step,
    device,
    max_samples=None,
    fps=30
):
    """
    Generate and save motion visualization during training
    Saves: Generated, Ground Truth, Source motions in NPY + MP4 + BVH

    Args:
        model: Diffusion model
        diffusion: Diffusion process
        batch_data: Training batch (source_tuple, target_tuple, _, _, metadata) -> data_loader
        save_dir: Directory to save files
        step: Current training step
        device: torch device
        num_samples: Number of samples to generate (for visualization)
        fps: Frames per second for video
    """
    print(f"\n{'='*80}")
    print(f"VISUALIZATIONS AT STEP {step}")

    model.eval()

    
    # Create save directory
    vis_dir = Path(save_dir) / 'visualizations' / f'step{step:09d}'
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Load skeleton metadata
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    opt = get_opt(device)
    cond_dict_full = np.load(opt.cond_file, allow_pickle=True).item()
    
    # Statistics
    stats = {
        'total_generated': 0,
        'skipped_existing': 0,
        'skeletons': {}
    }
    
    sample_count = 0
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(data_loader, desc=f"Step {step}")):
            # Unpack batch
            source_tuple, target_tuple, _, _, metadata = batch_data
            source_motion, source_cond = source_tuple
            target_motion, target_cond = target_tuple

            # Prepare conditioning for generation
            cond = {}
            cond['y'] = {}
            for k, v in target_cond['y'].items():
                if torch.is_tensor(v):
                    cond['y'][k] = v.to(device)
                else:
                    cond['y'][k] = v

            # Add source motion as conditioning
            cond['y']['source_motion'] = source_motion.to(device)
            cond['y']['source_type'] = metadata['source_types']

            # Get shape info
            batch_size = target_motion.shape[0]
            bs, max_joints, n_feats, n_frames = target_motion.shape
            
            # Sample from model using p_sample_loop
            print(f"\n🎲 Sampling from diffusion model...")
            sample = diffusion.p_sample_loop(
                model,
                (batch_size, max_joints, n_feats, n_frames),
                clip_denoised=False,
                model_kwargs=cond,
                skip_timesteps=0,
                init_image=None,
                progress=True,
                dump_steps=None,
                noise=None,
                const_noise=False,
            )

            # ✅ Save ALL samples in this batch
            for i in range(batch_size):
                # Check max_samples limit
                if max_samples is not None and sample_count >= max_samples:
                    print(f"\n⏹️  Reached max_samples limit ({max_samples})")
                    break
                
                # Get metadata
                skeleton_type = metadata['target_types'][i]
                action_name = metadata['action_names'][i]
                n_joints = cond['y']['n_joints'][i].item()
                
                # Track skeleton
                if skeleton_type not in stats['skeletons']:
                    stats['skeletons'][skeleton_type] = 0
                stats['skeletons'][skeleton_type] += 1
                
                # Create filename
                filename = f"sample{sample_count:04d}_{skeleton_type}_{action_name}"
                save_path = vis_dir / filename
                
                # Check if exists
                if save_path.with_suffix('.mp4').exists():
                    stats['skipped_existing'] += 1
                    sample_count += 1
                    continue
                
                # Get motions
                generated_motion = sample[i][:n_joints]
                # ground_truth = target_motion[i][:n_joints]
                # source = source_motion[i][:n_joints]
                
                # Get skeleton info
                parents = cond['y']['parents'][i][:n_joints]
                mean = cond['y']['mean'][i].cpu().numpy()[:n_joints]
                std = cond['y']['std'][i].cpu().numpy()[:n_joints]
                
                offsets = cond_dict_full[skeleton_type]['offsets']
                joints_names = cond_dict_full[skeleton_type]['joints_names']
                
                # Save generated motion
                save_visualization(
                    generated_motion.cpu(),
                    parents,
                    offsets,
                    mean,
                    std,
                    joints_names,
                    skeleton_type,
                    str(save_path) + '_generated',
                    fps=fps,
                    title=f'Generated - {skeleton_type} - {action_name} - Step {step}'
                )
                
                stats['total_generated'] += 1
                sample_count += 1
            
            # Check max_samples limit
            if max_samples is not None and sample_count >= max_samples:
                break
        
    model.train()