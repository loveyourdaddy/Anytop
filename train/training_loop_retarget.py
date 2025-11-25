"""
Training loop for retargeting model
Based on AnyTop's TrainLoop implementation
"""
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
import copy
import random

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
        
        # Load checkpoint if exists
        self._load_and_sync_parameters()
        
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
        
        if self.resume_step:
            self._load_optimizer_state()
        
        # Device
        self.device = torch.device("cpu")
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

        while self.total_step() < self.num_steps:
            print(f'Starting a new epoch at step {self.total_step()}')
            
            for batch_data in tqdm(self.data):
                if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                    break
                
                # Unpack retargeting batch
                # batch_data = (source_data, target_data, source_kwargs, target_kwargs, metadata)
                source_tuple, target_tuple, source_kwargs, target_kwargs, metadata = batch_data
                
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
                source_motion = source_motion.to(self.device) # ['motion']
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
                
                # Save checkpoint
                if (self.total_step() % self.save_interval == 0 and self.total_step() != 0) or \
                   self.total_step() == self.num_steps - 1:
                    self.save()
                    # self.model.eval()
                    # self.evaluate()
                    # self.generate_during_training()
                    # self.model.train()
                
                # Integration test
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
                
                self.step += 1
                
                if self.total_step() == self.num_steps:
                    break
            
            if not (not self.lr_anneal_steps or self.total_step() < self.lr_anneal_steps):
                break
    
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