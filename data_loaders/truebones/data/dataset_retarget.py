"""
Retargeting Dataset for training
"""
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data._utils.collate import default_collate
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from data_loaders.truebones.data.dataset import create_temporal_mask_for_window
from data_loaders.tensors import truebones_batch_collate
from model.conditioners import T5Conditioner
from data_loaders.truebones.truebones_utils.motion_process import remove_joints_augmentation, add_joint_augmentation
import random


class RetargetDataset(Dataset):
    """
    Dataset for retargeting training
    Loads pairs of motions from the same action but different skeletons
    """

    def __init__(
        self,
        args,
        data_root,
        num_frames,
        temporal_window,
        t5_name='t5-base',
        source_skeleton=None,
        target_skeletons=None,
        use_augmentation=False
    ):
        """
        Args:
            data_root: Root directory containing motion data
            num_frames: Number of frames per sequence
            temporal_window: Window size for temporal masking
            t5_name: T5 model name
            source_skeleton: Source skeleton type
            target_skeletons: List of target skeleton types
            use_augmentation: Whether to use joint augmentation
        """
        self.data_root = data_root
        self.num_frames = num_frames
        self.temporal_window = temporal_window
        self.source_skeleton = source_skeleton
        self.target_skeletons = target_skeletons if target_skeletons else []
        self.use_augmentation = use_augmentation

        # Load condition dictionary
        from data_loaders.truebones.truebones_utils.get_opt import get_opt
        self.opt = get_opt('cuda')
        cond_file = self.opt.cond_file
        cond_dict_full = np.load(cond_file, allow_pickle=True).item()

        # Filter to only include source_skeleton (and target_skeletons if specified)
        if self.source_skeleton:
            skeleton_types_to_keep = [self.source_skeleton]
            if self.target_skeletons:
                skeleton_types_to_keep.extend(self.target_skeletons)

            # Only keep specified skeletons
            self.cond_dict = {
                k: v for k, v in cond_dict_full.items()
                if k in skeleton_types_to_keep
            }
        else:
            # Use all skeletons
            self.cond_dict = cond_dict_full

        # Initialize T5 conditioner
        self.t5_conditioner = T5Conditioner(
            name=t5_name,
            finetune=False,
            word_dropout=0.0,
            normalize_text=False,
            device='cpu'  # CPU to avoid CUDA multiprocessing issues
        )

        # Build motion pairs
        self.motion_pairs = []
        self._build_motion_pairs()

        # joint name embeddings (cache)
        self.joints_names_embs_cache = {}
        for skeleton_type in self.cond_dict.keys():
            joints_names = self.cond_dict[skeleton_type]['joints_names']
            joints_names_padded = joints_names + [None] * (self.opt.max_joints - len(joints_names))
            self.joints_names_embs_cache[skeleton_type] = self._encode_joints_names(joints_names_padded)

        # TODO : 첫번째 페어만 사용 (일단 1개 모션만 학습)
        # for motion in self.motion_pairs:
        #     if motion['action_name'] == 'Dash':
        #         self.motion_pairs = [motion]
        #         break
        # self.motion_pairs = self.motion_pairs[:1]

    def _build_motion_pairs(self):
        """
        Build pairs of motions from different skeletons
        Groups motions by action name and creates cross-skeleton pairs
        """
        motion_dict = {}  # action_name -> [(skeleton_type, motion_path)]

        # Determine which skeletons to use
        if self.source_skeleton and self.target_skeletons:
            skeleton_types = [self.source_skeleton] + self.target_skeletons  # source + targets
        else:
            # Use all available skeletons
            skeleton_types = list(self.cond_dict.keys())

        # Load motion directory path from opt
        from data_loaders.truebones.truebones_utils.get_opt import get_opt
        opt = get_opt('cuda')
        motions_dir = opt.motion_dir
        if not os.path.exists(motions_dir):
            raise FileNotFoundError(f"Motion directory not found: {motions_dir}")

        # Scan all motion files
        all_files = [f for f in os.listdir(motions_dir) if f.endswith('.npy')]
        print(f"    Found {len(all_files)} motion files in {motions_dir}")

        # 모든 모션파일에서 source_type이 있는것을 찾기
        for motion_file in all_files:
            # Format: SkeletonType_SkeletonType_Action_ID.npy (e.g., "Horse_Horse_Idle_428.npy")
            parts = motion_file.replace('.npy', '').replace('.npz', '').split('_')
            if len(parts) < 3:
                continue

            skeleton_type = parts[0]
            action_name = parts[2] if len(parts) >= 3 else parts[-1]

            if skeleton_type not in skeleton_types:
                continue

            motion_path = os.path.join(motions_dir, motion_file)
            if action_name not in motion_dict:
                motion_dict[action_name] = []
            motion_dict[action_name].append((skeleton_type, motion_path))

        if len(motion_dict) == 0:
            print(f">> No motions found for {skeleton_types}")
            return

        # Create pairs from same actions across different skeletons
        for action_name, skeleton_motions in motion_dict.items():
            for i, (source_type, source_path) in enumerate(skeleton_motions):
                for target_type, target_path in skeleton_motions:
                    # If specific source/target skeletons are specified, filter
                    if self.source_skeleton and source_type != self.source_skeleton:
                        continue
                    if self.target_skeletons and target_type not in self.target_skeletons:
                        continue

                    self.motion_pairs.append({
                        'source_path': source_path,
                        'source_type': source_type,
                        'target_path': target_path,
                        'target_type': target_type,
                        'action_name': action_name
                    })
        print(f"    Built {len(self.motion_pairs)} motion pairs")

    def _load_motion(self, motion_path, skeleton_type):
        """Load and preprocess motion data"""
        # Load motion file
        if motion_path.endswith('.npz'):
            data = np.load(motion_path)
            motion = data['q']  # Quaternion representation
        else:
            motion = np.load(motion_path)

        # Get expected shape from condition dict
        n_joints = len(self.cond_dict[skeleton_type]['parents'])

        # Handle different motion formats
        if len(motion.shape) == 2:  # [frames, features]
            feature_len = motion.shape[1] // n_joints
            motion = motion.reshape(motion.shape[0], n_joints, feature_len)

        # Trim or pad to desired length
        if motion.shape[0] > self.num_frames:
            # Random crop
            start_idx = np.random.randint(0, motion.shape[0] - self.num_frames + 1)
            motion = motion[start_idx:start_idx + self.num_frames]
        elif motion.shape[0] < self.num_frames:
            # Pad with last frame
            pad_length = self.num_frames - motion.shape[0]
            last_frame = motion[-1:].repeat(pad_length, axis=0)
            motion = np.concatenate([motion, last_frame], axis=0)
        # print(">>> Loaded motion:", motion.shape)

        return motion

    def _apply_augmentation(self, motion, skeleton_type):
        """Apply augmentation (borrowed from original dataset)"""
        if not self.use_augmentation or skeleton_type == "Dragon":
            return motion

        aug_type = random.choice([0, 1, 2])

        if aug_type == 0:  # No augmentation
            return motion

        # Prepare data dict for augmentation
        data = {
            'motion': motion,
            'length': len(motion),
            'object_type': skeleton_type,
            'parents': self.cond_dict[skeleton_type]['parents'],
            'joints_graph_dist': self.cond_dict[skeleton_type]['joints_graph_dist'],
            'joints_relations': self.cond_dict[skeleton_type]['joint_relations'],
            'tpos_first_frame': self.cond_dict[skeleton_type]['tpos_first_frame'],
            'offsets': self.cond_dict[skeleton_type]['offsets'],
            'joints_names_embs': self._encode_joints_names(self.cond_dict[skeleton_type]['joints_names']).detach().cpu().numpy(),
            'kinematic_chains': self.cond_dict[skeleton_type].get('kinematic_chains', None)
        }

        mean = self.cond_dict[skeleton_type]['mean']
        std = self.cond_dict[skeleton_type]['std']

        if aug_type == 1:  # Remove joints
            removal_rate = random.choice([0.1, 0.2, 0.3])
            augmented_data = remove_joints_augmentation(data, removal_rate, mean, std)
            return augmented_data[0]  # Return motion only
        else:  # Add joint
            augmented_data = add_joint_augmentation(data, mean, std)
            return augmented_data[0]  # Return motion only

    def _create_batch_item(self, motion, skeleton_type, motion_type='target'):
        """
        Create batch item in the format expected by truebones_batch_collate

        Returns:
            List: [motion, m_length, parents, tpos_first_frame, offsets, 
                   temporal_mask, joints_graph_dist, joint_relations, 
                   object_type, joints_names_embs, idx, mean, std, max_joints]
        """
        cond_info = self.cond_dict[skeleton_type]
        n_joints = len(cond_info['parents'])
        feature_len = motion.shape[2]

        # ✅ Use cached embeddings instead of runtime encoding
        joints_names_embs = self.joints_names_embs_cache[skeleton_type]
        # print(f"n_joints {n_joints}")
        joints_names_embs = joints_names_embs[:n_joints]  # ← 추가!

        batch_item = [
            motion,  # motion [frames, joints, features]
            self.num_frames,  # m_length
            cond_info['parents'],  # parents
            cond_info['tpos_first_frame'],  # tpos_first_frame
            cond_info['offsets'],  # offsets
            create_temporal_mask_for_window(self.temporal_window, self.num_frames),  # temporal_mask
            cond_info['joints_graph_dist'],  # joints_graph_dist
            cond_info['joint_relations'],  # joint_relations
            skeleton_type,  # object_type
            joints_names_embs,  # joints_names_embs (cached)
            0,  # idx (placeholder)
            cond_info['mean'],  # mean
            cond_info['std'],  # std
            self.opt.max_joints  # max_joints
        ]
        # print(f">> {motion.shape}, {self.opt.max_joints}")

        return batch_item

    def _encode_joints_names(self, joints_names):
        """Encode joint names using T5"""
        names_tokens = self.t5_conditioner.tokenize(joints_names)
        embs = self.t5_conditioner(names_tokens)
        return embs.detach().cpu().numpy()

    def __len__(self):
        return len(self.motion_pairs)

    def __getitem__(self, idx):
        """
        Returns:
            Tuple: (source_batch_item, target_batch_item, pair_info)
        """
        pair = self.motion_pairs[idx]

        # Load source and target motions
        source_motion = self._load_motion(pair['source_path'], pair['source_type'])
        target_motion = self._load_motion(pair['target_path'], pair['target_type'])

        # Apply augmentation
        if self.use_augmentation:
            source_motion = self._apply_augmentation(source_motion, pair['source_type'])
            target_motion = self._apply_augmentation(target_motion, pair['target_type'])

        # Create batch items
        source_batch = self._create_batch_item(source_motion, pair['source_type'], 'source')
        target_batch = self._create_batch_item(target_motion, pair['target_type'], 'target')

        return {
            'source': source_batch,
            'target': target_batch,
            'source_type': pair['source_type'],
            'target_type': pair['target_type'],
            'action_name': pair['action_name']
        }


def collate_retarget_batch(batch):
    """
    Collate function for retargeting dataloader

    Args:
        batch: List of items from RetargetDataset
        Each item is a dict with keys: 'source', 'target', 'source_type', 'target_type', 'action_name'

    Returns:
        Tuple of 5 elements:
        - source_tuple: (motion_tensor, cond_dict) from truebones_batch_collate for source
        - target_tuple: (motion_tensor, cond_dict) from truebones_batch_collate for target
        - source_kwargs: Not used (kept for compatibility)
        - target_kwargs: Not used (kept for compatibility)
        - metadata: Dict with source_types, target_types, action_names
    """
    # Separate source and target batches
    source_batches = [item['source'] for item in batch]
    target_batches = [item['target'] for item in batch]

    # Collate using existing function
    # truebones_batch_collate returns: (motion_tensor, cond_dict)
    source_tuple = truebones_batch_collate(source_batches)
    target_tuple = truebones_batch_collate(target_batches)

    # Add metadata
    metadata = {
        'source_types': [item['source_type'] for item in batch],
        'target_types': [item['target_type'] for item in batch],
        'action_names': [item['action_name'] for item in batch]
    }

    # Return 5 elements for compatibility (some are None/unused)
    return source_tuple, target_tuple, None, None, metadata


class RetargetSampler(WeightedRandomSampler):
    """
    Balanced sampler for retargeting dataset
    Ensures equal sampling across different action types or skeleton pairs
    """

    def __init__(self, dataset):
        """
        Args:
            dataset: RetargetDataset instance
        """
        num_samples = len(dataset)

        # Count samples per action
        action_counts = {}
        for pair in dataset.motion_pairs:
            action = pair['action_name']
            if action not in action_counts:
                action_counts[action] = 0
            action_counts[action] += 1

        # Calculate weights - inverse of action frequency
        weights = np.zeros(num_samples)
        for i, pair in enumerate(dataset.motion_pairs):
            action = pair['action_name']
            # Weight = 1 / (number of samples with this action)
            weights[i] = 1.0 / action_counts[action]

        # Normalize weights
        weights = weights / weights.sum()

        print(f"RetargetSampler: {len(action_counts)} unique actions")
        print(f"Sample distribution: min={weights.min():.6f}, max={weights.max():.6f}")

        super().__init__(
            weights=weights,
            num_samples=num_samples,
            replacement=True
        )
