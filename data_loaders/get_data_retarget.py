"""
Data loader for retargeting training
Following the AnyTop get_data.py pattern
"""
import os
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from torch.utils.data import DataLoader
from data_loaders.truebones.data.dataset_retarget import RetargetDataset, collate_retarget_batch

from torch.utils.data import ConcatDataset


def _parse_paired_data_file(filepath):
    """Parse tab-separated paired data file.

    Format per line: SkelA/SkelA_Action.npz<TAB>SkelB/SkelB_Action.npz

    Returns:
        set of (src_skel, src_action, tgt_skel, tgt_action)
    """
    pairs = set()
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            src_path = parts[0]  # e.g. "BrownBear/BrownBear_Attack2.npz"
            tgt_path = parts[1]  # e.g. "Buffalo/Buffalo_Attack.npz"

            src_skel = src_path.split('/')[0]
            src_stem = os.path.splitext(src_path.split('/')[-1])[0]  # "BrownBear_Attack2"
            src_action = src_stem[len(src_skel) + 1:]               # "Attack2"

            tgt_skel = tgt_path.split('/')[0]
            tgt_stem = os.path.splitext(tgt_path.split('/')[-1])[0]
            tgt_action = tgt_stem[len(tgt_skel) + 1:]

            pairs.add((src_skel, src_action, tgt_skel, tgt_action))
    return pairs


def get_dataset_class(name):
    """Get dataset class by name"""
    return RetargetDataset


def get_retarget_dataset(
    args,
    num_frames,
    split='train',
    temporal_window=31,
    t5_name='t5-base',
    source_skeleton=None,
    target_skeletons=None,
    data_root='dataset/truebones/zoo/truebones_processed',
    use_augmentation=False
):
    """
    Get retargeting dataset

    Args:
        num_frames: Number of frames per sequence
        split: Train/val/test split (currently unused for retargeting)
        temporal_window: Temporal window size for masking
        t5_name: T5 model name for text conditioning
        source_skeleton: Source skeleton type (e.g., 'Alligator')
        target_skeletons: List of target skeleton types (e.g., ['Horse', 'Parrot2'])
        data_root: Root directory of motion data
        use_augmentation: Whether to use joint augmentation

    Returns:
        RetargetDataset instance
    """
    dataset = RetargetDataset(
        args=args,
        data_root=data_root,
        num_frames=num_frames,
        temporal_window=temporal_window,
        t5_name=t5_name,
        source_skeleton=source_skeleton,
        target_skeletons=target_skeletons,
        use_augmentation=use_augmentation
    )
    return dataset

def get_retarget_dataset_loader(
    args,
    batch_size,
    num_frames,
    temporal_window,
    t5_name,
    source_group=None,          # "quadropeds", "bipeds", etc.
    source_skeletons=None,      # Or custom list
    split='train',
):
    """
    Get retargeting/reconstruction dataset loader.

    Args:
        source_group: Group name from OBJECT_SUBSETS_DICT
                     ('quadropeds', 'bipeds', 'flying', 'all', etc.)
        source_skeletons: Or specific list of skeleton names
        reconstruction_mode: If True, target = source
    """

    print(f"\n{'='*80}")
    print(f"RECONSTRUCTION DATASET CONFIGURATION")
    print(f"{'='*80}")

    # Get skeletons from group or custom list
    if source_group is not None:
        if source_group not in OBJECT_SUBSETS_DICT:
            raise ValueError(f"Unknown group: {source_group}. "
                             f"Available: {list(OBJECT_SUBSETS_DICT.keys())}")
        skeletons = OBJECT_SUBSETS_DICT[source_group]
        print(f"📁 Group: {source_group}")
        print(f"   Skeletons ({len(skeletons)}): {skeletons[:5]}{'...' if len(skeletons) > 5 else ''}")
    elif source_skeletons is not None:
        skeletons = source_skeletons
        print(f"📁 Custom skeleton list ({len(skeletons)}): {skeletons}")
    else:
        raise ValueError("Must specify either source_group or source_skeletons")

    print(f"📦 Batch size: {batch_size}")
    print(f"🎬 Num frames: {num_frames}")
    print(f"{'='*80}\n")

    # args.source_skeleton: 지정된 경우 해당 skeleton만 source로 사용, None이면 group 전체 사용
    selected_source_skeletons = getattr(args, 'source_skeleton', None)  # list or None
    data_dir = getattr(args, 'data_dir', './dataset/truebones/zoo/truebones_processed')

    # Parse paired data file if specified
    paired_data_file = getattr(args, 'paired_data_file', None)
    paired_pairs = None
    if paired_data_file is not None:
        paired_pairs = _parse_paired_data_file(paired_data_file)
        print(f"Paired mode: loaded {len(paired_pairs)} pairs from {paired_data_file}")

    # Build shared T5 conditioner and joint-name embedding cache once for all datasets
    from model.conditioners import T5Conditioner
    from data_loaders.truebones.truebones_utils.get_opt import get_opt
    import numpy as np
    shared_t5 = T5Conditioner(name=t5_name, finetune=False, word_dropout=0.0, normalize_text=False, device='cpu')
    _opt = get_opt('cuda')
    _cond_dict_full = np.load(_opt.cond_file, allow_pickle=True).item() # all animals
    shared_embs_cache = {}
    _names_to_emb = {}
    print("Building shared T5 conditioner and joint-name embedding cache...")
    for _skel, _cond in _cond_dict_full.items():
        # print(f"_skel: {_skel}")
        _joints_names_padded = _cond['joints_names'] + [None] * (_opt.max_joints - len(_cond['joints_names'])) # padded with None to max_joints
        _key = tuple(_joints_names_padded)
        if _key not in _names_to_emb:
            _names_tokens = shared_t5.tokenize(_joints_names_padded)
            _names_to_emb[_key] = shared_t5(_names_tokens).detach().cpu().numpy()
        shared_embs_cache[_skel] = _names_to_emb[_key]

    datasets = []
    for source_skel in skeletons:
        # selected_source_skeletons: source skeleton이 옵션으로 지정된 것
        # source skeleton이 옵션으로 지정되었을 경우, source_skel 해당 목록에 없으면 skip
        if selected_source_skeletons is not None and source_skel not in selected_source_skeletons:
            continue

        # Paired mode: derive target skeletons from the txt file for this source.
        # Default mode: all other skeletons in the group.
        if paired_pairs is not None:
            target_skels = sorted({tgt for (src, _, tgt, _) in paired_pairs if src == source_skel})
            if not target_skels:
                continue  # this source has no entries in the paired file
        else:
            target_skels = [s for s in skeletons if s != source_skel]
            if not target_skels:
                continue

        print(f"RetargetDataset of {source_skel} -> {target_skels}:")
        use_self_reconstruction = getattr(args, 'use_self_reconstruction', True)
        use_cross_reconstruction = getattr(args, 'use_cross_reconstruction', True)
        dataset = RetargetDataset(
            args=args,
            data_root=data_dir,
            num_frames=num_frames,
            temporal_window=temporal_window,
            t5_name=t5_name,
            source_skeleton=source_skel,
            target_skeletons=target_skels,
            use_augmentation=False,
            include_self_reconstruction=use_self_reconstruction,
            include_cross_reconstruction=use_cross_reconstruction,
            t5_conditioner=shared_t5,
            joints_names_embs_cache=shared_embs_cache,
            paired_pairs=paired_pairs,
        )
        if len(dataset) == 0:
            continue
        datasets.append(dataset)

    # Concatenate all datasets
    combined_dataset = ConcatDataset(datasets)

    print(f"\n✅ Combined dataset created:")
    print(f"   Individual datasets: {len(datasets)}")
    print(f"   Total samples: {len(combined_dataset)}")
    print()

    # Create DataLoader
    loader = DataLoader(
        combined_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=getattr(args, 'num_workers', 0),
        drop_last=True,
        collate_fn=collate_retarget_batch,
        pin_memory=True if getattr(args, 'num_workers', 0) > 0 else False
    )

    print(f"✅ DataLoader created: {len(combined_dataset)} samples, "
          f"{len(loader)} batches per epoch\n")

    return loader
