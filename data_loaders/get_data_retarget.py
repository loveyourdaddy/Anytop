"""
Data loader for retargeting training
Following the AnyTop get_data.py pattern
"""
from data_loaders.truebones.truebones_utils.param_utils import OBJECT_SUBSETS_DICT
from torch.utils.data import DataLoader
from data_loaders.truebones.data.dataset_retarget import RetargetDataset, collate_retarget_batch

from torch.utils.data import ConcatDataset


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


# def get_retarget_dataset_loader(
#     args,
#     batch_size,
#     num_frames,
#     split='train',
#     temporal_window=31,
#     t5_name='t5-base',
#     source_skeleton=None,
#     target_skeletons=None,
#     data_root='dataset/truebones/zoo/truebones_processed',
#     balanced=False,
#     use_augmentation=False,
#     num_workers=8
# ):
#     """
#     Get data loader for retargeting training

#     Args:
#         batch_size: Batch size
#         num_frames: Number of frames per sequence
#         split: Train/val/test split (currently unused)
#         temporal_window: Temporal window size for masking
#         t5_name: T5 model name for text conditioning
#         source_skeleton: Source skeleton type (e.g., 'Alligator')
#         target_skeletons: List of target skeleton types (e.g., ['Horse', 'Parrot2'])
#         data_root: Root directory of motion data
#         balanced: Whether to use balanced sampling (currently unused for retargeting)
#         use_augmentation: Whether to use joint augmentation
#         num_workers: Number of data loading workers

#     Returns:
#         DataLoader instance
#     """
#     # Get dataset
#     dataset = get_retarget_dataset(
#         args,
#         num_frames=num_frames,
#         split=split,
#         temporal_window=temporal_window,
#         t5_name=t5_name,
#         source_skeleton=source_skeleton,
#         target_skeletons=target_skeletons,
#         data_root=data_root,
#         use_augmentation=use_augmentation
#     )

#     # Collate function
#     collate = collate_retarget_batch

#     # Sampler
#     sampler = None
#     if balanced:
#         from data_loaders.truebones.data.dataset_retarget import RetargetSampler
#         print("Using balanced sampling for retargeting...")
#         sampler = RetargetSampler(dataset)

#     # Create dataloader
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         sampler=sampler,
#         shuffle=True if sampler is None else False,
#         num_workers=num_workers if num_workers > 0 else 0,  # ✅ 0이면 main process에서 로드
#         drop_last=True,
#         collate_fn=collate,
#         pin_memory=True if num_workers > 0 else False  # num_workers=0일 때 pin_memory=False
#     )

#     return loader


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

    # Use existing RetargetDataset!
    # args.source_skeleton: 지정된 경우 해당 skeleton만 source로 사용, None이면 group 전체 사용
    selected_source_skeletons = getattr(args, 'source_skeleton', None)  # list or None
    data_dir = getattr(args, 'data_dir', './dataset/truebones/zoo/truebones_processed')
    datasets = []
    for source_skel in skeletons:
        # source skeleton이 지정된 경우, 해당 목록에 없으면 skip
        if selected_source_skeletons is not None and source_skel not in selected_source_skeletons:
            continue
        
        # 지정된 group안에 있는 skeleton만 선택
        target_skels = [s for s in skeletons if s != source_skel]
        if not target_skels:
            continue

        print(f"RetargetDataset of {source_skel} -> {target_skels}:")
        include_self = getattr(args, 'self_reconstruction', True)
        dataset = RetargetDataset(
            args=args,
            data_root=data_dir,
            num_frames=num_frames,
            temporal_window=temporal_window,
            t5_name=t5_name,
            source_skeleton=source_skel,
            target_skeletons=target_skels,
            use_augmentation=False,
            include_self_reconstruction=include_self
        )

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
