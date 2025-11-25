"""
Data loader for retargeting training
Following the AnyTop get_data.py pattern
"""
from torch.utils.data import DataLoader
from data_loaders.truebones.data.dataset_retarget import RetargetDataset, collate_retarget_batch


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
    split='train',
    temporal_window=31,
    t5_name='t5-base',
    source_skeleton=None,
    target_skeletons=None,
    data_root='dataset/truebones/zoo/truebones_processed',
    balanced=False,
    use_augmentation=False,
    num_workers=8
):
    """
    Get data loader for retargeting training

    Args:
        batch_size: Batch size
        num_frames: Number of frames per sequence
        split: Train/val/test split (currently unused)
        temporal_window: Temporal window size for masking
        t5_name: T5 model name for text conditioning
        source_skeleton: Source skeleton type (e.g., 'Alligator')
        target_skeletons: List of target skeleton types (e.g., ['Horse', 'Parrot2'])
        data_root: Root directory of motion data
        balanced: Whether to use balanced sampling (currently unused for retargeting)
        use_augmentation: Whether to use joint augmentation
        num_workers: Number of data loading workers

    Returns:
        DataLoader instance
    """
    # Get dataset
    dataset = get_retarget_dataset(
        args,
        num_frames=num_frames,
        split=split,
        temporal_window=temporal_window,
        t5_name=t5_name,
        source_skeleton=source_skeleton,
        target_skeletons=target_skeletons,
        data_root=data_root,
        use_augmentation=use_augmentation
    )

    # Collate function
    collate = collate_retarget_batch

    # Sampler
    sampler = None
    if balanced:
        from data_loaders.truebones.data.dataset_retarget import RetargetSampler
        print("Using balanced sampling for retargeting...")
        sampler = RetargetSampler(dataset)

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=True if sampler is None else False,
        num_workers=num_workers if num_workers > 0 else 0,  # ✅ 0이면 main process에서 로드
        drop_last=True,
        collate_fn=collate,
        pin_memory=True if num_workers > 0 else False  # num_workers=0일 때 pin_memory=False
    )

    return loader
