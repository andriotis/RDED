"""RDED Table 2 reference values (top-1 accuracy, % mean ± std).

Source: Sun et al., "On the Diversity and Realism of Distilled Datasets" (2312.03526v2),
Table 2. We only carry the cells in our experimental scope: {cifar100, tinyimagenet}
x {conv3/conv4, resnet18_modified} x {1, 10, 50}.

Keys match the values stored in our results.jsonl rows:
  - dataset matches args.subset
  - arch matches args.arch_name
  - ipc matches args.ipc
"""

RDED_TABLE2 = {
    ("cifar100", "conv3", 1):              (19.6, 0.2),
    ("cifar100", "conv3", 10):             (48.1, 0.3),
    ("cifar100", "conv3", 50):             (57.0, 0.2),
    ("cifar100", "resnet18_modified", 1):  (11.0, 0.3),
    ("cifar100", "resnet18_modified", 10): (42.6, 0.2),
    ("cifar100", "resnet18_modified", 50): (62.6, 0.1),
    ("tinyimagenet", "conv4", 1):              (12.0, 0.1),
    ("tinyimagenet", "conv4", 10):             (39.6, 0.2),
    ("tinyimagenet", "conv4", 50):             (47.6, 0.2),
    ("tinyimagenet", "resnet18_modified", 1):  (9.7, 0.4),
    ("tinyimagenet", "resnet18_modified", 10): (41.9, 0.2),
    ("tinyimagenet", "resnet18_modified", 50): (58.2, 0.1),
}


def expected_cells():
    """List of (dataset, arch, ipc) tuples we expect in the baseline matrix."""
    return list(RDED_TABLE2.keys())
