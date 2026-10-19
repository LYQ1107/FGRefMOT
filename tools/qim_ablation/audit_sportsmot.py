#!/usr/bin/env python3
"""Audit the local SportsMOT + Refer-Sports expression protocol."""
import argparse
import json
from collections import Counter
from pathlib import Path


def image_key(path):
    try:
        return int(path.stem)
    except ValueError:
        return path.name


def read_label_ids(path):
    ids = set()
    with path.open() as handle:
        for line in handle:
            fields = line.replace(',', ' ').split()
            if len(fields) >= 6:
                ids.add(int(float(fields[1])))
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--sportsmot-root', required=True)
    parser.add_argument('--expression-root', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    publish = Path(args.sportsmot_root)
    dataset = publish / 'dataset'
    expression_root = Path(args.expression_root)
    manifest = json.loads(Path(args.manifest).read_text())
    records = manifest['records']

    split_summary = {}
    for split in ('train', 'val', 'test'):
        split_path = publish / 'splits_txt' / '{}.txt'.format(split)
        names = [x.strip() for x in split_path.read_text().splitlines() if x.strip()]
        dirs = [name for name in names if (dataset / split / name).is_dir()]
        images = sum(len(list((dataset / split / name / 'img1').glob('*'))) for name in dirs)
        gt = sum((dataset / split / name / 'gt' / 'gt.txt').is_file() for name in dirs)
        split_summary[split] = {
            'listed_videos': len(names),
            'video_dirs': len(dirs),
            'image_files': images,
            'gt_txt_files': gt,
        }

    expression_paths = sorted(expression_root.glob('*/*.json'))
    expression_ids = {p.parent.name + ':' + p.stem for p in expression_paths}
    manifest_ids = {r['expression_id'] for r in records}
    manifest_by_video = {}
    for record in records:
        manifest_by_video.setdefault(record['original_sportsmot_id'], []).append(record)

    positive_checks = Counter()
    missing_expression = []
    missing_label_dirs = []
    missing_label_frames = []
    positive_id_mismatches = []
    for record in records:
        video = record['original_sportsmot_id']
        expression_id = record['expression_id']
        expression_path = expression_root / video / (expression_id.split(':', 1)[1] + '.json')
        if not expression_path.is_file():
            missing_expression.append(expression_id)
            continue
        expression = json.loads(expression_path.read_text())
        label_dir = dataset / 'labels_with_ids' / video
        if not label_dir.is_dir():
            missing_label_dirs.append(video)
            continue
        for frame, object_ids in expression.get('label', {}).items():
            label_path = label_dir / '{:06d}.txt'.format(int(frame))
            if not label_path.is_file():
                missing_label_frames.append('{}:{}'.format(expression_id, frame))
                continue
            available = read_label_ids(label_path)
            requested = {int(x) for x in object_ids}
            if not requested.issubset(available):
                positive_id_mismatches.append({
                    'expression_id': expression_id,
                    'frame': int(frame),
                    'requested': sorted(requested),
                    'available': sorted(available),
                })
                positive_checks['mismatch'] += 1
            else:
                positive_checks['matched_frames'] += 1
            positive_checks['labelled_frames'] += 1

    summary = {
        'sportsmot_root': str(publish),
        'expression_root': str(expression_root),
        'manifest': str(Path(args.manifest)),
        'split_summary': split_summary,
        'manifest_records': len(records),
        'manifest_videos': len(manifest_by_video),
        'manifest_split_counts': dict(Counter(r['original_split'] for r in records)),
        'local_expression_files': len(expression_paths),
        'local_expression_videos': len({p.parent.name for p in expression_paths}),
        'expression_ids_missing_locally': sorted(manifest_ids - expression_ids),
        'local_expression_ids_not_in_manifest': sorted(expression_ids - manifest_ids),
        'positive_label_checks': dict(positive_checks),
        'missing_expression_files': sorted(set(missing_expression)),
        'missing_global_label_dirs': sorted(set(missing_label_dirs)),
        'missing_global_label_frames_count': len(missing_label_frames),
        'missing_global_label_frames_examples': missing_label_frames[:20],
        'positive_id_mismatches_count': len(positive_id_mismatches),
        'positive_id_mismatch_examples': positive_id_mismatches[:20],
        'protocol_note': (
            'The expression manifest defines positive target IDs; absent IDs are UNKNOWN. '
            'The global labels_with_ids namespace is required for expression ID matching. '
            'The published test split has images but no local GT labels, so it is not a '
            'supervised training or HOTA ground-truth source.'
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
