"""Create a separate, auditable training dataset. Original files are untouched."""
import argparse
import csv
import hashlib
import json
import random
import shutil
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', default='data/ar-plaquesegm1')
    ap.add_argument('--out', default='data/plaque_v2')
    ap.add_argument('--groups', help='CSV: filename,patient_id; must cover every train/test image')
    ap.add_argument('--allow-image-split', action='store_true', help='Exploratory only: independence not established')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    src, out = Path(args.source), Path(args.out)
    if out.exists():
        ap.error('Output exists; choose a new --out to avoid overwriting data')
    if not args.groups and not args.allow_image_split:
        ap.error('Supply patient groups, or explicitly use --allow-image-split for exploratory work')
    groups = {}
    if args.groups:
        with open(args.groups, newline='') as f:
            for row in csv.DictReader(f):
                key, group = row['filename'], row['patient_id'].strip()
                if not group or key in groups:
                    ap.error('Empty patient ID or duplicate filename in groups CSV')
                groups[key] = group
    records = []
    seen = {}
    for split in ['train', 'test']:
        for ip in sorted((src/split).glob('*.png')):
            if '_labeled' in ip.stem:
                continue
            mp = ip.with_name(ip.stem.strip() + '_labeled.png')
            if not mp.exists():
                ap.error(f'Missing mask: {ip}')
            key = f'{split}/{ip.name}'
            if args.groups and key not in groups:
                ap.error(f'Missing patient group for {key}')
            digest = hashlib.sha256(ip.read_bytes()).hexdigest()
            if digest in seen:
                ap.error(f'Duplicate image bytes: {key} and {seen[digest]}')
            seen[digest] = key
            records.append((split, ip, mp, groups.get(key, key)))
    test_groups = {r[3] for r in records if r[0] == 'test'}
    train_groups = {r[3] for r in records if r[0] == 'train'}
    if test_groups & train_groups:
        ap.error('Patient overlap between original train and test; rebuild source split by patient first')
    ids = sorted(train_groups)
    if len(ids) < 2 or not test_groups:
        ap.error('Need at least two training groups and nonempty test')
    random.Random(args.seed).shuffle(ids)
    val = set(ids[:max(1, min(len(ids)-1, round(len(ids)*0.2)))])
    manifest = {'seed': args.seed, 'patient_grouped': bool(args.groups),
                'test_status': 'Previously used for model selection; NOT untouched confirmatory test', 'items': []}
    for original, ip, mp, group in records:
        split = 'test' if original == 'test' else ('val' if group in val else 'train')
        d = out/split
        d.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ip, d/ip.name)
        shutil.copy2(mp, d/mp.name)
        manifest['items'].append({'source': f'{original}/{ip.name}', 'split': split, 'patient_id': group})
    (out/'split_manifest.json').write_text(json.dumps(manifest, indent=2))
    print({s: sum(x['split']==s for x in manifest['items']) for s in ['train','val','test']})
    print('Start a fresh model: existing plaque checkpoints have already seen validation/test scores.')

if __name__ == '__main__':
    main()
