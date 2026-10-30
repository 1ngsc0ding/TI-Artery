"""Preserve annotations and split identical decoded images together. Originals untouched."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil
import tempfile
import cv2


def prepare(source, destination, seed=42, groups_path=None):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise ValueError(f'Output already exists: {destination}. Choose a new --out; do not delete originals.')
    groups = {}
    if groups_path:
        with open(groups_path, newline='', encoding='utf-8-sig') as f:
            for row in csv.DictReader(f):
                key, patient = row['filename'], row['patient_id'].strip()
                if not patient or key in groups:
                    raise ValueError(f'Empty/duplicate patient entry: {key}')
                groups[key] = patient
    records = []
    for split in ('train', 'test'):
        folder = source/split
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing source folder: {folder}')
        for image in sorted(folder.iterdir()):
            if image.suffix.lower() != '.png' or '_labeled' in image.stem:
                continue
            mask = image.with_name(image.stem.strip()+'_labeled.png')
            gray = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            annotation = cv2.imread(str(mask), cv2.IMREAD_GRAYSCALE)
            if gray is None or annotation is None:
                raise ValueError(f'Cannot read image/mask: {image}, {mask}')
            if gray.shape != annotation.shape:
                raise ValueError(f'Shape mismatch: {image}, {mask}')
            key = f'{split}/{image.name}'
            if groups_path and key not in groups:
                raise ValueError(f'Missing patient entry: {key}')
            digest = hashlib.sha256(str(gray.shape).encode()+gray.tobytes()).hexdigest()
            records.append(dict(original_split=split, image=image, mask=mask, key=key,
                                digest=digest, patient=groups.get(key)))
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a,b):
        parent[find(b)] = find(a)
    images, patients = {}, {}
    for i,r in enumerate(records):
        if r['digest'] in images:
            union(i, images[r['digest']])
        images[r['digest']] = i
        if r['patient'] is not None:
            if r['patient'] in patients:
                union(i,patients[r['patient']])
            patients[r['patient']] = i
    test_groups = {find(i) for i,r in enumerate(records) if r['original_split']=='test'}
    train_groups = sorted({find(i) for i,r in enumerate(records)}-test_groups)
    if len(train_groups) < 2 or not test_groups:
        raise ValueError('Need >=2 independent train groups and nonempty test')
    random.Random(seed).shuffle(train_groups)
    validation = set(train_groups[:max(1,min(len(train_groups)-1,round(.2*len(train_groups))))])
    manifest = dict(seed=seed, patient_grouped=bool(groups_path),
        policy='Keep all annotations; identical grayscale pixels and supplied patient groups stay together; original test membership takes priority',
        test_status='Exploratory: original test was used for model selection. New untouched test required for confirmation.', items=[])
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.plaque_prepare_',dir=destination.parent))
    try:
        for split in ('train','val','test'):
            (staging/split).mkdir()
        for i,r in enumerate(records):
            group=find(i)
            split='test' if group in test_groups else ('val' if group in validation else 'train')
            # Prefix source split so moving a train member into test cannot overwrite a test image.
            stem=r['original_split']+'_'+r['image'].stem.strip()
            image_target=staging/split/(stem+'.png')
            mask_target=staging/split/(stem+'_labeled.png')
            if image_target.exists() or mask_target.exists():
                raise ValueError(f'Output filename collision: {stem}')
            shutil.copy2(r['image'],image_target)
            shutil.copy2(r['mask'],mask_target)
            manifest['items'].append(dict(source=r['key'],split=split,group=group,
                patient_id=r['patient'],image_sha256=r['digest'],image=image_target.name,mask=mask_target.name))
        (staging/'split_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging,ignore_errors=True)
        raise
    counts={s:sum(x['split']==s for x in manifest['items']) for s in ('train','val','test')}
    print('DONE:', counts)
    print('Annotations preserved:',len(records))
    print('Repeated-image pairs:',len(records)-len(images))
    print('Dataset:',destination)
    print('Exploratory split; patient independence is not established without a patient CSV.')
    return manifest


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',required=True)
    ap.add_argument('--out',required=True)
    ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--groups',help='CSV columns filename,patient_id')
    ap.add_argument('--allow-image-split',action='store_true')
    a=ap.parse_args()
    if not a.groups and not a.allow_image_split:
        ap.error('Supply --groups or explicitly choose --allow-image-split for exploratory work')
    try:
        prepare(a.source,a.out,a.seed,a.groups)
    except Exception as e:
        ap.exit(1,f'{type(e).__name__}: {e}\n')

if __name__=='__main__':
    main()
