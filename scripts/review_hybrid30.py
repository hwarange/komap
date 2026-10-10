"""Fixed per-job evaluation and a final, paired comparison of approved pilots."""
import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.chdir(ROOT)
import run_hybrid30 as common

LABELS=('control_single_original','candidate_single_original','control_single_cap6','candidate_single_cap6',
        'control_pair_original','candidate_pair_original','selected_pair_cap6','candidate_pair_cap6')


def clean(value):
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    return value


def probability(path,item):
    import numpy as np
    stored=np.load(path,allow_pickle=False)
    assert stored.dtype==np.float16 and stored.shape==(4,*item['gray'].shape)
    assert np.isfinite(stored).all() and stored.min()>=0 and stored.max()<=1
    value=stored.astype(np.float32)
    assert np.max(np.abs(value.sum(0)-1))<.005
    return value


def scores(rows):
    import numpy as np
    from data import CLASSES
    return {label:dict(images=len(records),mIoU=float(np.mean([np.nanmean(x['tta_pp']) for x in records.values()])*100),
        class_IoU={c:float(np.nanmean([x['tta_pp'][i] for x in records.values()])*100) for i,c in enumerate(CLASSES)})
        for label,records in rows.items()}


def contrasts(rows,folds):
    from review_grains15 import metrics
    pairs=[('candidate_single_original','control_single_original'),('candidate_single_cap6','control_single_cap6'),
           ('candidate_pair_original','control_pair_original'),('candidate_pair_cap6','selected_pair_cap6'),
           ('candidate_pair_original','selected_pair_cap6')]
    return {a+'_vs_'+b:metrics(rows[a],rows[b],folds,repeats=10000) for a,b in pairs}


def review(identifier):
    import cv2
    import numpy as np
    import torch
    from data import load_cv
    from hybrid30 import install,predict_probs
    from infer import image_iou,postprocess
    train=install();torch.set_num_threads(4);cv2.setNumThreads(1)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
    plan=common.read(common.OUT/'plan.json');common.verify(plan)
    job=next(j for j in plan['jobs'] if j['id']==identifier);args=job['args'];fold=job['fold']
    prefix=f'reviews/{identifier}';out=common.OUT/prefix
    assert not out.exists();out.mkdir(parents=True)
    assert common.read(ROOT/args['out']/'args.json')==args
    logs=common.logs(job)
    assert [r['epoch'] for r in logs]==list(range(1,61))
    assert all(math.isfinite(r['loss']) and r['microbatches']==288 and r['optimizer_updates']==144 for r in logs)
    trained=common.read(ROOT/args['out']/'result.json')
    assert trained['args']==args and {r['name'] for r in trained['rows']}==set(job['validation_names'])
    checkpoint=ROOT/args['out']/'last.pt';ckpt=torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ckpt['epoch']==60 and ckpt['args']==args
    assert all(torch.isfinite(v).all() for v in ckpt['model'].values() if v.is_floating_point())
    model=train.build_from_args(args,weights=None);model.load_state_dict(ckpt['model'],strict=True);model.cuda().eval()
    _,valid=load_cv('data',fold,'splits/folds5.json')
    assert sorted(x['name'] for x in valid)==job['validation_names']
    cap=out/'cap6_oof';cap.mkdir()
    generated=[]
    for item in valid:
        if item['gray'].shape[1]!=675:continue
        with train.autocast(args['amp']):
            prob=predict_probs(model,item['gray'],args['width'],args['tile'],args['overlap'],tta=True,
                device='cuda',scale_multiplier=args['scale_multiplier'],extra_scale=.75)
        path=cap/f"{item['name']}.npy";np.save(path,prob.astype(np.float16));generated.append(path)
    del model,ckpt;torch.cuda.empty_cache()
    rows={label:{} for label in LABELS}
    reference=common.read(ROOT/'runs/review27_finetune_full/per_image_rows.json')
    for item in valid:
        n=item['name'];old=probability(ROOT/f'runs/oof/{common.BASE}/{n}.npy',item)
        candidate_path=ROOT/args['save_oof']/f'{n}.npy';new=probability(candidate_path,item);generated.append(candidate_path)
        u=probability(ROOT/job['interface_oof']/f'{n}.npy',item)
        if item['gray'].shape[1]==675:
            oc=probability(ROOT/job['cap6_control_records'][n]['path'],item);nc=probability(cap/f'{n}.npy',item)
        else:oc,nc=old,new
        values=(old,new,oc,nc,(old+u)/2,(new+u)/2,(oc+u)/2,(nc+u)/2)
        for label,p in zip(LABELS,values):
            rows[label][n]=dict(tta_pp=image_iou(item['lab'],postprocess(p)).tolist(),low=item['low_contrast'],fold=fold)
        for label,ref_label in [('control_pair_original','interface_original'),('selected_pair_cap6','interface_cap6')]:
            np.testing.assert_allclose(rows[label][n]['tta_pp'],reference[ref_label][n]['tta_pp'],rtol=0,atol=1e-12,equal_nan=True)
    result_scores=scores(rows);folds={n:fold for n in job['validation_names']}
    common.verify(plan)
    files=generated+[checkpoint,ROOT/args['out']/'args.json',ROOT/args['out']/'log.jsonl',ROOT/args['out']/'result.json']
    result=dict(completed=common.now(),job=identifier,fold=fold,images=18,epochs=60,architecture=job['architecture'],
        scores=result_scores,comparisons=contrasts(rows,folds),plan_sha256=common.sha(common.OUT/'plan.json'),
        checkpoint_sha256=common.sha(checkpoint),output_receipts=[dict(path=str(p.relative_to(ROOT)),sha256=common.sha(p)) for p in sorted(set(files))],
        training_minutes=sum(r['train_sec'] for r in logs)/60,control_probabilities_reproduce_selected_reference=True,
        test_images_or_labels_used=False,previously_viewed_validation=True,submission_changed=False)
    common.save(prefix+'/per_image_rows.json',clean(rows));common.save(prefix+'/results.json',clean(result))
    common.save(prefix+'/all.done',dict(time=common.now(),plan_sha256=common.sha(common.OUT/'plan.json'),
        results_sha256=common.sha(out/'results.json'),rows_sha256=common.sha(out/'per_image_rows.json'),
        checkpoint_sha256=common.sha(checkpoint),images=18,epochs=60))
    print(json.dumps(dict(job=identifier,scores=result_scores),indent=2),flush=True)


def aggregate():
    import numpy as np
    from review_grains15 import metrics
    plan=common.read(common.OUT/'plan.json');common.verify(plan)
    assert not (common.OUT/'results.json').exists() and not (common.OUT/'all.done').exists()
    old=ROOT/'runs/review29_hybrid_pilot';done29=common.read(old/'all.done')
    assert done29['plan_sha256']==plan['existing_gpu0_plan_sha256']
    assert done29['results_sha256']==common.sha(old/'results.json')
    assert done29['rows_sha256']==common.sha(old/'per_image_rows.json')
    original=common.read(old/'per_image_rows.json')
    records={};job_results={};done_hashes={}
    for job in plan['jobs']:
        path=common.OUT/'reviews'/job['id'];done=common.read(path/'all.done')
        assert done['plan_sha256']==common.sha(common.OUT/'plan.json')
        assert done['results_sha256']==common.sha(path/'results.json') and done['rows_sha256']==common.sha(path/'per_image_rows.json')
        records[job['id']]=common.read(path/'per_image_rows.json');job_results[job['id']]=common.read(path/'results.json')
        done_hashes[job['id']]=common.sha(path/'all.done')
    folds0={n:0 for n in original['candidate_single_original']}
    comparisons={}
    for identifier in ['gated_fold0','context_fold0']:
        rows=records[identifier]
        for label in LABELS:
            assert set(rows[label])==set(original[label])
            if not label.startswith('candidate_'):
                for n in rows[label]:np.testing.assert_allclose(rows[label][n]['tta_pp'],original[label][n]['tta_pp'],rtol=0,atol=1e-12,equal_nan=True)
        comparisons[identifier]={label:metrics(rows[label],original[label],folds0,repeats=10000)
            for label in LABELS if label.startswith('candidate_')}
    second=records['hybrid_fold1'];merged={}
    for label in LABELS:
        assert set(original[label]).isdisjoint(second[label])
        merged[label]={**original[label],**second[label]};assert len(merged[label])==36
    folds36={n:r['fold'] for n,r in merged['candidate_single_original'].items()}
    result=dict(completed=common.now(),plan_sha256=common.sha(common.OUT/'plan.json'),
        same_fold0_18=dict(base_hybrid=scores(original),gated=job_results['gated_fold0']['scores'],
            whole_context=job_results['context_fold0']['scores'],candidate_vs_base_hybrid=comparisons),
        hybrid_folds0_1_36=dict(scores=scores(merged),comparisons=contrasts(merged,folds36)),
        job_completion_sha256=done_hashes,existing_hybrid29_done_sha256=common.sha(old/'all.done'),
        independent_validation=False,test_images_or_labels_used=False,submission_changed=False,
        automatic_remaining_folds=[],notes=['Compare architectures only on matched18. Hybrid36 is a separate scope.',
            'Fresh ImageNet, seed0,60epochs for all. No per-image or per-fold recipe selection.',
            'Whole-view candidate also adds CNN capacity; context-only causality not isolated.',
            'All validation has been repeatedly inspected; bootstrap intervals do not remove selection bias.'])
    common.verify(plan)
    common.save('hybrid36_per_image_rows.json',clean(merged));common.save('results.json',clean(result))
    common.save('all.done',dict(time=common.now(),outcome='three_approved_hybrid_experiments_complete',
        plan_sha256=common.sha(common.OUT/'plan.json'),results_sha256=common.sha(common.OUT/'results.json'),
        rows36_sha256=common.sha(common.OUT/'hybrid36_per_image_rows.json'),job_completion_sha256=done_hashes,
        automatic_remaining_folds=[],submission_changed=False))
    print(json.dumps(clean(result),indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--job',choices=list(common.CONFIGS));g.add_argument('--aggregate',action='store_true')
    a=p.parse_args();aggregate() if a.aggregate else review(a.job)
