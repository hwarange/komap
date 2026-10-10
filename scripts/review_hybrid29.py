"""Complete fixed cap6 inference and compare the hybrid pilot on all18 held-out images."""
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
os.chdir(ROOT)
import run_hybrid29 as common


def clean(x):
    if isinstance(x, float) and not math.isfinite(x): return None
    if isinstance(x, dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x, list): return [clean(v) for v in x]
    return x


def probability(path, item):
    import numpy as np
    stored = np.load(path, allow_pickle=False)
    assert stored.dtype == np.float16 and stored.shape == (4, *item['gray'].shape)
    assert np.isfinite(stored).all() and stored.min() >= 0 and stored.max() <= 1
    value = stored.astype(np.float32)
    assert np.max(np.abs(value.sum(0)-1)) < 0.005
    return value


def main():
    import cv2
    import numpy as np
    import torch
    from data import load_cv, CLASSES
    from hybrid29 import install
    from infer import predict_probs, image_iou, postprocess
    from review_grains15 import metrics
    train = install()
    cv2.setNumThreads(1); torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
    out = common.OUT; plan = common.read(out/'plan.json'); common.verify(plan)
    assert not (out/'results.json').exists() and not (out/'all.done').exists()
    args = common.read(ROOT/plan['args']['out']/'args.json'); assert args == plan['args']
    logs = [json.loads(x) for x in (ROOT/args['out']/'log.jsonl').read_text().splitlines() if x]
    assert [x['epoch'] for x in logs] == list(range(1,61))
    assert all(math.isfinite(x['loss']) and x['optimizer_updates']==144 and x['microbatches']==288 for x in logs)
    result = common.read(ROOT/args['out']/'result.json'); assert result['args']==args
    assert {x['name'] for x in result['rows']} == set(plan['validation_names'])
    checkpoint = ROOT/args['out']/'last.pt'
    ckpt = torch.load(checkpoint,map_location='cpu',weights_only=False)
    assert ckpt['epoch']==60 and ckpt['args']==args
    assert all(torch.isfinite(x).all() for x in ckpt['model'].values() if x.is_floating_point())
    model = train.build_from_args(args,weights=None)
    model.load_state_dict(ckpt['model'],strict=True)
    model.cuda().eval()
    _, valid = load_cv('data',0,'splits/folds5.json')
    assert sorted(x['name'] for x in valid)==plan['validation_names']
    cap = out/'candidate_cap6_oof'; cap.mkdir()
    cap_receipts = []
    for item in valid:
        if item['gray'].shape[1] != 675: continue
        with torch.no_grad(), train.autocast(True):
            prob = predict_probs(model,item['gray'],args['width'],args['tile'],args['overlap'],
                tta=True,device='cuda',scale_multiplier=args['scale_multiplier'],extra_scale=0.75)
        path = cap/f"{item['name']}.npy"; np.save(path,prob.astype(np.float16))
        cap_receipts.append(dict(name=item['name'],path=str(path.relative_to(ROOT)),sha256=common.sha(path)))
    del model,ckpt; torch.cuda.empty_cache()
    labels = ('control_single_original','candidate_single_original','control_single_cap6','candidate_single_cap6',
              'control_pair_original','candidate_pair_original','selected_pair_cap6','candidate_pair_cap6')
    rows = {label:{} for label in labels}; receipt_paths = [checkpoint,ROOT/args['out']/'args.json',
        ROOT/args['out']/'log.jsonl',ROOT/args['out']/'result.json']
    for item in valid:
        n=item['name']; old_path=ROOT/f'runs/oof/{common.BASE}/{n}.npy'; new_path=ROOT/args['save_oof']/f'{n}.npy'
        unet_path=ROOT/f'runs/review26_final_strengthening/finetune_plan/oof/interface/{n}.npy'
        old,new,u=(probability(p,item) for p in [old_path,new_path,unet_path])
        if item['gray'].shape[1]==675:
            oc_path=ROOT/plan['cap6_control_records'][n]['path']; nc_path=cap/f'{n}.npy'
            oc,nc=probability(oc_path,item),probability(nc_path,item)
            receipt_paths += [oc_path,nc_path]
        else: oc,nc=old,new
        probs=(old,new,oc,nc,(old+u)/2,(new+u)/2,(oc+u)/2,(nc+u)/2)
        for label,p in zip(labels,probs):
            rows[label][n]=dict(tta_pp=image_iou(item['lab'],postprocess(p)).tolist(),low=item['low_contrast'],fold=0)
        receipt_paths += [old_path,new_path,unet_path]
    scores={label:dict(images=18,mIoU=float(np.mean([np.nanmean(x['tta_pp']) for x in records.values()])*100),
        class_IoU={c:float(np.nanmean([x['tta_pp'][i] for x in records.values()])*100) for i,c in enumerate(CLASSES)}) for label,records in rows.items()}
    reference=common.read(ROOT/'runs/review27_finetune_full/results.json')['scopes']['reused_fold0_18']['scores']
    assert abs(scores['control_pair_original']['mIoU']-reference['interface_original']['mIoU'])<1e-10
    assert abs(scores['selected_pair_cap6']['mIoU']-reference['interface_cap6']['mIoU'])<1e-10
    pairs=[('candidate_single_original','control_single_original'),('candidate_single_cap6','control_single_cap6'),
           ('candidate_pair_original','control_pair_original'),('candidate_pair_cap6','selected_pair_cap6'),
           ('candidate_pair_original','selected_pair_cap6')]
    folds={n:0 for n in plan['validation_names']}
    comparisons={a+'_vs_'+b:metrics(rows[a],rows[b],folds,repeats=10000) for a,b in pairs}
    candidate_names=['candidate_pair_original','candidate_pair_cap6']
    best=max(candidate_names,key=lambda n:scores[n]['mIoU'])
    common.verify(plan)
    output=dict(completed=common.now(),fold=0,images=18,epochs=60,scores=scores,comparisons=comparisons,
        highest_fixed_candidate_pair=best,delta_best_candidate_vs_selected_pair_pp=scores[best]['mIoU']-scores['selected_pair_cap6']['mIoU'],
        plan_sha256=common.sha(out/'plan.json'),architecture=plan['architecture'],checkpoint_sha256=common.sha(checkpoint),
        output_receipts=[dict(path=str(p.relative_to(ROOT)),sha256=common.sha(p)) for p in sorted(set(receipt_paths))],
        cap6_receipts=cap_receipts,training_minutes=sum(x['train_sec'] for x in logs)/60,
        pilot_only=True,independent_validation=False,test_images_or_labels_used=False,submission_changed=False,
        further_training_scheduled=False,notes=['Fresh60-epoch training; same control config except architecture/outputpaths.',
            'Only reused fold0 OOF18. Do not compare its score numerically to full90.',
            'Original and cap6_width675 are two fixed inference recipes. Confidence intervals are reported, not an adoption gate.',
            'No automatic submission selection or remaining-fold training.'])
    common.save('per_image_rows.json',clean(rows)); common.save('results.json',clean(output))
    common.save('all.done',dict(time=common.now(),outcome='hybrid60_fold0_fixed_review_complete',images=18,
        plan_sha256=common.sha(out/'plan.json'),results_sha256=common.sha(out/'results.json'),
        rows_sha256=common.sha(out/'per_image_rows.json'),checkpoint_sha256=common.sha(checkpoint),further_training_scheduled=False))
    print(json.dumps(clean(output),indent=2),flush=True)


if __name__=='__main__': main()
