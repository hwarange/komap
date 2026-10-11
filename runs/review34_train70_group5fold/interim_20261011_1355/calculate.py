"""Read-only CPU probability analysis of completed fold2 pairs and matched42 OOF."""
import os
from pathlib import Path
import sys
ROOT=Path('/home/j-j15a106/komap')
BASE=ROOT/'runs/review34_train70_group5fold'
OUT=BASE/'interim_20261011_1355'
OLD=BASE/'interim_20261011_1215'
os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',
                  MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
os.chdir(ROOT)
from review_group34 import read,sha,save,relative,now,verify,load_training70,split_job,probability,image_row,summary

def main():
    import cv2,json,numpy as np,torch
    from infer import postprocess
    torch.set_num_threads(4);cv2.setNumThreads(1)
    assert not (OUT/'results.json').exists() and not (OUT/'plan.json').exists()
    frozen=read(BASE/'plan.json');assignment=verify(frozen)
    items=load_training70(frozen,assignment)
    valid=[item for item in items if assignment[item['name']]==2]
    assert len(valid)==14
    families=['mit2','unet3']
    if (BASE/'reviews/mit4_fold2/all.done').exists():families.append('mit4')
    pins=dict(frozen['input_hashes'])
    pins[relative(BASE/'plan.json')]=sha(BASE/'plan.json')
    pins[relative(Path(__file__))]=sha(Path(__file__))
    old_done=read(OLD/'all.done')
    for key,name in [('plan_sha256','plan.json'),('results_sha256','results.json'),('rows_sha256','per_image_rows.json')]:
        assert old_done[key]==sha(OLD/name);pins[relative(OLD/name)]=sha(OLD/name)
    pins[relative(OLD/'all.done')]=sha(OLD/'all.done')
    old_plan=read(OLD/'plan.json')
    assert old_plan['frozen_plan']['sha256']==sha(BASE/'plan.json')
    old=read(OLD/'per_image_rows.json')
    manifests,stored={},{}
    for family in families:
        identifier=family+'_fold2';job=next(j for j in frozen['jobs'] if j['id']==identifier)
        split_job(job,frozen,items,assignment)
        folder=BASE/'reviews'/identifier;done=read(folder/'all.done')
        assert done['job']==identifier and done['plan_sha256']==sha(BASE/'plan.json')
        assert done['results_sha256']==sha(folder/'results.json') and done['rows_sha256']==sha(folder/'per_image_rows.json')
        result=read(folder/'results.json')
        assert result['args']==job['args'] and result['strict_cpu_reload'] and result['finite_checkpoint_tensors']
        assert result['training_names']==job['training_names'] and result['validation_names']==job['validation_names']
        assert set(result['probabilities'])=={item['name'] for item in valid}
        assert not result['official_validation_images_used'] and not result['test_images_or_labels_used']
        for receipt in result['output_receipts']:
            path=(ROOT/receipt['path']).resolve();assert path.is_relative_to(BASE.resolve())
            assert sha(path)==receipt['sha256'];pins[receipt['path']]=receipt['sha256']
        for path in result['probabilities'].values():assert path in pins
        for name in ('all.done','results.json','per_image_rows.json'):pins[relative(folder/name)]=sha(folder/name)
        manifests[family]=result;stored[family]=read(folder/'per_image_rows.json')['original']
    pairs={'mit2_plus_unet3':('mit2','unet3')}
    if 'mit4' in families:pairs['mit4_plus_unet3']=('mit4','unet3')
    plan=dict(created=now(),frozen_plan_sha256=sha(BASE/'plan.json'),families=families,new_fold=2,new_images=14,
        previous_folds=[0,1],previous_images=28,combined_images=42,context4_fold2='pending_not_evaluated',
        input_hashes=pins,recipes={key:dict(models=list(value),weights=[.5,.5]) for key,value in pairs.items()},
        cpu_only=True,no_new_training=True,no_new_inference=True,automatic_adoption=False)
    save(OUT/'plan.json',plan)
    fold2={key:{} for key in [*families,*pairs]}
    for item in valid:
        name=item['name'];probs={family:probability(ROOT/manifests[family]['probabilities'][name],item) for family in families}
        for family in families:
            assert name not in manifests[family]['training_names']
            fold2[family][name]=image_row(item,postprocess(probs[family]),2)
            np.testing.assert_allclose(np.asarray(fold2[family][name]['tta_pp'],dtype=float),
                np.asarray(stored[family][name]['tta_pp'],dtype=float),rtol=0,atol=0,equal_nan=True)
        for recipe,(first,second) in pairs.items():
            fold2[recipe][name]=image_row(item,postprocess((probs[first]+probs[second])/2),2)
    combined={}
    expected={item['name'] for item in items if assignment[item['name']] in (0,1,2)}
    for recipe in pairs:
        assert len(old[recipe])==28 and set(old[recipe]).isdisjoint(fold2[recipe])
        assert all(row['fold'] in (0,1) and assignment[name]==row['fold'] for name,row in old[recipe].items())
        combined[recipe]={**old[recipe],**fold2[recipe]}
        assert set(combined[recipe])==expected and len(expected)==42
    fold2_scores={key:summary(rows) for key,rows in fold2.items()}
    for family in families:assert abs(fold2_scores[family]['mIoU']-manifests[family]['scores']['original']['mIoU'])<1e-12
    combined_scores={key:summary(rows) for key,rows in combined.items()}
    comparison=None
    if 'mit4' in families:
        names=sorted(expected);first='mit2_plus_unet3';second='mit4_plus_unet3'
        delta={name:float((np.nanmean(np.asarray(combined[first][name]['tta_pp'],dtype=float))-
                          np.nanmean(np.asarray(combined[second][name]['tta_pp'],dtype=float)))*100) for name in names}
        values=np.asarray(list(delta.values()))
        comparison=dict(scope='Matched42 held-out images fromfolds0/1/2',direction='MiT2 pair minus MiT4 pair',
            mean_delta_pp=float(values.mean()),improved=int((values>0).sum()),worsened=int((values<0).sum()),
            unchanged=int((values==0).sum()),per_image_delta_pp=delta,
            class_delta_pp={key:combined_scores[first]['class_IoU'][key]-combined_scores[second]['class_IoU'][key]
                            for key in combined_scores[first]['class_IoU']})
    for name,digest in pins.items():assert sha(ROOT/name)==digest,name
    verify(frozen)
    result=dict(completed=now(),plan_sha256=sha(OUT/'plan.json'),partial=True,
        fold2_images=14,fold2_scores=fold2_scores,combined_images=42,combined_scores=combined_scores,
        matched42_comparison=comparison,context4_fold2='pending_not_evaluated',
        pending_folds=[3,4],remaining_context_fold2=True,source_singles_exactly_reproduced=True,
        cpu_only=True,new_training=False,new_inference=False,official_validation_images_used=False,
        test_images_or_labels_used=False,automatic_model_selection=False,automatic_submission_change=False,
        notes=['Only completedfold2 jobs are added; each pair uses its own held-out-fold models.',
               '42-image scores must not be ranked against the other28-image partial results.',
               'This is partial grouped CV within train70, not the final70OOF result or an independent test.'])
    save(OUT/'per_image_rows.json',dict(fold2=fold2,combined=combined))
    save(OUT/'results.json',result)
    save(OUT/'all.done',dict(time=now(),partial=True,plan_sha256=sha(OUT/'plan.json'),
        results_sha256=sha(OUT/'results.json'),rows_sha256=sha(OUT/'per_image_rows.json'),automatic_submission_change=False))
    print(json.dumps({key:value for key,value in result.items() if key!='matched42_comparison'},indent=2),flush=True)
    if comparison:print(json.dumps({key:value for key,value in comparison.items() if key!='per_image_delta_pp'},indent=2),flush=True)

if __name__=='__main__':main()
