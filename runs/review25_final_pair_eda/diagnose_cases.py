"""Read-only CPU OOF diagnostics for four intentionally selected difficult cases.

No inference/training/tuning. Server artifacts are rehashed against review22.
Raw image displays are written only to the ignored document QA directory.
"""
import base64
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
QA = ROOT/'reports/modeling_document_work/final_pair_eda'
WORKER = r'''
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
 os.environ[k]='2'
os.environ['CUDA_VISIBLE_DEVICES']=''
import base64,hashlib,io,json,sys,time
from pathlib import Path
import cv2
import numpy as np
from PIL import Image,ImageDraw
ROOT=Path('/home/j-j15a106/komap')
sys.path.insert(0,str(ROOT/'src'))
from data import load_split,MASK_RGB
from infer import image_iou,postprocess
cv2.setNumThreads(1)
start=time.monotonic()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
prior=read(ROOT/'runs/review22_pair/results.json')
plan=read(ROOT/'runs/review22_pair/plan.json')
reference=read(ROOT/'runs/review22_pair/per_image_rows.json')
names=['A11_step1_sample4','A42_step2_sample4','A147_step4_sample4','A139_step3_sample4']
def verify(r):
 p=ROOT/r['path']; assert hashlib.sha256(p.read_bytes()).hexdigest()==r['sha256'],str(p)
 return r
sources=[verify(r) for r in plan['source_receipts']]
sources += [verify(r) for r in plan['data_receipts'] if any(n in r['path'] for n in names)]
items={it['name']:it for it in load_split(ROOT/'data','train')+load_split(ROOT/'data','valid') if it['name'] in names}
classes=['Al','Al3Ni','Primary Si','Eutectic Si']
def confusion(gt,pr,region=None):
 if region is not None: gt,pr=gt[region],pr[region]
 c=np.bincount(gt.ravel().astype(np.int64)*4+pr.ravel(),minlength=16).reshape(4,4)
 out={'matrix_gt_rows_prediction_columns':c.tolist(),'pixels':int(c.sum()),'pixel_error_fraction':float(1-c.trace()/max(1,c.sum()))}
 for k in (1,3):
  tp=int(c[k,k]); fn=int(c[k].sum()-tp); fp=int(c[:,k].sum()-tp)
  out[classes[k]]={'tp':tp,'fn':fn,'fp':fp,'IoU_pct':100*tp/max(1,tp+fn+fp),
   'recall_pct':100*tp/max(1,tp+fn),'precision_pct':100*tp/max(1,tp+fp),
   'FN_predicted_as':{classes[j]:int(c[k,j]) for j in range(4) if j!=k},
   'FP_true_class':{classes[j]:int(c[j,k]) for j in range(4) if j!=k}}
 return out
rows={}; panels={}
for name in names:
 item=items[name];gt=item['lab'];gray=item['gray']; p={}
 for source in ('candidate','unet','x2'):
  r=next(r for r in prior['source_files'] if r['name']==name and r['source']==source)
  sources.append(verify(r));p[source]=np.load(ROOT/r['path']).astype(np.float32)
 probs={'MiT4':p['candidate'],'UNet3':p['unet'],'sub6':(p['x2']+p['unet'])/2,
        'final_pair':(p['candidate']+p['unet'])/2}
 preds={k:postprocess(v) for k,v in probs.items()}
 expected=np.asarray(reference['rows']['x4_mit_x3_unetpp_pair'][name]['tta_pp'],float)
 np.testing.assert_allclose(image_iou(gt,preds['final_pair']),expected,atol=1e-10,equal_nan=True)
 edge=np.zeros(gt.shape,np.uint8)
 d=gt[:,1:]!=gt[:,:-1];edge[:,1:]|=d;edge[:,:-1]|=d
 d=gt[1:]!=gt[:-1];edge[1:]|=d;edge[:-1]|=d
 rows[name]={'shape':list(gt.shape),'alloy':item['alloy'],'step':item['step'],
  'models':{k:{'IoU_pct':(image_iou(gt,v)*100).tolist(),'confusion':confusion(gt,v)} for k,v in preds.items()},'boundary_regions':{}}
 for radius in (1,2):
  band=cv2.dilate(edge,np.ones((2*radius+1,2*radius+1),np.uint8)).astype(bool)
  rows[name]['boundary_regions'][str(radius)]={
   'boundary_band':confusion(gt,preds['final_pair'],band),
   'interior':confusion(gt,preds['final_pair'],~band)}
 rgb=np.repeat(gray[:,:,None],3,axis=2)
 err=preds['final_pair']!=gt
 overlay=rgb.copy();overlay[err]=(0.35*rgb[err]+0.65*np.array([255,0,0])).astype(np.uint8)
 tile_width=450;tile_height=round(gray.shape[0]*tile_width/gray.shape[1]);header=52
 canvas=Image.new('RGB',(tile_width*4,tile_height+header),'white');draw=ImageDraw.Draw(canvas)
 draw.text((7,5),name+' | red = any wrong class | display resized, metrics native pixels',fill='black')
 for idx,(title,img) in enumerate([('Original',rgb),('Ground truth',MASK_RGB[gt]),('Final MiT4 + UNet3',MASK_RGB[preds['final_pair']]),('Error overlay',overlay)]):
  draw.text((idx*tile_width+7,28),title,fill='black')
  mode=Image.Resampling.BILINEAR if idx in (0,3) else Image.Resampling.NEAREST
  canvas.paste(Image.fromarray(img).resize((tile_width,tile_height),mode),(idx*tile_width,header))
 buf=io.BytesIO();canvas.save(buf,format='PNG');panels[name]=base64.b64encode(buf.getvalue()).decode()
result={'selection':'Two largest observed mIoU regressions versus sub6, plus two lowest observed final-pair Eutectic IoUs; selected examples are not representative or independent validation.',
 'classes':classes,'sources':sources,'cases':rows,'boundary_definition':'Ground-truth four-neighbor class changes marked on both sides, dilated by a fixed square radius 1 or 2 in native image pixels. Region metrics are conditional descriptive measures, not tolerance scores or threshold tuning.',
 'duration_seconds':time.monotonic()-start,'gpu_used':False,'training':False,'threshold_or_weight_search':False,'panels':panels}
def clean(x):
 if isinstance(x,float) and not np.isfinite(x):return None
 if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
 if isinstance(x,list):return [clean(v) for v in x]
 return x
print(json.dumps(clean(result),allow_nan=False))
'''


def main():
    r = subprocess.run(['ssh','-o','BatchMode=yes','komap',
        'cd /home/j-j15a106/komap && .venv/bin/python -'],
        input=WORKER.encode(),capture_output=True,check=True,timeout=180)
    result=json.loads(r.stdout)
    QA.mkdir(parents=True,exist_ok=True)
    panels=result.pop('panels')
    paths=[]
    for name,data in panels.items():
        p=QA/f'{name}.png';p.write_bytes(base64.b64decode(data));paths.append(str(p.relative_to(ROOT)))
    result['qa_images_ignored_by_git']=paths
    (OUT/'case_diagnostics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'duration_seconds':result['duration_seconds'],'cases':list(result['cases']),'images':paths},indent=2))


if __name__=='__main__':main()
