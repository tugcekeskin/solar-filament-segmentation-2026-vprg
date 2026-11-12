import sys,json,itertools;sys.path.insert(0,'src')
import pandas as pd
from filseg.infer import load_cache
from filseg.gt import build_gt_df
from filseg.metrics import fast_evaluate
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows
stems=sorted(json.load(open('data/splits/split_v1.json'))['val'])
gt=build_gt_df('data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',set(stems))
def score(rows): 
    r=fast_evaluate(gt,pd.DataFrame(rows,columns=['filament_id','segmentation_rle']));return f'PQ={r.pq:.4f} SQ={r.sq:.4f} RQ={r.rq:.4f} TP={r.tp} FP={r.fp} FN={r.fn}'
for name in sys.argv[1:]:
    print('==',name)
    raw=load_cache(name)
    if name.endswith('native070.pkl'):
        rows=[(f'{s}_{i}',r) for s,g in raw.groupby('stem') for i,r in enumerate(g.rle,1)]
        print('direct native 0.70:',score(rows),'inst/stem=%.2f'%(len(raw)/142), 'stems_with=%d'%raw.stem.nunique()); continue
    g=group_by_stem(raw);s=group_soft_by_stem(raw)
    for conf,thr,er in itertools.product([.25,.3,.5,.7],[.5,.7],[0,1]):
        p=PPParams(conf,100,20,0.3,er,'cross',thr)
        rows=postprocess_to_rows(g,p,stems,s)
        print(f'conf={conf} thr={thr} erode={er}: {score(rows)} inst/stem={len(rows)/142:.2f}',flush=True)
