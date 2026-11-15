import sys,json;sys.path.insert(0,'src')
import pandas as pd
from filseg.infer import load_cache
from filseg.gt import build_gt_df
from filseg.metrics import fast_evaluate
from filseg.postprocess import PPParams, group_by_stem, group_soft_by_stem, postprocess_to_rows
cache=sys.argv[1]; d=json.load(open(sys.argv[2]))['best_params'] if sys.argv[2].endswith('.json') else None
stems=sorted(json.load(open('data/splits/split_v1.json'))['val'])
gt=build_gt_df('data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json',set(stems))
raw=load_cache(cache);g=group_by_stem(raw);s=group_soft_by_stem(raw)
grid=[d] if len(sys.argv)<4 else [dict(d,**json.loads(x)) for x in sys.argv[3:]]
for q in grid:
  p=PPParams(q['conf'],q['min_area'],q['max_det'],q['nms'],q['erode'],q['erode_shape'],q['mask_thr'])
  r=fast_evaluate(gt,pd.DataFrame(postprocess_to_rows(g,p,stems,s),columns=['filament_id','segmentation_rle']))
  print(q,f'PQ={r.pq:.4f} SQ={r.sq:.4f} RQ={r.rq:.4f} TP={r.tp} FP={r.fp} FN={r.fn}',flush=True)
