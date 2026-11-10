"""Fixed-alpha W0 baseline and paired R experiment, with spatial O holdouts."""
import tibet_tao_kernel_v2 as kernel  # isolated JIT runtime must load before numpy
from pathlib import Path
import argparse,json,time
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
import rasterio
from rasterio.transform import Affine
from mead_full2d_alpha_masking_v1 import source_analysis as tao,sha,TAO,alpha_function
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'05_Results/codex/by_date/2026-09-15/tibet_three_lakes_v1'
OUT=ROOT/'05_Results/codex/by_date/2026-09-16/tibet_three_group_v2'
LAKES=['laang_co','mapang_yongco','ngangla_rinco']
def savejson(path,obj):path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')

def initialize(z,unknown):
    # All unknown heights are redacted before every derivative or shoreline fit.
    known=~unknown&np.isfinite(z);dem=np.where(known,z,-1.).astype(float)
    safe=np.where(known,z,0.)
    gx=ndi.correlate(safe,np.array([[-1,0,1],[-2,0,2],[-1,0,1]])/240,mode='nearest')
    gy=ndi.correlate(safe,np.array([[-1,-2,-1],[0,0,0],[1,2,1]])/240,mode='nearest')
    slope_deg=np.degrees(np.arctan(np.hypot(gx,gy)))
    slope_valid=ndi.minimum_filter(known.astype('uint8'),3)==1
    near=ndi.distance_transform_edt(~unknown,sampling=30)<=600
    valid=slope_valid&known&near
    if not valid.any():raise ValueError('No valid known shoreline slope support')
    fallback=float(np.median(slope_deg[valid]));slope_deg[~slope_valid]=fallback
    angle,buffer,index=tao.LakeParameter(unknown.astype('uint8'),dem,slope_deg,np.floor_divide(np.arange(100,601,100),30).astype(int))
    buffer_len=np.floor_divide(np.arange(100,601,100),30).astype(int)[index]
    contours=measure.find_contours(unknown.astype('uint8'),0)
    if not contours:raise ValueError('No initial shoreline')
    slope=np.zeros_like(z,dtype='float32')
    for contour in contours:tao.FirstSlopeCal(dem,0.,unknown,np.asarray(contour,dtype='int16'),slope,np.tan(np.radians(angle)),[30.,30.],buffer_len)
    if not np.isfinite(slope).all():raise ValueError('Invalid directional initialization')
    stretched,mean=tao.Data_stretching(slope_deg,slope,buffer,np.tan(np.radians(angle)))
    slope=stretched if mean<np.tan(np.radians(angle)) else tao.Data_stretching_1(slope)
    if not np.isfinite(slope).all():raise ValueError('Invalid slope stretching')
    info={'selected_buffer_pixels':int(buffer_len),'returned_buffer_index':int(index),'buffer_slope_deg':float(angle),
          'unknown_values_used':False,'valid_3x3_slope_count':int(valid.sum()),'edge_slope_imputation_deg':fallback,
          'nonzero_directional_slopes':int((slope>0).sum()),'directional_p05_p50_p95':np.quantile(slope[slope>0],[.05,.5,.95]).tolist(),
          'initialization':'released FirstSlopeCal and stretching; redacted unknowns; missing 3x3 derivative support imputed from known nearshore median'}
    return dem,known,slope,info

def recession(z,unknown,alpha,h0,label):
    start=time.time();dem,known,slope,info=initialize(z,unknown)
    origin=unknown.astype('uint8');water=origin.copy();h=h0;trace=[];deg=0
    for iteration in range(1,10001):
        contours=measure.find_contours(water,0)
        if not contours:break
        boundary=np.asarray(np.concatenate(contours),dtype='int64')
        if iteration>1:
            available=known[boundary[:,0],boundary[:,1]];boundary=boundary[available]
            if not len(boundary):break
            heights=dem[boundary[:,0],boundary[:,1]];high=heights>h
            if high.any():boundary=boundary[high]
            else:h=float(heights.min())
        points=kernel.frontier(boundary,water,known)
        if not len(points):break
        count,dd=kernel.advance(points,water,origin,known,dem,slope,float(alpha),30.)
        deg+=dd
        if iteration%25==0 or not water.any():
            rec={'iteration':iteration,'remaining':int(water.sum()),'elapsed_s':time.time()-start};trace.append(rec);print(label,rec,flush=True)
        if not water.any() or count==0:break
    remaining=int(water.sum());dem[water>0]=np.nan
    return dem,{'complete':remaining==0,'remaining_unknown':remaining,'iterations':iteration,'elapsed_s':time.time()-start,'degenerate_updates':deg,'initialization':info,'trace':trace}

def run_lake(key):
    folder=OUT/key;folder.mkdir(parents=True,exist_ok=True)
    with np.load(BASE/key/'frozen_terrain.npz') as f:z=f['dem'].astype(float);r=f['R'];o=f['O'];u=f['U'];w0=f['W0'];tr=Affine(*f['transform'][:6])
    rule=json.loads((BASE/key/'alpha_rule_frozen.json').read_text());alpha=rule['alpha'];h0=float(np.median(z[w0]))
    dist=ndi.distance_transform_edt(r,sampling=30.)
    tasks={'Tao_original_W0':(w0,np.zeros_like(o)),'Tao_R_baseline':(r,np.zeros_like(o)),'Tao_R_O_conditioned':(r&~o,o)}
    holdouts={}
    for frac in [.25,.5,.75]:
        cut=float(np.quantile(dist[o],1-frac));train=o&(dist<=cut-90);test=o&(dist>=cut)
        name=f'holdO_{int(frac*100)}';holdouts[name]={'cutoff_m':cut,'train_pixels':int(train.sum()),'test_pixels':int(test.sum()),'gap_m':90.,'eligible':bool(train.sum()>=100 and test.sum()>=100)}
        np.savez_compressed(folder/f'{name}_masks.npz',train=train,test=test)
        if holdouts[name]['eligible']:tasks[name]=(r&~train,train)
    protocol={'lake':key,'alpha':alpha,'alpha_source':rule,'H0_m':h0,'H0_assumption':'NASADEM original water median, provisional survey surface; not a measured 2017 stage',
      'sensitivity_offsets_m':[-5,-3,-1,0,1,3,5],'resolution_m':30,'crs':'EPSG:32644',
      'groups':'W0 baseline separate; R pair differs only in observed O; all outside target mask retains DEM',
      'G_definition':'R minus O_core minus common U; G and U unknown in conditioned group',
      'depth_policy':'raw H0-Z, no clipping for point error; nonpositive depths counted; integration clips at zero',
      'laang_units':'assumed metres for provisional numerical comparison only; source units still unconfirmed',
      'support_gate':'>=100 O cells and finite initialized slopes and all requested target pixels complete; holdout results reported without post-hoc performance exclusion',
      'holdouts':holdouts,'source_hashes':{f:sha(TAO/'lake_bathymetry'/f) for f in ['analysis.py','geometry_utils.py','core.py']},
      'terrain_sha256':sha(BASE/key/'frozen_terrain.npz'),'runner_sha256':sha(Path(__file__)),'kernel_sha256':sha(Path(kernel.__file__)),
      'baseline_label_note':'source-based numerical reproduction with disclosed runtime fixes and SWB mask override; not a claim to duplicate published maps exactly',
      'kernel_verification':kernel.verify()}
    savejson(folder/'protocol_before_predictions.json',protocol)
    _,patched=alpha_function();(folder/'calculate_alpha_enabled.txt').write_text(patched,encoding='utf-8')
    profile=dict(driver='GTiff',height=r.shape[0],width=r.shape[1],count=1,dtype='float32',crs='EPSG:32644',transform=tr,nodata=np.nan,compress='deflate')
    allstats={}
    for name,(unknown,observed) in tasks.items():
        path=folder/(name+'.tif');statpath=folder/(name+'.json')
        if path.exists() and statpath.exists():allstats[name]=json.loads(statpath.read_text());continue
        print('START',key,name,int(unknown.sum()),flush=True)
        inputz=z.copy();inputz[unknown]=np.nan
        assert np.isnan(inputz[unknown]).all()
        try:
            pred,stats=recession(inputz,unknown,alpha,h0,key+'/'+name)
            assert np.array_equal(pred[observed],z[observed])
            # Original W0 baseline retains original DEM outside W0 when evaluated over R.
            with rasterio.open(path,'w',**profile) as s:s.write(np.where(r|w0,pred,np.nan).astype('float32'),1)
            stats.update({'prediction_sha256':sha(path),'alpha':alpha,'H0':h0,'observed_preserved':True})
        except Exception as e:
            stats={'complete':False,'error':repr(e),'alpha':alpha,'H0':h0}
        savejson(statpath,stats);allstats[name]=stats
    savejson(folder/'predictions_finished.json',allstats)
    # No sonar/reference depth file is read anywhere in this runner.
    print('FINISHED',key,{k:v['complete'] for k,v in allstats.items()},flush=True)
if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--lake',choices=LAKES,required=True);args=parser.parse_args();run_lake(args.lake)
