"""Input-only TWDB screen for observed terrain inside historical DEM water."""
from pathlib import Path
import csv,json,math
import numpy as np
from scipy import ndimage as ndi
from pyproj import Transformer
import shapely
from shapely.ops import transform as geom_transform
from pyogrio.raw import read
import rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.windows import Window
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'05_Results/codex/by_date/2026-09-17/twdb_nasadem_intersection_screen'
TABLE=ROOT/'03_Data_Processed/ground_truth/twdb_v4_clean/twdb_entity_to_hydrolakes_v4_clean.csv'
HGT=ROOT/'02_Data_Raw/NASADEM_Extracted/HGT'
SWB=ROOT/'02_Data_Raw/NASADEM_Extracted/SWB'
SWO=ROOT/'02_Data_Raw/SWO_GSW_v1_5_2024/occurrence_v1_5_2024'

def component(mask,anchor):
 labels,n=ndi.label(mask,np.ones((3,3)))
 counts=np.bincount(labels[anchor].ravel(),minlength=n+1);counts[0]=0
 if counts.max()==0:raise ValueError('no_component_overlapping_identity_polygon')
 return labels==np.argmax(counts)

def sample(lon,lat):
 z=np.full(lon.shape,np.nan,'float32');swb=np.full(lon.shape,127,'uint8');occ=np.full(lon.shape,255,'uint8');sources=[]
 for la in np.unique(np.floor(lat).astype(int)):
  for lo in np.unique(np.floor(lon).astype(int)):
   sel=(np.floor(lat)==la)&(np.floor(lon)==lo)
   if not sel.any():continue
   stem=f'{"n" if la>=0 else "s"}{abs(la):02d}{"e" if lo>=0 else "w"}{abs(lo):03d}'
   hp=HGT/(stem+'.hgt');sp=SWB/(stem+'.swb')
   if not hp.exists() or not sp.exists():raise FileNotFoundError(f'missing_NASADEM_{stem}')
   h=np.memmap(hp,dtype='>i2',mode='r',shape=(3601,3601));s=np.memmap(sp,dtype='u1',mode='r',shape=(3601,3601))
   rr=np.rint((la+1-lat[sel])*3600).astype(int);cc=np.rint((lon[sel]-lo)*3600).astype(int)
   z[sel]=h[rr,cc];swb[sel]=s[rr,cc];sources.extend([str(hp),str(sp)])
 z[z==-32768]=np.nan
 for la in np.unique((np.ceil(lat/10)*10).astype(int)):
  for lo in np.unique((np.floor(lon/10)*10).astype(int)):
   sel=(np.ceil(lat/10)*10==la)&(np.floor(lon/10)*10==lo)
   if not sel.any():continue
   name=f'occurrence_{abs(lo)}{"E" if lo>=0 else "W"}_{abs(la)}{"N" if la>=0 else "S"}_v1_5_2024.tif';path=SWO/name
   with rasterio.open(path) as src:
    rr=np.floor((lat[sel]-src.transform.f)/src.transform.e).astype(int);cc=np.floor((lon[sel]-src.transform.c)/src.transform.a).astype(int)
    win=Window(int(cc.min()),int(rr.min()),int(cc.max()-cc.min()+1),int(rr.max()-rr.min()+1))
    a=src.read(1,window=win);occ[sel]=a[rr-rr.min(),cc-cc.min()]
   sources.append(str(path))
 return z,swb,occ,sources

def main():
 OUT.mkdir(parents=True,exist_ok=True)
 rows=list(csv.DictReader(TABLE.open(encoding='utf-8-sig')))
 protocol={'cohort':'all 114 local TWDB v4 entities; process 77 mapping_publishable=True; retain 37 identity exclusions',
 'no_reference_selection':'capacity/elevation/depth values are not used in screening; existing identity matches reused with provenance',
 'R':'SWO>=5 component with maximum overlap with HydroLAKES identity polygon, clipped to polygon buffer 3km; clipping flagged',
 'W0':'SWB=255 finite DEM connected component with maximum identity-polygon overlap',
 'candidate':'R & W0 & 5<=SWO<95',
 'decision':'exactly constant candidate = no terrain information; nonconstant candidate = unresolved, not automatically eligible',
 'alpha_versions':['O-slope fixed 1/2 rule','O-only spatial masking selection, candidates 0/0.5/1/1.5/2'],
 'model_gate':'must establish reliable exposed terrain, sufficient spatial support and reference readiness before models; no forcing water-surface values into O'}
 (OUT/'protocol_before_screen.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
 selected=[r for r in rows if r['mapping_publishable']=='True']
 ids=sorted({int(r['hylak_id']) for r in selected})
 meta,fids,geoms,fields=read(ROOT/'02_Data_Raw/input_mead/HydroLAKES_polys_v10_wgs84.gpkg',where='Hylak_id IN ('+','.join(map(str,ids))+')',columns=['Hylak_id'])
 lookup={int(i):shapely.from_wkb(g) for i,g in zip(fields[0],geoms)}
 reports=[]
 for row in rows:
  report={'entity_id':row['entity_id'],'name':row['entity_name'],'hylak_id':row['hylak_id'],'mapping_status':row['mapping_status'],'reference_available':'TWDB EAC curve in local manifest; spatial soundings not yet certified'}
  if row['mapping_publishable']!='True':
   report['status']='identity_not_publishable';reports.append(report);continue
  try:
   g=lookup[int(row['hylak_id'])];center=g.centroid;zone=int((center.x+180)//6)+1;crs=f'EPSG:{32600+zone}'
   fwd=Transformer.from_crs(4326,crs,always_xy=True);inv=Transformer.from_crs(crs,4326,always_xy=True);gp=geom_transform(fwd.transform,g);clip=gp.buffer(3000)
   left,bottom,right,top=clip.bounds;left=math.floor(left/30)*30;top=math.ceil(top/30)*30
   width=math.ceil((right-left)/30)+2;height=math.ceil((top-bottom)/30)+2;tr=from_origin(left,top,30,30)
   rr,cc=np.indices((height,width));lon,lat=inv.transform(left+(cc+.5)*30,top-(rr+.5)*30)
   z,s,o,sources=sample(lon,lat)
   anchor=rasterize([(gp,1)],out_shape=z.shape,transform=tr).astype(bool);extent=rasterize([(clip,1)],out_shape=z.shape,transform=tr).astype(bool)
   w=component((s==255)&np.isfinite(z),anchor);r=component((o>=5)&(o<=100),anchor)&extent;c=r&w&(o<95)
   report.update({'crs':crs,'R_area_km2':float(r.sum()*.0009),'W0_area_km2':float(w.sum()*.0009),'candidate_cells':int(c.sum()),'candidate_area_km2':float(c.sum()*.0009),
     'R_touches_3km_clip':bool((r&~ndi.binary_erosion(extent)).any()),'sources':sources})
   if not c.any():report['status']='no_intersection'
   else:
    vals,counts=np.unique(z[c],return_counts=True);p=z[c]
    report.update({'min_m':float(p.min()),'max_m':float(p.max()),'p05_p50_p95_m':np.quantile(p,[.05,.5,.95]).tolist(),'unique_heights':len(vals),'mode_m':float(vals[np.argmax(counts)]),'mode_fraction':float(counts.max()/len(p))})
    report['status']='constant_water_surface_no_O' if len(vals)==1 else 'nonconstant_requires_source_validation'
    if len(vals)>1:
     np.savez_compressed(OUT/(row['entity_id']+'_diagnostic.npz'),dem=z,swb=s,occurrence=o,R=r,W0=w,candidate=c,transform=np.array(tr))
  except Exception as exc:report.update(status='screening_error_or_missing_data',error=str(exc))
  reports.append(report);print(report['name'],report['status'],report.get('candidate_cells'),report.get('unique_heights'),flush=True)
  (OUT/'screening.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')
 (OUT/'screening.json').write_text(json.dumps(reports,indent=2,ensure_ascii=False),encoding='utf-8')
 print('DONE',len(reports),flush=True)

if __name__=='__main__':main()
