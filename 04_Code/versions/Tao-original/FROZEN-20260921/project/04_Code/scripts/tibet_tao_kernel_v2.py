"""JIT implementation of released Tao ray/profile formulas; tested against source."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'03_Data_Processed/tibet_three_lakes_v1/compute_runtime'))
import numpy as np
from numba import njit

@njit(cache=True)
def line_at(r0,c0,r1,c1,idx):
    steep=abs(r1-r0)>abs(c1-c0)
    if steep:x0,y0,x1,y1=r0,c0,r1,c1
    else:x0,y0,x1,y1=c0,r0,c1,r1
    dx=abs(x1-x0);dy=abs(y1-y0);sx=1 if x1>=x0 else -1;sy=1 if y1>=y0 else -1
    err=dx;x=x0;y=y0
    for i in range(idx):
        x+=sx;err-=2*dy
        if err<=0:y+=sy;err+=2*dx
    if steep:return y,x  # x is row, y is column; return column,row
    return x,y

@njit(cache=True)
def geometry(slope,origin,rr,cc,vr,vc):
    r1=float(rr);c1=float(cc);r2=float(rr);c2=float(cc)
    while origin[int(r1),int(c1)]!=0:r1+=vr;c1+=vc
    while origin[int(r2),int(c2)]!=0:r2-=vr;c2-=vc
    p1r=int(r1+vr);p1c=int(c1+vc);p2r=int(r2-vr);p2c=int(c2-vc)
    s1=slope[p1r,p1c];s2=slope[p2r,p2c]
    length=max(abs(int(r1)-int(r2)),abs(int(c1)-int(c2)))+1
    if s1==0 and s2==0:weight=.5
    else:
        if s1>3*s2:s2=s1
        elif s2>3*s1:s1=s2
        weight=s1/(s1+s2)
    idx=max(1,int(weight*length))-1
    lowc,lowr=line_at(int(r2),int(c2),int(r1),int(c1),idx)
    return length,p1r,p1c,p2r,p2c,lowr,lowc

@njit(cache=True)
def directional(slope,origin,water,rr,cc,nr,nc,dem_h,alpha,res):
    step=np.hypot(rr-nr,cc-nc);vr=(rr-nr)/step;vc=(cc-nc)/step
    tr=float(rr);tc=float(cc)
    while water[int(tr),int(tc)]!=0:tr+=vr;tc+=vc
    tr-=vr;tc-=vc
    current=np.hypot(rr-int(tr),cc-int(tc))
    length,r1,c1,r2,c2,lr,lc=geometry(slope,origin,rr,cc,vr,vc)
    s1=slope[r1,c1];s2=slope[r2,c2]
    total=np.hypot(r1-r2,c1-c2);x=np.hypot(rr-lr,cc-lc);x1=np.hypot(lr-r1,lc-c1);x2=np.hypot(lr-r2,lc-c2)
    if x1==0 or x2==0 or total==0:return dem_h,float(length),1
    if s1>3*s2:s1*=.5;s2=s1
    elif s2>3*s1:s2*=.5;s1=s2
    if min(r1,lr)<=rr<=max(r1,lr) and min(c1,lc)<=cc<=max(c1,lc):s=s1;side=x1
    else:s=s2;side=x2
    slope_now=(s/side)*x*(current/total)**alpha
    return dem_h-slope_now*res*step,float(length),0

@njit(cache=True)
def advance(points,water,origin,known,dem,slope,alpha,res):
    offsets=np.array([[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[-1,1],[1,-1],[1,1]])
    degenerate=0;updated=0
    for i in range(len(points)):
        rr=points[i,0];cc=points[i,1]
        if water[rr,cc]==0:continue
        numerator=0.;denominator=0.
        for j in range(8):
            nr=rr+offsets[j,0];nc=cc+offsets[j,1]
            if not known[nr,nc]:continue
            value,length,deg=directional(slope,origin,water,rr,cc,nr,nc,dem[nr,nc],alpha,res)
            if not np.isfinite(value) or length<=0:raise ValueError('Nonfinite ray update')
            numerator+=value/length;denominator+=1/length;degenerate+=deg
        if denominator>0:dem[rr,cc]=numerator/denominator;known[rr,cc]=True;water[rr,cc]=0;updated+=1
    return updated,degenerate

@njit(cache=True)
def frontier(contours,water,known):
    seen=np.zeros_like(water);out=np.empty((len(contours)*8,2),np.int64);n=0
    offsets=np.array([[0,-1],[0,1],[-1,0],[1,0],[-1,-1],[-1,1],[1,-1],[1,1]])
    for i in range(len(contours)):
        r,c=contours[i]
        if not known[r,c]:continue
        for j in range(8):
            rr=r+offsets[j,0];cc=c+offsets[j,1]
            if water[rr,cc] and not seen[rr,cc]:out[n,0]=rr;out[n,1]=cc;n+=1;seen[rr,cc]=1
    return out[:n]

def verify():
    from skimage.draw import line
    from mead_full2d_alpha_masking_v1 import alpha_function
    from tao_compat import TotalDistance,draw_extended_line
    calculate,_=alpha_function();rng=np.random.default_rng(916)
    for i in range(1000):
        pts=rng.integers(3,70,4);r0,c0,r1,c1=pts;rs,cs=line(r0,c0,r1,c1)
        for idx in [0,len(rs)//2,len(rs)-1]:
            c,r=line_at(r0,c0,r1,c1,idx)
            assert (r,c)==(rs[idx],cs[idx]),(pts,idx,(r,c),(rs[idx],cs[idx]))
    origin=np.zeros((80,90),np.uint8);origin[10:70,10:80]=1
    slope=rng.uniform(.001,.4,origin.shape)
    for i in range(500):
        pt=rng.integers(15,65,2);dr,dc=[(1,0),(0,1),(1,1),(1,-1),(-1,1),(-1,-1)][i%6];nb=pt-[dr,dc]
        cur,v=draw_extended_line(origin,nb,pt);length,p1,p2,low=TotalDistance(slope,origin,pt,v)
        g=geometry(slope,origin,*pt,*v)
        assert tuple(g)==(length,*p1,*p2,*low),(g,(length,p1,p2,low))
        for alpha in [0.,1.,2.]:
            expected=calculate(slope,pt,4500.,30*np.hypot(dr,dc),p1,p2,low,0.,cur,alpha)
            actual,_,deg=directional(slope,origin,origin,*pt,*nb,4500.,alpha,30.)
            assert abs(actual-expected)<1e-8,(actual,expected)
    return {'line_parity_cases':1000,'ray_profile_parity_cases':1500,'passed':True,'intentional_driver_fixes':['float inverse-distance weights','physical diagonal step','zero-length half-profile returns parent height and is counted']}
if __name__=='__main__':print(verify())
