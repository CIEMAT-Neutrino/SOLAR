"""
check_fiducial_z_margin.py — what does the FiducialZ margin do in the scan, and is it applied downstream?
======================================================================================================
01_fiducialize.py scans FiducialZ for every folder, but lib/fiducial.py: build_fiducial_spatial_mask applies it only for
folder == "Nominal" (Truncated and Reduced reject the Z endcaps with the SignalParticleSurface < 3 filter instead).
This tool measures the consequence for the stored Truncated volumes (config/analysis/fiducial/truncated/BestFiducials.json):

  part 1  smoothed fiducial-scan significance (the metric 02_best_fiducial.py maximises, same MC gate and smoothing) at the stored
          (X, Y, Z), at the same (X, Y) with Z = 0, and at the best volume that has Z = 0
  part 2  weighted fraction of signal and of each background retained with and without the Z margin (truth_position caches,
          10-20 MeV, quality mask, reco position)

Read-only. Run under the container:  python3 src/tools/check_fiducial_z_margin.py
"""
# ---- part 1 -----------------------------------------------------------------------------
import sys, os, json
from pathlib import Path
ROOT=str(Path(__file__).resolve().parents[2])
sys.path.insert(0, ROOT)
src=open(f"{ROOT}/src/physics/signal/02_best_fiducial.py").read()
head=src[:src.index("parser = argparse.ArgumentParser")]
ns={"__file__": f"{ROOT}/src/physics/signal/02_best_fiducial.py", "__name__":"bf"}
exec(compile(head, "02_best_fiducial_head", "exec"), ns)
np=ns["np"]; pd=ns["pd"]; root=ns["root"]; _ai=ns["_analysis_info"]
folder="truncated"; energy="SolarEnergy"; exposure=100.0; mc=float(ns["get_analysis_threshold"](str(root),"FIDUCIALIZATION",stage="MC",fallback=0.0))
best=json.load(open(f"{ROOT}/config/analysis/fiducial/{folder}/BestFiducials.json"))
rows=[]
for config in ("hd_1x2x6_centralAPA","hd_1x2x6_lateralAPA","vd_1x8x14_3view_30deg_nominal","vd_1x8x14_3view_30deg_shielded"):
    dfs=[pd.read_pickle(f"{_ai['PATH']}/FIDUCIAL/{folder}/{config}/marley/{config}_marley_{energy}_Fiducial_Scan.pkl")]
    for b in ns["get_background_samples"](str(root)):
        fp=f"{_ai['PATH']}/FIDUCIAL/{folder}/{config}/{b}/{config}_{b}_{energy}_Fiducial_Scan.pkl"
        if os.path.exists(fp): dfs.append(pd.read_pickle(fp))
    plot_df=ns["explode"](pd.concat(dfs,ignore_index=True),["Counts","Error+","Error-","Energy","MCCounts"],debug=False).copy()
    for c in ("Counts","Error+","Error-","MCCounts"): plot_df[c]=pd.to_numeric(plot_df[c],errors="coerce").fillna(0.0)
    plot_df["Energy"]=pd.to_numeric(plot_df["Energy"],errors="coerce")
    for an in ("DAYNIGHT","HEP","SENSITIVITY"):
        acfg=ns["get_fiducialization_config"](str(root),an)
        ref=str(_ai.get("BEST_SIGMA_SIGNIFICANCE_REFERENCE",{}).get(an,"")).lower()
        if ref in {"asimov","gaussian"}:
            acfg=dict(acfg); acfg["significance_type"]=ref
        sm=ns["get_smoothing_config"](str(root),analysis_name=an,dimensions="1d",stage="fiducial")
        gated=ns["apply_fiducial_mc_threshold"](plot_df,acfg,mc,str(root))
        _, sd=ns["select_best_fiducial"](gated,acfg,sm,exposure)
        if sd.empty: continue
        g=sd.loc[sd.SmoothedSignificance.idxmax()]
        st=best[config][an][energy]
        key=lambda x,y,z: sd[(sd.FiducializedX==x)&(sd.FiducializedY==y)&(sd.FiducializedZ==z)]
        at=key(st["FiducialX"],st["FiducialY"],st["FiducialZ"]); at0=key(st["FiducialX"],st["FiducialY"],0)
        z0=sd[sd.FiducializedZ==0]; b0=z0.loc[z0.SmoothedSignificance.idxmax()]
        nofid=key(0,0,0)
        f=lambda d: float(d.SmoothedSignificance.iloc[0]) if len(d) else float("nan")
        rows.append((config[:14],an,(int(g.FiducializedX),int(g.FiducializedY),int(g.FiducializedZ)),float(g.SmoothedSignificance),f(at),(st["FiducialX"],st["FiducialY"],st["FiducialZ"]),f(at0),(int(b0.FiducializedX),int(b0.FiducializedY),0),float(b0.SmoothedSignificance),f(nofid)))
print(f"MC gate {mc}, exposure {exposure}")
print(f"{'config':15s}{'analysis':12s}{'scan optimum (X,Y,Z)':>24s}{'sig':>7s} | {'stored':>16s}{'sig@stored':>11s} | {'same X,Y with Z=0':>18s}{'sig':>7s} | {'best with Z=0':>16s}{'sig':>7s} | {'no fid':>7s}")
for r in rows:
    print(f"{r[0]:15s}{r[1]:12s}{str(r[2]):>24s}{r[3]:7.2f} | {str(r[5]):>16s}{r[4]:11.2f} | {'':>18s}{r[6]:7.2f} | {str(r[7]):>16s}{r[8]:7.2f} | {r[9]:7.2f}")


# ---- part 2 -----------------------------------------------------------------------------
import json, numpy as np, sys
from lib import root
from lib.fiducial import accepted_flash_planes
from lib.background import is_surface_background
B=json.load(open("config/analysis/fiducial/truncated/BestFiducials.json"))
def q(d,name):
    m=accepted_flash_planes(d["plane"],str(root),True)&(d["pe"]>0)
    if is_surface_background(str(root),name): m&=(d["surface"]>=0)&(d["surface"]<3)
    return m
print("weighted fraction retained by the stored volume WITH Z vs the same X,Y with Z=0 (10-20 MeV, quality mask, reco position)")
for c in ("hd_1x2x6_centralAPA","hd_1x2x6_lateralAPA","vd_1x8x14_3view_30deg_nominal","vd_1x8x14_3view_30deg_shielded"):
    info=json.load(open(f"config/{c}/{c}_config.json")); LX=info["DETECTOR_SIZE_X"]; LY=info["DETECTOR_SIZE_Y"]; LZ=info["DETECTOR_SIZE_Z"]
    for an in ("DAYNIGHT","SENSITIVITY"):
        st=B[c][an]["SolarEnergy"]; fx,fy,fz=st["FiducialX"],st["FiducialY"],st["FiducialZ"]
        out=[]
        for n in ("marley","gamma","neutron","radiological"):
            d=np.load(f"output/data/solar/truth_position/{c}/{c}_{n}.npz"); m=q(d,n)&(d["energy"]>=10)&(d["energy"]<=20)
            x,y,z=d["reco_x"],d["reco_y"],d["reco_z"]
            mx=(np.abs(x)>fx) if c.startswith("hd_1x2x6_lateral") else ((np.abs(x)<LX/2-fx) if "central" in c else (x<LX/2-fx))
            mxy=mx&(np.abs(y)<LY/2-fy); mz=mxy&(z>fz)&(z<LZ-fz)
            w=d["w"]; t=w[m].sum()
            out.append(f"{n[:4]} {100*w[m&mz].sum()/t:5.1f}% (XY only {100*w[m&mxy].sum()/t:5.1f}%, N_MC {int((m&mz).sum())}/{int((m&mxy).sum())})")
        print(f"{c[:14]:15s}{an:12s}({fx},{fy},{fz}) "+" | ".join(out))
