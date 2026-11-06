"""Corrected M1-M3 Hybrid block diagram in the SIVP manuscript."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "figures" / "extended"
OUT.mkdir(parents=True, exist_ok=True)
fig, ax=plt.subplots(figsize=(7.35,2.9)); ax.set_xlim(0,12.1); ax.set_ylim(0,5.0); ax.axis('off')
blocks=[(0.2,2.35,1.5,1.0,'VV / VH\ncomplex CPI'),(2.2,2.35,2.0,1.0,'M1: Complex CNN\n+ LSTM / W₀'),(4.82,2.35,1.95,1.0,'M2: Reference-\nanchored cICA'),(7.38,2.35,2.0,1.0,'M3: NMF bank\n+ covariance'),(10.0,2.35,1.9,1.0,'Score t; detect\nif t > threshold')]
for x,y,w,h,label in blocks:
 fc='#ecf4f9' if x<7 else '#f0f2f4'
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.05,rounding_size=0.12',facecolor=fc,edgecolor='#234c65',linewidth=1.15)
 ax.add_patch(p); ax.text(x+w/2,y+h/2,label,fontsize=8.4,ha='center',va='center',color='#1c3543')
for i in range(len(blocks)-1):
 x1=blocks[i][0]+blocks[i][2]+0.08;x2=blocks[i+1][0]-0.12
 ax.add_patch(FancyArrowPatch((x1,2.85),(x2,2.85),arrowstyle='-|>', mutation_scale=12,lw=1.2,color='#234c65'))
ax.text(3.18,3.75,'Learned weights and warm-start',fontsize=8,ha='center',color='#396478')
ax.text(5.78,3.75,'Per-window ICA refinement',fontsize=8,ha='center',color='#396478')
ax.text(8.4,3.75,'Separated-domain statistic',fontsize=8,ha='center',color='#396478')
# training branch
ax.add_patch(FancyBboxPatch((0.75,.55),4.3,.9,boxstyle='round,pad=0.07,rounding_size=0.13',facecolor='#fcf7eb',edgecolor='#bb873a',linewidth=1))
ax.text(2.9,1.0,'TRAIN: synthetic mixtures / loss\n(physics term evaluated as ablation)',fontsize=8.2,ha='center',va='center')
ax.add_patch(FancyArrowPatch((2.9,1.5),(2.9,2.28),arrowstyle='-|>',mutation_scale=12,lw=1,linestyle='--',color='#ae812e'))
# calibration branch
ax.add_patch(FancyBboxPatch((6.5,.55),5.2,.9,boxstyle='round,pad=0.07,rounding_size=0.13',facecolor='#eff6ed',edgecolor='#568157',linewidth=1))
ax.text(9.1,1.0,'CAL: clutter-only pool  →  C_sep and threshold\nTEST-H0: independently assessed realised Pfa',fontsize=8.2,ha='center',va='center')
ax.add_patch(FancyArrowPatch((9.1,1.5),(9.1,2.27),arrowstyle='-|>',mutation_scale=12,lw=1,linestyle='--',color='#568157'))
ax.text(6.15,4.7,'Network initialization is not a detector decision',fontsize=8.8,ha='center',color='#35444e')
fig.savefig(OUT/'fig0_hybrid_architecture.pdf',bbox_inches='tight',pad_inches=.06)
fig.savefig(OUT/'fig0_hybrid_architecture.png',bbox_inches='tight',pad_inches=.06,dpi=220)
plt.close(fig)
