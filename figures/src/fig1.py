from pl import *
F=Fig(7.16,3.12); S=np.sin
fl=lambda t:0.6*S(t*6.28*7)+0.25*S(t*6.28*14)
fs_=lambda t:max(0,S(t*6.28*7))*1.2-0.5
fw=lambda t:0.5*S(t*6.28*2.3)+0.3*S(t*6.28*5.1+1)
T=7.5; H=8.5
def head(x,n,t): F.badge(x+0.07,0.16,n); F.text(x+0.18,0.19,t,fs=H,bold=True,ha='left',va='baseline')
head(0.05,1,'Input window')
rows=[('Load','load',fl,0.34),('Solar','solar',fs_,0.92),('Wind','wind',fw,1.50)]
for nm,c,f,y in rows:
    F.box(0.05,y,0.98,0.52,c=c); F.text(0.10,y+0.10,nm,fs=T,bold=True,ha='left'); F.series(0.40,y+0.05,0.58,0.36,f,C[c][1]); F.text(0.10,y+0.40,'1×168',fs=7,color='#333',ha='left')
F.box(0.05,2.18,0.98,0.40,'Weather\n$E$×168',c='ex',fs=T); F.box(0.05,2.66,0.98,0.40,'Calendar\n$M$×168',c='ex',fs=T)
head(1.15,2,'Normalise, split')
F.box(1.15,0.34,0.30,2.72,c='grey'); F.text(1.30,1.70,'RevIN',fs=8.5,rot=90,bold=True)
for y in [0.60,1.18,1.76,2.38,2.86]: F.arrow([(1.03,y),(1.15,y)],ms=5)
F.arrow([(1.45,1.18),(1.60,1.18)],ms=5); F.arrow([(1.45,2.62),(1.60,2.62)],ms=5)
F.box(1.60,0.34,1.08,1.84,c='white',ec='#555')
F.text(2.14,0.48,'Trend + remainder',fs=T,bold=True); F.text(2.14,0.63,'25-h moving average',fs=7,color='#333')
f0=lambda t:0.55*S(t*6.28*3)+0.25*S(t*6.28*16)
F.series(1.66,0.72,0.96,0.26,f0,C['load'][1],lw=0.7)
F.series(1.66,1.04,0.44,0.22,lambda t:0.55*S(t*6.28*1.5),C['load'][1],lw=0.7); F.series(2.20,1.04,0.42,0.22,lambda t:0.25*S(t*6.28*8),C['gate'][1],lw=0.7)
F.text(1.88,1.32,'trend',fs=7,color='#333'); F.text(2.41,1.32,'remainder',fs=7,color='#333')
for i in range(8): F.a.plot([1.66+i*0.96/7]*2,[F.Y(1.45),F.Y(1.70)],c='#888',lw=0.4,ls=(0,(2,2)),zorder=3)
F.series(1.66,1.45,0.96,0.25,f0,C['load'][1],lw=0.6)
F.text(2.14,1.84,'7 patches of 24 h',fs=T,bold=True); F.text(2.14,2.02,'Linear 24$\\rightarrow$128',fs=7,color='#333')
F.box(1.60,2.30,1.08,0.76,c='ex'); F.text(2.14,2.46,'Token per channel',fs=T,bold=True); F.text(2.14,2.68,'Linear 168$\\rightarrow$128\n(no patching)',fs=7,color='#333')
head(2.82,3,'Tokens ($d$=128)')
tx=2.82; sz=0.115; gp=0.02
trow=[('Load','load',0.46),('Solar','solar',1.04),('Wind','wind',1.62)]
for nm,c,y in trow:
    F.text(tx,y,nm,fs=T,bold=True,ha='left')
    for i in range(8): F.sq(tx+i*(sz+gp),y+0.08,sz,c if i<7 else 'glob')
    F.arrow([(2.68,y+0.14),(2.80,y+0.14)],ms=5)
F.text(tx,0.46+0.08+sz+0.11,'7 patches + 1 global token',fs=7,color='#333',ha='left')
F.text(tx,2.36,'Weather, calendar tokens',fs=T,bold=True,color=C['ex'][1],ha='left')
for i in range(6): F.sq(tx+i*(sz+gp),2.46,sz,'ex')
F.text(tx+6*(sz+gp)+0.03,2.53,'…',fs=8,ha='left')
F.text(tx,2.74,'$E+M$ tokens, reused in\nboth blocks',fs=7,color='#333',ha='left')
F.arrow([(2.68,2.62),(2.80,2.62)],ms=5)
ex0=4.08; ew=1.50
head(ex0,4,'Encoder block ×2')
F.box(ex0,0.34,ew,2.72,c='white',dash=True,ec='#666',lw=0.6,fill=False)
bx=ex0+0.08; bw=ew-0.16
F.box(bx,0.46,bw,0.58,'Temporal\nself-attention',c='load',fs=T)
F.box(bx,1.22,bw,1.00,c='gate'); F.text(bx+bw/2,1.52,'CrossMix',fs=8.5,bold=True); F.text(bx+bw/2,1.76,'other sources\n+ weather, gated',fs=7)
F.box(bx,2.40,bw,0.54,'Feed-forward',c='white',fs=T,ec='#555')
F.arrow([(bx+bw/2,1.04),(bx+bw/2,1.22)],ms=5); F.arrow([(bx+bw/2,2.22),(bx+bw/2,2.40)],ms=5)
for nm,c,y in trow: F.a.plot([tx+8*(sz+gp)-0.01,4.00],[F.Y(y+0.14)]*2,c='#222',lw=0.7,zorder=3)
F.a.plot([4.00,4.00],[F.Y(0.60),F.Y(1.76)],c='#222',lw=0.7,zorder=3)
F.arrow([(4.00,0.75),(bx,0.75)],ms=5)
F.arrow([(3.74,2.53),(4.00,2.53),(4.00,2.08),(bx,2.08)],c=C['ex'][1],lw=0.8,ms=5)
hx=5.78; hw=1.33
head(hx,5,'Output head')
F.box(hx,0.34,hw,0.50,'Flatten\n8×128 = 1024',c='white',fs=T,ec='#555')
F.box(hx,1.00,hw,0.56,'Source-specific\nlinear 1024$\\rightarrow$24',c='white',fs=T,ec='#555')
F.box(hx,1.72,hw,0.40,'Inverse RevIN',c='white',fs=T,ec='#555')
F.arrow([(hx+hw/2,0.84),(hx+hw/2,1.00)],ms=5); F.arrow([(hx+hw/2,1.56),(hx+hw/2,1.72)],ms=5)
F.arrow([(bx+bw,2.67),(5.68,2.67),(5.68,0.59),(hx,0.59)],ms=5)
F.a.plot([hx+hw/2]*2,[F.Y(2.12),F.Y(2.24)],c='#222',lw=0.7); F.a.plot([hx+0.21,hx+hw-0.21],[F.Y(2.24)]*2,c='#222',lw=0.7)
ow=0.42
for i,(nm,c,f) in enumerate([('Load','load',lambda t:0.6*S(t*6.28)+0.1),('Solar','solar',lambda t:max(0,S(t*6.28+1.5))*1.1-0.5),('Wind','wind',lambda t:0.4*S(t*6.28*0.7+1))]):
    x=hx+i*(ow+0.035); F.arrow([(x+ow/2,2.24),(x+ow/2,2.36)],lw=0.7,ms=4)
    F.box(x,2.36,ow,0.70,c=c,r=0.03); F.text(x+ow/2,2.48,nm,fs=7,bold=True); F.series(x+0.04,2.58,ow-0.08,0.26,f,C[c][1],lw=0.7); F.text(x+ow/2,2.94,'24 h',fs=7,color='#333')
F.save('Fig1_GateXer_overview')
