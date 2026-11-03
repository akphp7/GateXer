from pl import *
F=Fig(6.62,3.20); T=7.5
F.box(0.05,1.20,0.62,0.62,'$x$\n3×8×128',c='white',fs=T,ec='#555')
F.box(0.80,1.36,0.56,0.30,'LayerNorm',c='grey',fs=7)
F.arrow([(0.67,1.51),(0.80,1.51)],ms=5)
cx=1.52
kxc=4.62+1.95/2
F.a.plot([1.36,cx],[F.Y(1.51)]*2,c='#222',lw=0.8); F.a.plot([cx,cx],[F.Y(0.62),F.Y(2.62)],c='#222',lw=0.8)
bx=1.68; bw=2.40
for y in (0.62,1.55,2.62): F.arrow([(cx,y),(bx,y)],ms=5)
# residual
F.a.plot([0.36,0.36,kxc],[F.Y(1.20),F.Y(0.18),F.Y(0.18)],c='#777',lw=0.7,ls=(0,(3,2)),zorder=2)
F.arrow([(kxc,0.18),(kxc,0.98)],c='#777',lw=0.7,dash=True,ms=5)
F.text(2.70,0.10,'residual: $x$ is passed on unchanged',fs=7,color='#444')
# (a)
F.box(bx,0.28,bw,0.92,c='white',ec=C['load'][1],lw=0.8)
F.badge(bx+0.12,0.43,'a'); F.text(bx+0.26,0.45,'Source mixing',fs=8,bold=True,ha='left',va='baseline')
gx,gy,g=bx+0.14,0.62,0.10
for r,c in enumerate(['load','solar','wind']):
    for i in range(8): F.sq(gx+i*(g+0.014),gy+r*(g+0.035),g,c,lw=0.6 if i!=3 else 1.3,ec='#D55E00' if i==3 else None)
F.arrow([(gx+3*(g+0.014)+0.03,gy+(g+0.035)+g),(gx+3*(g+0.014)+0.03,gy+g+0.01)],c='#D55E00',lw=0.7,ms=4)
F.arrow([(gx+3*(g+0.014)+0.07,gy+2*(g+0.035)+g),(gx+3*(g+0.014)+0.07,gy+g+0.01)],c='#D55E00',lw=0.7,ms=4)
F.text(bx+1.25,0.82,'each source attends to\nthe other two, at the\nsame token position',fs=7,ha='left')
# (b)
F.box(bx,1.30,bw,0.86,c='white',ec=C['ex'][1],lw=0.8)
F.badge(bx+0.12,1.45,'b'); F.text(bx+0.26,1.47,'Weather mixing',fs=8,bold=True,ha='left',va='baseline')
for i in range(8): F.sq(bx+0.14+i*0.115,1.64,0.10,'load')
for i in range(5): F.sq(bx+1.36+i*0.115,1.64,0.10,'ex')
F.a.plot([bx+0.5,bx+0.5,bx+1.70,bx+1.70],[F.Y(1.76),F.Y(1.85),F.Y(1.85),F.Y(1.76)],c=C['ex'][1],lw=0.7,zorder=3)
F.arrow([(bx+1.70,1.85),(bx+1.70,1.77)],c=C['ex'][1],lw=0.7,ms=4)
F.text(bx+0.14,2.02,'source tokens query weather and calendar tokens',fs=7,ha='left')
# (c)
F.box(bx,2.30,bw,0.82,c='white',ec=C['gate'][1],lw=0.8)
F.badge(bx+0.12,2.45,'c'); F.text(bx+0.26,2.47,'Router',fs=8,bold=True,ha='left',va='baseline'); F.text(bx+0.80,2.47,'(per token): Linear 128$\\rightarrow$3, softmax',fs=7,ha='left',va='baseline')
for i,(t,col) in enumerate([('$g_0$  none','#9E9E9E'),('$g_1$  other sources',C['load'][1]),('$g_2$  weather',C['ex'][1])]):
    y=2.58+i*0.17; F.text(bx+0.14,y+0.05,t,fs=7,ha='left'); F.a.add_patch(Rectangle((bx+1.12,F.Y(y+0.10)),0.50,0.10,fc=col,ec='none',zorder=3))
F.text(bx+1.72,2.82,'start: all $=\\frac{1}{3}$',fs=7,ha='left',color='#333')
# combine
kx=4.62; kw=1.95
F.box(kx,0.98,kw,1.52,c='gate')
F.text(kx+kw/2,1.14,'Gated sum',fs=8.5,bold=True)
F.text(kx+kw/2,1.46,'$y=x$',fs=8)
F.text(kx+kw/2,1.68,'$+\\,\\tanh(\\alpha_{src})\\,g_1\\,o_{src}$',fs=8)
F.text(kx+kw/2,1.90,'$+\\,\\tanh(\\alpha_{ex})\\,g_2\\,o_{ex}$',fs=8)
F.text(kx+kw/2,2.24,'$\\alpha$ starts at 0, so $y=x$\nat initialisation',fs=7,color='#222')
F.arrow([(bx+bw,0.74),(4.40,0.74),(4.40,1.68),(kx,1.68)],ms=5); F.text(4.26,0.62,'$o_{src}$',fs=7)
F.arrow([(bx+bw,1.75),(4.40,1.75),(4.40,1.90),(kx,1.90)],ms=5); F.text(4.26,1.60,'$o_{ex}$',fs=7)
F.arrow([(bx+bw,2.80),(4.47,2.80),(4.47,2.35),(kx,2.35)],ms=5); F.text(4.29,2.93,'$g_1,\\,g_2$',fs=7)
F.arrow([(kx+kw/2,2.50),(kx+kw/2,2.70)],ms=5)
F.box(kx+0.20,2.70,kw-0.40,0.40,'$y$  $\\rightarrow$ feed-forward',c='white',fs=7.5,ec='#555')
F.save('Fig2_CrossMix_zoom')
