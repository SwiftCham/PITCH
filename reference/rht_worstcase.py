"""Worst-case inputs for the randomised Hadamard transform vs. Theorem 1 (reviewer item 17)."""
import numpy as np, math
import pitch_reference as R
rng=np.random.default_rng(0)
BOUND=lambda b: math.sqrt(3)*math.pi/2*4.0**-b

def rel(X,Xh): return float((((X-Xh)**2).sum(1)/(X**2).sum(1)).mean())

def tq(X,b,seed=R.DEFAULT_SEED):
    c,n=R.turbo_encode(X,b,seed); return R.turbo_decode(c,n.astype(np.float32).astype(np.float64),b,seed)

def tq_haar(X,b,Q):
    d=X.shape[1]; n=np.linalg.norm(X,axis=1,keepdims=True); U=(X/n)@Q.T
    cb=R.turbo_codebook(d,b).astype(np.float32); e=(np.float32(.5)*(cb[1:]+cb[:-1])).astype(float)
    return (R.turbo_codebook(d,b)[np.searchsorted(e,U)] @ Q)*n

def haar(d):
    A=rng.standard_normal((d,d)); Q,Rr=np.linalg.qr(A); return Q*np.sign(np.diag(Rr))

def tq_double(X,b,s1=R.DEFAULT_SEED,s2=0x12345678):
    # mitigation: two independent RHT rounds (still O(d log d))
    n=np.linalg.norm(X,axis=1,keepdims=True)
    U=R.rotate(R.rotate(X/n,s1),s2); d=X.shape[1]
    cb=R.turbo_codebook(d,b).astype(np.float32); e=(np.float32(.5)*(cb[1:]+cb[:-1])).astype(float)
    Uh=R.turbo_codebook(d,b)[np.searchsorted(e,U)]
    return R.unrotate(R.unrotate(Uh,s2),s1)*n

for d in (64,128):
    Q=haar(d)
    print(f"\n=== d={d} ===  (mean normalised error; bound = sqrt(3)pi/2 * 4^-b)")
    print(f"{'input':34s}"+"".join(f"  b={b}:RHT/2xRHT/Haar      " for b in (2,3,4)))
    inputs={}
    inputs["one-hot, every position (d vecs)"]=np.eye(d)
    k=np.zeros((d,d)); 
    for i in range(d): k[i,i]=1; k[i,(i+1)%d]=1
    inputs["two equal spikes"]=k
    for ratio in (10,30):
        X=rng.standard_normal((2048,d)); ch=rng.integers(0,d,2048)
        X[np.arange(2048),ch]*=ratio
        inputs[f"Gaussian + one channel x{ratio}"]=X
    inputs["Gaussian (control)"]=rng.standard_normal((2048,d))
    for name,X in inputs.items():
        row=f"{name:34s}"
        for b in (2,3,4):
            row+=f"  {rel(X,tq(X,b)):.4f}/{rel(X,tq_double(X,b)):.4f}/{rel(X,tq_haar(X,b,Q)):.4f}"
        print(row)
    print(f"{'Theorem 1 bound':34s}"+"".join(f"  {BOUND(b):.4f}                " for b in (2,3,4)))

# ---- mitigation 2: RHT -> fixed random permutation -> RHT
def tq_perm(X,b,perm,s1=R.DEFAULT_SEED,s2=0x12345678):
    n=np.linalg.norm(X,axis=1,keepdims=True); d=X.shape[1]
    U=R.rotate(R.rotate(X/n,s1)[:,perm],s2)
    cb=R.turbo_codebook(d,b).astype(np.float32); e=(np.float32(.5)*(cb[1:]+cb[:-1])).astype(float)
    Uh=R.turbo_codebook(d,b)[np.searchsorted(e,U)]
    Y=R.unrotate(Uh,s2); inv=np.argsort(perm); return R.unrotate(Y[:,inv],s1)*n
print("\n--- RHT + permutation + RHT, worst over sparse inputs ---")
for d in (64,128):
    perm=np.random.default_rng(7).permutation(d)
    worst={b:0 for b in (2,3,4,5,6)}
    for spikes in (1,2,3,4):
        X=np.zeros((d,d))
        for i in range(d):
            for j in range(spikes): X[i,(i+j*7)%d]=1
        for b in worst: worst[b]=max(worst[b],rel(X,tq_perm(X,b,perm)))
    print(d, {b:(round(v,4), round(v/BOUND(b),2)) for b,v in worst.items()})
print("\n--- single RHT, worst over the same sparse inputs (ratio to bound) ---")
for d in (64,128):
    worst={b:0 for b in (2,3,4,5,6)}
    for spikes in (1,2,3,4):
        X=np.zeros((d,d))
        for i in range(d):
            for j in range(spikes): X[i,(i+j*7)%d]=1
        for b in worst: worst[b]=max(worst[b],rel(X,tq(X,b)))
    print(d, {b:(round(v,4), round(v/BOUND(b),2)) for b,v in worst.items()})
