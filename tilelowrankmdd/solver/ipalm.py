from numpy.linalg import norm
from numpy import zeros, sqrt, inf

def back_tracking(loss_new, loss_cur, x_new, x_tmp, gg, tau):
    
    if loss_new < loss_cur + 0.5 * ( norm(x_new - x_tmp)**2 * tau - norm(gg)**2 / tau ):
        return 'succeed'
    else:
        return False

def ipalm(H, ib, proxs, xs0, tau_init, tol, iter_lim=None, inc=2, inertia_kind=0, verbose=False, nbt_lim=10): 
    
    nblocks=len(proxs)
    
    xs1=list(xs0)
    xs2=list(xs0)
    
    it=0 

    total_loss_new = inf
    t0 = t1 = 1.0
    if True == verbose:
        print("%11s%15s%11s%15s%15s"%('iter', 'loss','cnt', 'prox_before', 'prox_after'), end='\n'*2, flush=True)
           
    while (True): 
        
        loss_fs = 0.0    
        
        if 0 == inertia_kind: 
            inertia = it/(it+3.0)
        else:         
            inertia = (t0-1)/t1
            
            t2 = 1 + sqrt(1+4*t0**2) 
            t0 = t1
            t1 = 0.5*t2
             
        for i in range(nblocks):

            yi = xs1[i] + inertia * (xs1[i] - xs0[i])
            zi = xs1[i] + inertia * (xs1[i] - xs0[i])
             
            xs2[i] = zi
            
            loss_H_cur = H.loss(xs2, i, ib)
            gg = H.grads(xs2, i)
            
            ############################################################## 

            cnt = 0
            while(True):           
    
                tau = tau_init[i]
                
                y_tmp = yi - 1.0/tau * gg
                
                x_new = proxs[i].project(y_tmp, tau)
                nv  = proxs[i].norm_before
                nnv = proxs[i].norm_after
                
                xs2[i] = x_new
                loss_H_new = H.loss(xs2, i, ib)
                
                if back_tracking(loss_H_new, loss_H_cur, x_new, y_tmp, gg, tau) == 'succeed' or cnt > nbt_lim:
                    break

                cnt += 1
                tau_init[i] *= inc
            ############################################################## 

            loss_fs += nv

        total_loss_cur = loss_H_cur + loss_fs
        
        if verbose and (it<20 or 0==(it+1)%50):
            print("%11s%15.3f%11d"%(it+1, total_loss_cur,cnt+1), end='', flush=True)
            for i in range(nblocks):
                print("%15.3f%15.3f"%(nv, nnv), end='', flush=True)
            print("\n")
       
        if abs(total_loss_cur/total_loss_new-1.0) < tol: break
        
        total_loss_new = total_loss_cur
      
        xs0 = list(xs1)
        xs1 = list(xs2)
        
        it+=1 
        
        if iter_lim is not None and iter_lim<it: break
    
    return xs1
