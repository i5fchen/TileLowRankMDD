from numpy.linalg import norm
from numpy import array
class x2qqt_global:
    def __init__(self, ia, rank):
        self.ia = ia
        self._q_shape = (self.ia.shape[-1], rank), 
    def loss(self, x, idx, ib):
        res = self.ia
        for i in range(len(self._q_shape)):
            xi = x[i].reshape(self._q_shape[i])
            res = res.dot(xi).dot(xi.T)
        self.res = res-ib
        loss = norm(self.res.ravel())**2
        return 0.5*loss
    def grads(self, x, ig):
        ret = self.ia.T.conj().dot(self.res)
        ret = ret+ret.T
        for i in range(len(self._q_shape)):
            xi = x[i].reshape(self._q_shape[i])
            ret = ret.dot(xi.conj())
        return ret.ravel()

class x2lr_global:
    def __init__(self, ia, rank):
        self.ia = ia
        self._l_shape = (self.ia.shape[-1], rank) 
        self._r_shape = self._l_shape[::-1]
        self._lr_shape = (self._l_shape, self._r_shape)
    def loss(self, x, idx, ib): # idx is not used for global factorization
        res = self.ia
        for i in range(len(self._lr_shape)):
            xi = x[i].reshape(self._lr_shape[i])
            res = res.dot(xi)
        self.res = res-ib
        loss = norm(self.res.ravel())**2
        return 0.5*loss
    def grads(self, x, ig):
        ret = self.ia.T.conj().dot(self.res)
        xi = x[1-ig].reshape(self._lr_shape[1-ig])
        if ig==0:
            ret = ret.dot(xi.conj().T)
        if ig==1:
            ret = xi.conj().T.dot(ret)
        return ret.ravel()
       
class x2qlr_tiled_dense_mmm:
    
    def __init__(self, ia, dim_tiles, address):
        # address: a list of tuple to indicate which low-rank factors belong to the same group to be updated
        self.ia = ia
        # The dimension of tiled/block matrix, the element (iv,jv) in each entry indicates the dimension of the tile
        self.dim_tiles = dim_tiles
        self.factor_address = address
        n_dense = self.ia.shape[-1]
        # Dense matrix to hold X constructed from tiled xq, xl, and xr to calculate A.dot(X)
        self.x = np.empty_like(self.ia, dtype=ia.dtype, shape=(n_dense,n_dense))
        for indicator in self.xgroup_tmple:
            _tile2dense(self, indicator)
        
        self.m_tiles, self.n_tiles = self.dim_tiles.shape

        # auxilliaries for dense-tile conversion
        self.cum_r = np.array([self.dim_tiles[i,0][0] for i in range(self.m_tiles)])
        self.cum_c = np.array([self.dim_tiles[0,i][1] for i in range(self.n_tiles)])
        self.cum_r = np.cumsum(self.cum_r)
        self.cum_c = np.cumsum(self.cum_c)
    
    def _tile2dense_qqt(self, x, indicator):
        for idx in self.factor_address[indicator]:
            dim_local_tile = self.dim_tiles[idx]
            num_local_tile = np.product(dim_local_tile)
            
            xtmp = x[:num_local].reshape(dim_local_tile)
            
            xtmp = xtmp.dot(xtmp.T)
            ir, ic = xtmp.shape
            ii1, jj1 = self.factor_address[indicator]
            i1, j1 = self.cum_r[ii1], self.cum_c[jj1]
            self.x[i1-ir:i1, j1-ic:j1] = xtmp
            x = x[num_local:]
            return None
    def _tile2dense(self, x, indicator):
        for idx in self.factor_address[indicator]:
            dim_local_tile = self.dim_tiles[idx]
            num_local_tile = np.product(dim_local_tile)
            
            xtmp = x[:num_local].reshape(dim_local_tile)
            
            xtmp = xtmp.dot(xtmp.T)
            ir, ic = xtmp.shape
            ii1, jj1 = self.factor_address[indicator]
            i1, j1 = self.cum_r[ii1], self.cum_c[jj1]
            self.x[i1-ir:i1, j1-ic:j1] = xtmp
            x = x[num_local:]
            return None
    def _tile2dense_lr(self, xl, xr, indicator):
        for idx in self.factor_group[indicator]:
            dim_local_tile = self.dim_tiles[idx]
            num_local_tile = np.product(dim_local_tile)
            
            xltmp = xl[:num_local].reshape(dim_local_tile)
            xrtmp = xr[:num_local].reshape(dim_local_tile[::-1])
            
            xrtmp = xltmp.dot(xrtmp.T)
            ir, ic = xrtmp.shape
            ii1, jj1 = self.factor_address[indicator]
            i1, j1 = self.cum_r[ii1], self.cum_c[jj1]
            self.x[i1-ir:i1, j1-ic:j1] = xrtmp
            self.x[j1-ic:j1, i1-ir:i1] = xrtmp
            xl = xl[num_local:]
            xr = xr[num_local:]
            return None
    
    def loss(self, x, idx, rhs):
        # x = [arr0, arr1, ....]
        # idx is the index indicating which array in x has been updated
        if 0==idx:
            _tile2dense_qqt(self, x[idx], idx)
        elif 'l' in self.factor_address[idx]:
            _tile2dense_lr(self, x[idx], x[idx+1], idx)
        else:
            _tile2dense_lr(self, x[idx-1], x[idx], idx)

        self.res = self.ia.dot(self.x)-rhs
        loss = 0.5 * norm(res.ravel()) ** 2
        return loss
 
    def _dense2tile(self, x):
        x = np.hsplit(x, self.cum_c)
        g = [np.hsplit(ii,self.cum_r) for ii in x]
        g = np.array(g, dtype=object)
        return g   

    def grads(self, x, idx):
        ata = self.ia.T.conj().dot(self.res)
        ata = ata + ata.T
        ata = _dense2tile(self, ata)
        if 0==idx:
            gg = []
            for iaddr in self.factor_address(idx):
                iata = ata[iaddr]
                dim_local_tile = self.dim_tiles[iaddr]
                num_local_tile = np.product(dim_local_tile)
                xtmp = x[idx][:num_local].reshape(dim_local_tile)
                iata = iata.dot(xtmp.conj())
                gg.append(iata.ravel()) 
        elif 'l' in self.factor_address[idx]:
            gg = []
            for iaddr in self.factor_address(idx+1):
                iata = ata[iaddr]
                dim_local_tile = self.dim_tiles[iaddr]
                num_local_tile = np.product(dim_local_tile)    
                xrtmp = x[idx+1][:num_local].reshape(dim_local_tile[::-1])
                iata = iata.dot(xrtmp.T.conj())
                gg = append(iata.ravel())
        else:
            gg = []
            for iaddr in self.factor_address(idx):
                iata = ata[iaddr]
                dim_local_tile = self.dim_tiles[iaddr]
                num_local_tile = np.product(dim_local_tile)
                xltmp = x[idx-1][:num_local].reshape(dim_local_tile)
                iata = xltmp.T.conj().dot(iata)
                gg.append(iata.ravel())
  
        return np.array(gg)
