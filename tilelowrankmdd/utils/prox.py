from numpy.linalg import norm

class prox_Fro:
    
    def __init__(self, lam):
    
        self.lam = lam
    
    def project(self, x, scale):
        
        self.norm_before = 0.5 * self.lam * norm(x) ** 2
        x = x/(1+self.lam/scale)
        self.norm_after = 0.5 * self.lam * norm(x) ** 2
        
        return x
