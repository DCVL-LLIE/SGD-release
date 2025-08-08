import torch
import torch.nn.functional as F

def svd_decomposition(images):
    B, C, H, W = images.shape
    U, S, V = [], [], []

    for b in range(B):
        U_b, S_b, V_b = [], [], []
        for c in range(C):  
            u, s, v = torch.linalg.svd(images[b, c], full_matrices=False)
            U_b.append(u)
            S_b.append(s)
            V_b.append(v)
        
        U.append(torch.stack(U_b, dim=0))  
        S.append(torch.stack(S_b, dim=0))  
        V.append(torch.stack(V_b, dim=0))  
    
    U = torch.stack(U, dim=0)  
    S = torch.stack(S, dim=0)  
    V = torch.stack(V, dim=0)  

    return U, S, V

def svd_reconstruction(U, S, V):
    B, C, H, W = V.shape
    reconstructed = []

    for b in range(B):  
        reconstructed_channels = []
        for c in range(C):
            S_diag = torch.diag(S[b, c])

            channel = torch.matmul(torch.matmul(U[b, c], S_diag), V[b, c])
            reconstructed_channels.append(channel)

        reconstructed.append(torch.stack(reconstructed_channels, dim=0))

    reconstructed = torch.stack(reconstructed, dim=0)
    return reconstructed

def downsample(x, k):
    return F.avg_pool2d(x, kernel_size=k)

def upsample(x, k):
    return F.interpolate(x, scale_factor=k, mode = 'bilinear', align_corners=False)

def check_image_size(x, down_factor):
    _, _, h, w = x.size()
    mod_pad_h = (down_factor - h % down_factor) % down_factor
    mod_pad_w = (down_factor - w % down_factor) % down_factor
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x