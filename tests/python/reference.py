"""Small differentiable CPU oracle. No Metal extension; used away from branch boundaries."""
import math
import torch

C0 = 0.28209479177387814
C1 = 0.4886025119029199
C2 = [1.0925484305920792, -1.0925484305920792, 0.31539156525252005, -1.0925484305920792, 0.5462742152960396]
C3 = [-0.5900435899266435, 2.890611442640554, -0.4570457994644658, 0.3731763325901154, -0.4570457994644658, 1.445305721320277, -0.5900435899266435]


def sh_color(means, campos, sh, degree):
    direction = means - campos
    direction = direction / direction.norm(dim=-1, keepdim=True)
    x, y, z = direction.unbind(-1)
    basis = [torch.ones_like(x) * C0]
    if degree > 0:
        basis += [-C1*y, C1*z, -C1*x]
    if degree > 1:
        basis += [C2[0]*x*y, C2[1]*y*z, C2[2]*(2*z*z-x*x-y*y), C2[3]*x*z, C2[4]*(x*x-y*y)]
    if degree > 2:
        basis += [C3[0]*y*(3*x*x-y*y), C3[1]*x*y*z, C3[2]*y*(4*z*z-x*x-y*y),
                  C3[3]*z*(2*z*z-3*x*x-3*y*y), C3[4]*x*(4*z*z-x*x-y*y),
                  C3[5]*z*(x*x-y*y), C3[6]*x*(x*x-3*y*y)]
    return (torch.stack(basis, -1)[..., None] * sh[:, :len(basis)]).sum(1).add(0.5).clamp_min(0)


def covariance(scales, rotations, modifier):
    # Upstream GLM constructor takes columns. No quaternion normalization here.
    r, x, y, z = rotations.unbind(-1)
    columns = [1-2*(y*y+z*z), 2*(x*y-r*z), 2*(x*z+r*y),
               2*(x*y+r*z), 1-2*(x*x+z*z), 2*(y*z-r*x),
               2*(x*z-r*y), 2*(y*z+r*x), 1-2*(x*x+y*y)]
    rotation = torch.stack(columns, -1).reshape(-1, 3, 3).transpose(-1, -2)
    matrix = torch.diag_embed(scales*modifier) @ rotation
    sigma = matrix.transpose(-1, -2) @ matrix
    return torch.stack([sigma[:,0,0], sigma[:,0,1], sigma[:,0,2], sigma[:,1,1], sigma[:,1,2], sigma[:,2,2]], -1)


def render(data, settings):
    means = data['means3D']
    n = len(means)
    w, h = settings.image_width, settings.image_height
    view = settings.viewmatrix.reshape(4, 4).T
    proj = settings.projmatrix.reshape(4, 4).T
    colors = data.get('colors_precomp')
    if colors is None:
        colors = sh_color(means, settings.campos, data['shs'], settings.sh_degree)
    cov = data.get('cov3D_precomp')
    if cov is None:
        cov = covariance(data['scales'], data['rotations'], settings.scale_modifier)
    fx, fy = w/(2*settings.tanfovx), h/(2*settings.tanfovy)
    projected, radii = [], []
    for i in range(n):
        hom = torch.cat([means[i], means.new_ones(1)])
        t = (view @ hom)[:3]
        if t[2].item() <= float(torch.tensor(0.2, dtype=torch.float32)):
            projected.append(None); radii.append(0); continue
        clip = proj @ hom
        center = ((clip[:2]/(clip[3]+1e-7)+1)*means.new_tensor([w,h])-1)*0.5
        # Original means2D is a dummy whose backward reports the normalized screen gradient.
        center = center + data['means2D'][i,:2]*means.new_tensor([w*0.5,h*0.5])
        tx = (t[0]/t[2]).clamp(-1.3*settings.tanfovx,1.3*settings.tanfovx)*t[2]
        ty = (t[1]/t[2]).clamp(-1.3*settings.tanfovy,1.3*settings.tanfovy)*t[2]
        zero = t.new_zeros(())
        jacobian = torch.stack([fx/t[2],zero,-fx*tx/t[2]**2,zero,fy/t[2],-fy*ty/t[2]**2]).reshape(2,3)
        v = cov[i]
        sigma = torch.stack([v[0],v[1],v[2],v[1],v[3],v[4],v[2],v[4],v[5]]).reshape(3,3)
        a = jacobian @ view[:3,:3]
        cov2d = a @ sigma @ a.T + torch.eye(2,dtype=means.dtype)*0.3
        xx, xy, yy = cov2d[0,0],cov2d[0,1],cov2d[1,1]
        det = xx*yy-xy*xy
        conic = torch.stack([yy/det,-xy/det,xx/det])
        mid = (xx+yy)*0.5
        radius = math.ceil((3*(mid+(mid*mid-det).clamp_min(0.1).sqrt()).sqrt()).detach().item())
        cx, cy = center.detach().tolist()
        tiles_x, tiles_y = (w+15)//16,(h+15)//16
        rect = [max(0,min(tiles_x,int((cx-radius)/16))),max(0,min(tiles_y,int((cy-radius)/16))),
                max(0,min(tiles_x,int((cx+radius+15)/16))),max(0,min(tiles_y,int((cy+radius+15)/16)))]
        if rect[0]==rect[2] or rect[1]==rect[3]:
            projected.append(None);radii.append(0);continue
        projected.append((center, conic, rect, float(t[2].detach())))
        radii.append(radius)
    pixels=[]
    for y in range(h):
        for x in range(w):
            ids = [i for i,p in enumerate(projected) if p is not None and p[2][0]<=x//16<p[2][2] and p[2][1]<=y//16<p[2][3]]
            ids.sort(key=lambda i:(projected[i][3],i))
            transmittance=means.new_ones(())
            color=means.new_zeros(3)
            for i in ids:
                center,conic,_,_=projected[i]
                dx,dy = center-means.new_tensor([x,y])
                power=-0.5*(conic[0]*dx*dx+conic[2]*dy*dy)-conic[1]*dx*dy
                if power.detach().item()>0:continue
                alpha=(data['opacities'][i,0]*power.exp()).clamp_max(0.99)
                if alpha.detach().item()<1/255:continue
                next_t=transmittance*(1-alpha)
                if next_t.detach().item()<1e-4:break
                color=color+colors[i]*alpha*transmittance
                transmittance=next_t
            pixels.append(color+transmittance*settings.bg if n else color)
    return torch.stack(pixels).reshape(h,w,3).permute(2,0,1),torch.tensor(radii,dtype=torch.int32)
