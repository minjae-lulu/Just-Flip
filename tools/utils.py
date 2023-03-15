import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from PIL import Image
import math
from torchvision.transforms import ToTensor

matplotlib.use('Agg')


def count_trainable_parameters(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    return sum([np.prod(p.size()) for p in model_parameters])


def get_nsamples(data_loader, N):
    x = []
    n = 0
    while n < N:
        x_next, _ = next(iter(data_loader))
        x.append(x_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    return x


def get_camera_wireframe(scale: float = 0.03):
    """
    Returns a wireframe of a 3D line-plot of a camera symbol.
    """
    a = 0.5 * torch.tensor([-2, 1.5, 4])
    up1 = 0.5 * torch.tensor([0, 1.5, 4])
    up2 = 0.5 * torch.tensor([0, 2, 4])
    b = 0.5 * torch.tensor([2, 1.5, 4])
    c = 0.5 * torch.tensor([-2, -1.5, 4])
    d = 0.5 * torch.tensor([2, -1.5, 4])
    C = torch.zeros(3)
    F = torch.tensor([0, 0, 3])
    camera_points = [a, up1, up2, up1, b, d, c, a, C, b, d, C, c, C, F]
    lines = torch.stack([x.float() for x in camera_points]) * scale
    return lines


def plot_cameras(ax, c2w, color: str = "blue", scale=1.0):
    device = c2w.device
    nbatch = c2w.shape[0]
    cam_wires_canonical = get_camera_wireframe(scale)[None].to(device)
    R = c2w[:, :3, :3] @ torch.tensor([[1., 0, 0], [0, 1, 0], [0, 0, -1]], device=device)
    R = torch.cat([r.T[None] for r in R], 0)
    cam_wires_trans = torch.bmm(cam_wires_canonical.repeat(nbatch, 1, 1), R) + c2w[:, None, :3, -1]
    plot_handles = []
    for wire in cam_wires_trans:
        # the Z and Y axes are flipped intentionally here!
        x_, y_, z_ = wire.detach().cpu().numpy().T.astype(float)
        (h,) = ax.plot(x_, y_, z_, color=color, linewidth=0.3)
        plot_handles.append(h)
    return plot_handles


def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def plot_camera_scene(c2w, c2w_gt=None, plot_radius=5.0, status=''):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=45., azim=60)
    ax.set_xlim3d([-plot_radius, plot_radius])
    ax.set_ylim3d([-plot_radius, plot_radius])
    ax.set_zlim3d([0, plot_radius * 2])

    xspan, yspan, zspan = 3 * [np.linspace(-plot_radius, plot_radius, 20)]
    zero = np.zeros_like(xspan)
    ax.plot3D(xspan, zero, zero, 'k--')
    ax.plot3D(zero, yspan, zero, 'k--')
    ax.plot3D(zero, zero, zspan + plot_radius, 'k--')
    ax.text(plot_radius, .5, .5, "x", color='red')
    ax.text(.5, plot_radius, .5, "y", color='green')
    ax.text(.5, .5, plot_radius * 2, "z", color='blue')

    scale = 0.05 * plot_radius
    handle_cam = plot_cameras(ax, c2w, color="#FF7D1E", scale=scale)
    if c2w_gt is not None:
        handle_cam_gt = plot_cameras(ax, c2w_gt, color="#812CE5", scale=scale)

        labels_handles = {
            "Estimated Cameras": handle_cam[0],
            "GT Cameras": handle_cam_gt[0],
        }
    else:
        labels_handles = {
            "Estimated cameras": handle_cam[0]
        }

    ax.legend(
        labels_handles.values(),
        labels_handles.keys(),
        loc="upper center",
        bbox_to_anchor=(0.32, 0.7),
        prop={'size': 8}
    )

    ax.axis('off')
    fig.tight_layout()

    img = fig2img(fig)

    plt.close(fig)

    img = ToTensor()(img)

    return img





# X_cordi, Y_codri, Z_cordi  # input camera coordinatesare defined by FloatTensor
# Ax + By + Cz + D = 0 are symmetric plane equation

def Find_Optimal_Sphere(X_cordi,Y_cordi,Z_cordi):
    A = np.zeros((len(X_cordi),4))
    A[:,0] = X_cordi*2
    A[:,1] = Y_cordi*2
    A[:,2] = Z_cordi*2
    A[:,3] = 1
    
    f = torch.zeros((len(X_cordi),1))
    f[:,0] = (X_cordi*X_cordi) + (Y_cordi*Y_cordi) + (Z_cordi*Z_cordi)
    C, residules, rank, singval = np.linalg.lstsq(A,f)
    
    radius2 = (C[0]*C[0])+(C[1]*C[1])+(C[2]*C[2])+C[3]
    radius = math.sqrt(radius2)

    return radius, C[0], C[1], C[2]


def Symmetrically_Transforming(X_cordi,Y_cordi,Z_cordi, A,B,C,D):
    X_flipped_cordi = []
    Y_flipped_cordi = []
    Z_flipped_cordi = []
    
    for i in range(len(X_cordi)):
        X_flipped_cordi.append(X_cordi[i] + 2*A*(-(A*X_cordi[i] + B*Y_cordi[i] + C*Z_cordi[i] + D)/(A*A+B*B+C*C)))
        Y_flipped_cordi.append(Y_cordi[i] + 2*B*(-(A*X_cordi[i] + B*Y_cordi[i] + C*Z_cordi[i] + D)/(A*A+B*B+C*C)))
        Z_flipped_cordi.append(Z_cordi[i] + 2*C*(-(A*X_cordi[i] + B*Y_cordi[i] + C*Z_cordi[i] + D)/(A*A+B*B+C*C)))
    
    X_flipped_cordi = torch.FloatTensor(X_flipped_cordi)
    Y_flipped_cordi = torch.FloatTensor(Y_flipped_cordi)
    Z_flipped_cordi = torch.FloatTensor(Z_flipped_cordi)

    return X_flipped_cordi, Y_flipped_cordi, Z_flipped_cordi


def Projection2Sphere(x,y,z, x0,y0,z0,r):
    a = x*x + y*y + z*z
    b = x*x0 + y*y0 + z*z0 
    c = x0*x0 + y0*y0 + z0*z0 -r*r
    
    alpha1 = (-b + math.sqrt(b*b - 4*a*c) )/(2*a)
    alpha2 = (-b - math.sqrt(b*b - 4*a*c) )/(2*a)
    
    if abs(alpha1-1) > abs(alpha2-1):
        alpha = alpha2
    else:
        alpha = alpha1
    
    return alpha*x, alpha*y, alpha*z
    
    
up = torch.FloatTensor([0,0,1])
def Get_Transform_Matrics(x,y,z, x0,y0,z0, up):
    # c = torch.FloatTensor([x,y,z])
    c = torch.stack([x,y,z])
    c = c.squeeze()
    at = torch.FloatTensor([x0,y0,z0])
    at = at.squeeze()
    
    torch0 = torch.FloatTensor([0.])
    temp = torch.FloatTensor([0,0,0,1])

    z_axis = (c-at) / torch.norm(c-at)
    x_axis = torch.cross(up, z_axis) / torch.norm(torch.cross(up, z_axis))
    y_axis = torch.cross(z_axis,x_axis)
    
    z = torch.cat([z_axis, torch0])
    x = torch.cat([x_axis, torch0])
    y = torch.cat([y_axis, torch0])
    
    rotation_mat = torch.stack([x, y, z, temp], dim=-1)
    translation_mat = torch.eye(4)
    translation_mat[:3,3] = c 
    transform_mat = torch.matmul(translation_mat, rotation_mat) 
      
    return transform_mat


A = 0; B = 1; C = 0; D = 0
up = torch.FloatTensor([0,0,1])
def Find_Optimal_CameraPose (X_cordi, Y_cordi, Z_cordi):
    flipped_camera_pose_set = []
    
    r, x0, y0, z0 = Find_Optimal_Sphere(X_cordi,Y_cordi,Z_cordi)
    X_flipped_cordi, Y_flipped_cordi, Z_flipped_cordi = Symmetrically_Transforming(X_cordi,Y_cordi,Z_cordi, A,B,C,D)
    
    for i in range(len(X_cordi)):
        X_optim, Y_optim, Z_optim = Projection2Sphere(X_flipped_cordi[i], Y_flipped_cordi[i], Z_flipped_cordi[i], x0,y0,z0,r)
        transform_mat = Get_Transform_Matrics(X_optim.float(), Y_optim.float(), Z_optim.float(), x0,y0,z0, up)
        flipped_camera_pose_set.append(transform_mat)
        
    return flipped_camera_pose_set



