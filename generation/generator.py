import os, math, random, argparse, csv
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
import tifffile
from PIL import Image
from tqdm import tqdm

from ..models.vae import CustomVAE, reparameterize
from ..models.unet import TimeUNet
from ..diffusion.ddpm import DDPM
from ..utils.fem import TorchFEMMesh
from ..utils.image_metrics import _make_gaussian_kernel, compute_relative_SA_torch, average_slice_sa, average_slice_autocorr, compute_tpc_torch

def generate_single_volume(
    vae, unet, ddpm, device,
    init_image_path=None,
    num_samples=16, latent_ch=4, H=16, W=16,
    sds_steps=500, sds_lr=1e-3, t_min=200, t_max=500,
    refinement_K=10,
    tpc_targets=None,    # dict phase -> target mean tpc (1D numpy)
    save_dir=None,
    tpc_weight=0.0,       # weight for TPC guidance
    vf0 = 0.24,
    vf05 = 0.32,
    vf1 = 0.46,
    phases=None,
    w_m1=10000000, w_m2=10000000,
    sa_targets=None,   # dict phase -> target relative SA (e.g. {0:0.2,1:0.5,2:0.3})
    sa_weight=0.0,     # float scaling the SA loss
    rd_targets=None,
    rd_weight=0.0,
    save_interval_vol = 10000

):
    """
    Generates one 64×64×64 volume via latent-diffusion → 3D decode → refinements → 3D SDS.
    Returns a numpy array volume_q in {0.0,0.5,1.0}.
    """
    # 1) Initialize latent x
    if init_image_path:
        img = Image.open(init_image_path).convert('L')
        tfm = transforms.Compose([transforms.Resize((64,64)), transforms.ToTensor()])
        x_img = tfm(img).unsqueeze(0).to(device)        # [1,1,64,64]
        x_norm = 2*x_img - 1
        with torch.no_grad():
            mu, _ = vae.encode(x_norm)
            x = reparameterize(mu, _)
        if x.shape[0] < num_samples:
            x = x.repeat(num_samples,1,1,1)
        else:
            x = x[:num_samples]
    else:
        x = torch.randn(num_samples, latent_ch, H, W, device=device)

    # 2) Initial DDPM denoising
    for t in tqdm(reversed(range(ddpm.num_timesteps)), desc="DDPM denoise"):
        ts = torch.full((num_samples,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            x = ddpm.p_sample(unet, x, ts)
        if t > 0:
            x = x.transpose(0,2).transpose(3,0)

    # 3) Multi-axis decode → [64,1,64,64]
    with torch.no_grad():
        latent = x.permute(1,0,2,3).contiguous()  # [C,D,H,W]
        C,D,H_,W_ = latent.shape
        decoded_acc = torch.zeros((1,64,64,64), device=device)
        # depth
        for d in range(D):
            dec = vae.decode(latent[:,d,:,:].unsqueeze(0)).squeeze(0)  # [1,64,64]
            decoded_acc[:,d*4:(d+1)*4, :, :] += dec.unsqueeze(1)
        # height
        for h in range(H_):
            dec = vae.decode(latent[:,:,h,:].unsqueeze(0)).squeeze(0)
            decoded_acc[:,:,h*4:(h+1)*4,:] += dec.unsqueeze(2)
        # width
        for w in range(W_):
            dec = vae.decode(latent[:,:,:,w].unsqueeze(0)).squeeze(0)
            decoded_acc[:,:,:,w*4:(w+1)*4] += dec.unsqueeze(3)
        decoded_3d = (decoded_acc/3.0).permute(1,0,2,3).clamp(0,1)  # [64,1,64,64]

    # 4) Three-axis latent refinement
    X = decoded_3d.clone()
    Z, C, H3, W3 = X.shape
    with torch.no_grad():
        for _ in tqdm(range(refinement_K),desc="Three-axis latent refinement"):
            new_X = torch.zeros_like(X)
            # 1) depth slices
            for z in range(Z):
                img = X[z:z+1]                 # [1,1,64,64]
                img_in = img*2 - 1             # [-1,1]
                mu,_ = vae.encode(img_in)
                dec = vae.decode(mu)[0,0]      # [64,64]
                new_X[z,0] += dec              # [64,64] into depth plane
            # 2) height slices
            for y in range(H3):
                sl = X[:,0,y,:]               # [Z, W3] = [64,64]
                img = sl.unsqueeze(0).unsqueeze(0)  # [1,1,64,64]
                img_in = img*2 - 1
                mu,_ = vae.encode(img_in)
                dec = vae.decode(mu)[0,0]     # [64,64]
                new_X[:,0,y,:] += dec         # scatter across depths
            # 3) width slices
            for w in range(W3):
                sl = X[:,0, :,w].squeeze(1)   # [Z, H3] = [64,64]
                img = sl.unsqueeze(0).unsqueeze(0)
                img_in = img*2 - 1
                mu,_ = vae.encode(img_in)
                dec = vae.decode(mu)[0,0]     # [64,64]
                new_X[:,0,:,w] += dec         # scatter across depths
            X = (new_X/3.0).clamp(0,1)
    decoded_3d = X

    # 5) 3D SDS slice-wise refinement
    vol_cpu = decoded_3d.squeeze(1).cpu()  # [Z,H,W]
    Z, H3, W3 = vol_cpu.shape

    mesh2d = TorchFEMMesh(M=H3, N=W3, low_cond=0.001, device=device).to(device)

    yy, xx = torch.meshgrid(
        torch.arange(H3, device=device),
        torch.arange(W3, device=device),
        indexing='ij'
    )
    r = ( (yy - H3//2)**2 + (xx - W3//2)**2 ).sqrt().round().long().view(-1)
    nbins = int(r.max().item()) + 1
    bin_mat = F.one_hot(r, num_classes=nbins).float().transpose(0,1)   # [nbins, H3*W3]
    bin_counts = bin_mat.sum(dim=1, keepdim=True)                     # [nbins,1]

    for iter_sds in tqdm(range(sds_steps), desc="3D SDS refine with TPC"):
        axis = random.choice([0,1,2]); idx = random.randrange(Z)
        # extract slice
        if axis==0:
            sl = vol_cpu[idx]
        elif axis==1:
            sl = vol_cpu[:, idx, :]
        else:
            sl = vol_cpu[:, :, idx]
        img = sl.unsqueeze(0).unsqueeze(0).to(device)
        img_in = img*2 - 1

        # encode latent
        mu, _ = vae.encode(img_in)
        latent_s = mu.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent_s], lr=sds_lr)
        

        # SDS loss
        t = torch.randint(t_min, t_max, (1,), device=device)
        noise = torch.randn_like(latent_s)
        a = ddpm.sqrt_acp[t].view(1,1,1,1)
        s = ddpm.sqrt_om_acp[t].view(1,1,1,1)
        x_t = a*latent_s + s*noise
        pred = unet(x_t, t)
        loss_sds = (s.pow(2) * F.mse_loss(pred, noise, reduction='none')).mean()


 
        # 4) Volume fraction guidance
        decoded = vae.decode(latent_s).clamp(0,1).requires_grad_(True) 
        # compute per-sample VF then average

        # *** multi-phase volume-fraction guidance via moments ***
        #decoded = dec(x)                       # [B,1,H,W]
        B = decoded.shape[0]
        # compute empirical moments
        m1 = decoded.view(B, -1).mean(dim=1)   # E[D]
        m2 = (decoded**2).view(B, -1).mean(dim=1)# E[D^2]
        # compute target moments from (f0,f05,f1)
        t_mean   = 0.0*vf0   + 0.5*vf05   + 1.0*vf1
        t_sqmean = 0.0*vf0**2 + 0.5**2*vf05 + 1.0**2*vf1
        loss_m1  = F.mse_loss(m1, torch.full_like(m1,   t_mean))
        loss_m2  = F.mse_loss(m2, torch.full_like(m2, t_sqmean))     

        # SDS SA
        if sa_targets is not None and sa_weight > 0:
            rel_sa = compute_relative_SA_torch(decoded, phases,
                                               kernel_size=7, sigma=1.0)
            tgt_sa = torch.tensor([sa_targets[p] for p in phases],
                                  device=rel_sa.device)
            loss_sa = F.mse_loss(rel_sa, tgt_sa)
            #print(rel_sa, tgt_sa)

        else:
            loss_sa = torch.tensor(0., device=device)


        # inside your SDS refinement step, after you have `decoded` in [0,1] shape [1,1,H,W]:
        B, C, h_dec, w_dec = decoded.shape  # decoded is [1,1,64,64]
        P = len(phases)
        # build soft masks like we did for SA
       
        x = decoded.repeat(1, P, 1, 1)
        levels = torch.linspace(0,1,P,device=device)[None,:,None,None]  # [1,P,1,1]
        dist   = (x - levels).abs().view(1, P, -1)                      # [1,P,h_dec*w_dec]
        masks  = F.softmax(-30.0 * dist, dim=1).view(1, P, h_dec, w_dec)                

        # Deff
        if rd_targets is not None and rd_weight > 0:


            deff = []
            for p in range(P):
                mask_p = masks[0,p]          # [H,W]
                deff.append(mesh2d(mask_p))  # scalar, differentiable

            deff = torch.stack(deff)         # [P]

            #rd_targets = torch.tensor([1.0, 0, 0], dtype=torch.float32)

            deff_targets = torch.tensor([rd_targets[p] for p in phases],
                                  device=device)
            #print(deff, deff_targets)

            loss_rd = F.mse_loss(deff, deff_targets.to(device))

        else:
            loss_rd = torch.tensor(0., device=device)



        if tpc_targets is not None and tpc_weight > 0:
            loss_tpc = torch.tensor(0.0, device=device)
            # `masks`: [1,P,H3,W3] from your soft-assignment step
            masks_p = masks[0]  # [P,H3,W3]
            for pi, p in enumerate(phases):
                mask_p   = masks_p[pi]  # [H3,W3]
                tpc_pred = compute_tpc_torch(mask_p, bin_mat, bin_counts)  # [nbins]
                # build target tensor from your numpy profile
                tgt = torch.tensor(tpc_targets[p],
                                   device=device,
                                   dtype=tpc_pred.dtype)
                L   = min(tpc_pred.shape[0], tgt.shape[0])
                loss_tpc = loss_tpc + F.mse_loss(tpc_pred[:L], tgt[:L])
        else:
            loss_tpc = torch.tensor(0.0, device=device)



# -----------
        # total loss & update

        total_loss = (
            loss_sds
            + loss_m1 * w_m1
            + loss_m2 * w_m2
            + loss_sa * sa_weight
            + loss_rd * rd_weight
            + loss_tpc * tpc_weight
        )



        optimizer.zero_grad()
      
        total_loss.backward()
   
        optimizer.step()

        with torch.no_grad():
            recon = vae.decode(latent_s).clamp(0,1)    # [H3,W3]
            recon = vae.encode(recon)
            recon = vae.decode(latent_s).clamp(0,1)[0,0] 
            if axis==0:
                vol_cpu[idx] = recon
            elif axis==1:
                vol_cpu[:, idx, :] = recon
            else:
                vol_cpu[:, :, idx] = recon

        torch.cuda.empty_cache()

        if iter_sds% save_interval_vol ==0:

            with torch.no_grad():

                vol_sds = vol_cpu    

                X = vol_cpu.unsqueeze(1).to(device)
                Z, C, H3, W3 = X.shape
                for _ in tqdm(range(refinement_K),desc="Three-axis latent refinement (after sds)"):
                    new_X = torch.zeros_like(X)
                    # 1) depth slices
                    for z in range(Z):
                        img = X[z:z+1]                 # [1,1,64,64]
                        img_in = img*2 - 1             # [-1,1]
                        mu,_ = vae.encode(img_in)
                        dec = vae.decode(mu)[0,0]      # [64,64]
                        new_X[z,0] += dec              # [64,64] into depth plane
                    # 2) height slices
                    for y in range(H3):
                        sl = X[:,0,y,:]               # [Z, W3] = [64,64]
                        img = sl.unsqueeze(0).unsqueeze(0)  # [1,1,64,64]
                        img_in = img*2 - 1
                        mu,_ = vae.encode(img_in)
                        dec = vae.decode(mu)[0,0]     # [64,64]
                        new_X[:,0,y,:] += dec         # scatter across depths
                    # 3) width slices
                    for w in range(W3):
                        sl = X[:,0, :,w].squeeze(1)   # [Z, H3] = [64,64]
                        img = sl.unsqueeze(0).unsqueeze(0)
                        img_in = img*2 - 1
                        mu,_ = vae.encode(img_in)
                        dec = vae.decode(mu)[0,0]     # [64,64]
                        new_X[:,0,:,w] += dec         # scatter across depths
                    X = (new_X/3.0).clamp(0,1)
                vol_cpu_save = X.squeeze(1).cpu()  # [Z,H,W]

                t1, t2 = 0.33, 0.67
                # final threshold
                vol_q_save = torch.where(vol_cpu_save <= t1,
                                    0.0,
                                    torch.where(vol_cpu_save <= t2,
                                                0.5,
                                                1.0))

                # save the raw TIFF stack
                tifffile.imwrite(os.path.join(save_dir, f"volume_{iter_sds:06d}.tiff"), vol_q_save.numpy())
                print(f"Saved 3D volume to {save_dir}/volume+{iter_sds:06d}.tiff")


    vol_sds = vol_cpu    

    # 4) Last refinement
    with torch.no_grad():
        X = vol_cpu.unsqueeze(1).to(device)
        Z, C, H3, W3 = X.shape
        for _ in tqdm(range(refinement_K),desc="Three-axis latent refinement (after sds)"):
            new_X = torch.zeros_like(X)
            # 1) depth slices
            for z in range(Z):
                img = X[z:z+1]                 # [1,1,64,64]
                img_in = img*2 - 1             # [-1,1]
                mu,_ = vae.encode(img_in)
                dec = vae.decode(mu)[0,0]      # [64,64]
                new_X[z,0] += dec              # [64,64] into depth plane
            # 2) height slices
            for y in range(H3):
                sl = X[:,0,y,:]               # [Z, W3] = [64,64]
                img = sl.unsqueeze(0).unsqueeze(0)  # [1,1,64,64]
                img_in = img*2 - 1
                mu,_ = vae.encode(img_in)
                dec = vae.decode(mu)[0,0]     # [64,64]
                new_X[:,0,y,:] += dec         # scatter across depths
            # 3) width slices
            for w in range(W3):
                sl = X[:,0, :,w].squeeze(1)   # [Z, H3] = [64,64]
                img = sl.unsqueeze(0).unsqueeze(0)
                img_in = img*2 - 1
                mu,_ = vae.encode(img_in)
                dec = vae.decode(mu)[0,0]     # [64,64]
                new_X[:,0,:,w] += dec         # scatter across depths
            X = (new_X/3.0).clamp(0,1)
        vol_cpu = X.squeeze(1).cpu()  # [Z,H,W]

    t1, t2 = 0.33, 0.67
    # final threshold
    vol_q = torch.where(vol_cpu <= t1,
                        0.0,
                        torch.where(vol_cpu <= t2,
                                    0.5,
                                    1.0))

    # -----------------------------

    return vol_q.numpy(), decoded_3d.squeeze(1).detach().cpu().numpy(), vol_sds.numpy()

# ----------------------------------------
# Score Distillation Sampling Inference
# ----------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--vae_ckpt', required=True)
    parser.add_argument('--unet_ckpt', required=True)
    parser.add_argument('--init_image', type=str, default=None,
                        help='Path to image for VAE-based latent init; if omitted, uses random init')
    parser.add_argument('--save_dir', default='./sds_ldm_samples')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--latent_ch', type=int, default=4)
    parser.add_argument('--H', type=int, default=16)
    parser.add_argument('--W', type=int, default=16)
    parser.add_argument('--sds_steps', type=int, default=500)
    parser.add_argument('--sds_lr', type=float, default=1e-3)
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--t_min', type=int, default=50)
    parser.add_argument('--t_max', type=int, default=950)
    parser.add_argument('--target_vf', type=float, default=None,
                        help='Target volume fraction guidance')
    parser.add_argument('--vf_weight', type=float, default=1.0,
                        help='Weight for VF guidance')
    parser.add_argument('--training_tpc', required=False,
                        help="path to training autocorr .npz (keys S00_mean,S11_mean,S22_mean)")
    parser.add_argument('--n_volumes', type=int, default=1,
                        help="how many independent 3D samples to generate")
    parser.add_argument('--tpc_weight', type=float, default=0.0,
                        help="tpc weights")
    parser.add_argument('--vf0', type=float, default=0.205219,
                        help="target vf0")
    parser.add_argument('--vf05', type=float, default=0.341594,
                        help="target vf05")
    parser.add_argument('--vf1', type=float, default=0.453187,
                        help="target vf1")
    parser.add_argument('--w_m1', type=float, default=0.0,
                        help="m1 weight")
    parser.add_argument('--w_m2', type=float, default=0.0,
                        help="m2 weight")
    parser.add_argument('--sa_targets', type=str, default=None,
                        help="comma-separated phase:target pairs for SA, e.g. '0:0.2,1:0.5,2:0.3'")
    parser.add_argument('--sa_weight',  type=float, default=0.0,
                        help="weight for surface-area guidance")    
    parser.add_argument('--rd_targets', type=str, default=None,
                        help="comma-separated phase:target pairs for RD, e.g. '0:0.2,1:0.5,2:0.3'")   
    parser.add_argument('--rd_weight',  type=float, default=0.0,
                        help="weight for diffusivity guidance")  
    parser.add_argument('--save_interval_vol',  type=int, default=100,
                        help="save interavl for saving the intermediate volume")                                                   

    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    # Load & freeze VAE
    vae = CustomVAE(latent_ch=args.latent_ch).to(device)
    ckpt = torch.load(args.vae_ckpt, map_location=device)
    vae.load_state_dict(ckpt['vae'])
    vae.eval(); [p.requires_grad_(False) for p in vae.parameters()]
    # Load & freeze UNet
    unet = TimeUNet(args.latent_ch).to(device)
    unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    unet.eval(); [p.requires_grad_(False) for p in unet.parameters()]
    # DDPM scheduler
    ddpm = DDPM(timesteps=1000, beta_start=1e-4, beta_end=2e-2, device=device)


    if args.training_tpc:
        train_data = np.load(args.training_tpc)
        phases = sorted([int(k[1]) for k in train_data if k.endswith('_mean')])
        train_means = { p: train_data[f"S{p}{p}_mean"] for p in phases }
    else:
        train_means = None
        phases = [0, 1, 2]


    detailed = []
    all_vol_profiles = []  # will be list of dicts { phase_idx: 1D numpy profile }

    detailed_dec = []
    all_vol_profiles_dec = []  # will be list of dicts { phase_idx: 1D numpy profile }


    if args.sa_targets:
        sa_targets = {
            int(p): float(v)
            for p,v in (pair.split(':') for pair in args.sa_targets.split(','))
        }
    else:
        sa_targets = None

    if args.rd_targets:
        rd_targets = {
            int(p): float(v)
            for p,v in (pair.split(':') for pair in args.rd_targets.split(','))
        }
    else:
        rd_targets = None


    all_sa = {p: [] for p in phases}

    # Loop to generate many volumes
    for iv in range(args.n_volumes):
        print(f"\n=== Generating volume {iv+1}/{args.n_volumes} ===")
        vol_q, decoded_3D, vol_sds = generate_single_volume(
            vae=vae,
            unet=unet,
            ddpm=ddpm,
            device=device,
            init_image_path=args.init_image,
            num_samples=args.num_samples,
            latent_ch=args.latent_ch,
            H=args.H,
            W=args.W,
            sds_steps=args.sds_steps,
            sds_lr=args.sds_lr,
            t_min=args.t_min,
            t_max=args.t_max,
            refinement_K=5,  # or expose as an argument
            tpc_targets=train_means,
            phases = phases,
            tpc_weight=args.tpc_weight,
            save_dir = args.save_dir,
            vf0 = args.vf0,
            vf05 = args.vf05,
            vf1 = args.vf1,
            w_m1 = args.w_m1,
            w_m2 = args.w_m2,
            sa_targets=sa_targets,        # ← new
            sa_weight=args.sa_weight,     # ← new
            rd_targets = rd_targets,
            rd_weight = args.rd_weight,
            save_interval_vol = args.save_interval_vol             

        )

        # save the TIFF stack
        out_dir = os.path.join(args.save_dir, f"volume_{iv:03d}")
        os.makedirs(out_dir, exist_ok=True)
        tifffile.imwrite(os.path.join(out_dir, "volume.tiff"), vol_q)
        print(f"Saved 3D volume to {out_dir}/volume.tiff")
 

if __name__ == '__main__':
    main()