"""
MicroLAD: Inverse-Controlled 2D-to-3D Microstructure Generation

Main generation script
"""
import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
import tifffile
import random
from tqdm import tqdm

# Import from package
from models import CustomVAE, TimeUNet, DDPM, TorchFEMMesh
from losses import compute_tpc_loss_ste, setup_tpc_bins, compute_vf_loss, compute_sa_loss, compute_diffusivity_loss
from utils import visualize_3d_microstructure, plot_tpc_comparison, three_axis_refinement, compute_volume_tpc


def generate_single_volume(vae, unet, ddpm, fem_solver, bin_mat, bin_counts, 
                           tpc_targets, sa_targets, rd_targets, phases, args, device):
    """Generate one 64×64×64 microstructure volume"""
    
    # Determine active objectives
    active = ["SDS"]
    if args.vf_weight > 0: active.append("VF")
    if args.tpc_weight > 0: active.append("TPC")
    if args.rd_weight > 0: active.append("RD")
    if args.sa_weight > 0: active.append("SA")
    progress_msg = f"3D SDS refine with {'+'.join(active)}"
    
    # 1) Initialize random latent
    x = torch.randn(args.num_samples, args.latent_ch, args.H, args.W, device=device)
    
    # 2) DDPM denoising
    for t in tqdm(reversed(range(ddpm.num_timesteps)), desc="DDPM denoise"):
        ts = torch.full((args.num_samples,), t, dtype=torch.long, device=device)
        with torch.no_grad():
            x = ddpm.p_sample(unet, x, ts)
        if t > 0:
            x = x.transpose(0, 2).transpose(3, 0)
    
    # 3) Multi-axis decode
    with torch.no_grad():
        latent = x.permute(1, 0, 2, 3).contiguous()
        C, D, H_, W_ = latent.shape
        decoded_acc = torch.zeros((1, 64, 64, 64), device=device)
        
        total_decodes = D + H_ + W_
        pbar = tqdm(total=total_decodes, desc="Multi-axis decode", leave=False)
        
        for d in range(D):
            dec = vae.decode(latent[:, d, :, :].unsqueeze(0)).squeeze(0)
            decoded_acc[:, d*4:(d+1)*4, :, :] += dec.unsqueeze(1)
            pbar.update(1)
        
        for h in range(H_):
            dec = vae.decode(latent[:, :, h, :].unsqueeze(0)).squeeze(0)
            decoded_acc[:, :, h*4:(h+1)*4, :] += dec.unsqueeze(2)
            pbar.update(1)
        
        for w in range(W_):
            dec = vae.decode(latent[:, :, :, w].unsqueeze(0)).squeeze(0)
            decoded_acc[:, :, :, w*4:(w+1)*4] += dec.unsqueeze(3)
            pbar.update(1)
        
        pbar.close()
        decoded_3d = (decoded_acc / 3.0).permute(1, 0, 2, 3).clamp(0, 1)
    
    # 4) Three-axis latent refinement
    X = three_axis_refinement(decoded_3d.clone(), vae, args.refinement_K, device)
    
    # 5) SDS optimization
    vol_cpu = X.squeeze(1).cpu()
    Z, H, W = vol_cpu.shape
    
    pbar_sds = tqdm(range(args.sds_steps), desc=progress_msg)
    for iter_sds in pbar_sds:
        # Random slice selection
        axis = random.choice([0, 1, 2])
        idx = random.randrange(Z)
        
        if axis == 0: sl = vol_cpu[idx]
        elif axis == 1: sl = vol_cpu[:, idx, :]
        else: sl = vol_cpu[:, :, idx]
        
        img = sl.unsqueeze(0).unsqueeze(0).to(device)
        mu, _ = vae.encode(img * 2 - 1)
        latent_s = mu.detach().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([latent_s], lr=args.sds_lr)
        
        # SDS loss
        t = torch.randint(args.t_min, args.t_max, (1,), device=device)
        noise = torch.randn_like(latent_s)
        a = ddpm.sqrt_acp[t].view(1, 1, 1, 1)
        s = ddpm.sqrt_om_acp[t].view(1, 1, 1, 1)
        x_t = a * latent_s + s * noise
        
        with torch.no_grad():
            pred = unet(x_t, t)
        
        w = s.pow(2)
        target = noise - pred.detach()
        loss_sds = (w * (latent_s * target)).mean()
        
        # Decode for other losses
        decoded = vae.decode(latent_s).clamp(0, 1)
        
        # Volume fraction
        if args.vf_weight > 0:
            loss_m1, loss_m2, m1_actual = compute_vf_loss(
                decoded, args.vf0, args.vf05, args.vf1, 
                args.vf_weight, args.vf_weight, device
            )
        else:
            loss_m1 = torch.tensor(0., device=device)
            loss_m2 = torch.tensor(0., device=device)
            m1_actual = 0.0
        
        # Soft phase assignment
        P = len(phases)
        x = decoded.repeat(1, P, 1, 1)
        levels = torch.linspace(0, 1, P, device=device)[None, :, None, None]
        dist = (x - levels).abs().view(1, P, -1)
        masks = F.softmax(-30.0 * dist, dim=1).view(1, P, H, W)
        
        # TPC loss
        if args.tpc_weight > 0:
            loss_tpc = compute_tpc_loss_ste(masks[0], phases, tpc_targets, bin_mat, bin_counts, device)
        else:
            loss_tpc = torch.tensor(0., device=device)
        
        # RD loss
        if args.rd_weight > 0 and rd_targets and fem_solver:
            loss_rd = compute_diffusivity_loss(masks, fem_solver, rd_targets, phases, device)
        else:
            loss_rd = torch.tensor(0., device=device)
        
        # SA loss
        if args.sa_weight > 0 and sa_targets:
            loss_sa = compute_sa_loss(decoded, sa_targets, phases, device)
        else:
            loss_sa = torch.tensor(0., device=device)
        
        # Total loss
        total_loss = (
            loss_sds +
            args.w_m1 * loss_m1 +
            args.w_m2 * loss_m2 +
            args.tpc_weight * loss_tpc +
            args.rd_weight * loss_rd +
            args.sa_weight * loss_sa
        )
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        # Update progress bar
        if iter_sds % 100 == 0:
            postfix_dict = {'SDS': f'{loss_sds.item():.4f}', 'Total': f'{total_loss.item():.1f}'}
            if args.tpc_weight > 0:
                postfix_dict['TPC'] = f'{loss_tpc.item():.4f}'
            if args.vf_weight > 0:
                t_mean = 0.0*args.vf0 + 0.5*args.vf05 + 1.0*args.vf1
                postfix_dict['VF_err'] = f'{abs(m1_actual-t_mean):.4f}'
            if args.rd_weight > 0:
                postfix_dict['RD'] = f'{loss_rd.item():.4f}'
            if args.sa_weight > 0:
                postfix_dict['SA'] = f'{loss_sa.item():.4f}'
            pbar_sds.set_postfix(postfix_dict)
        
        # Update volume
        with torch.no_grad():
            recon = vae.decode(latent_s).clamp(0, 1)[0, 0]
            if axis == 0: vol_cpu[idx] = recon
            elif axis == 1: vol_cpu[:, idx, :] = recon
            else: vol_cpu[:, :, idx] = recon
        
        torch.cuda.empty_cache()
    
    # Final refinement
    vol_refined = three_axis_refinement(vol_cpu.unsqueeze(1).to(device), vae, args.refinement_K, device, desc="Final refinement")
    vol_cpu = vol_refined.squeeze(1).cpu()
    
    # Threshold to 3 phases
    vol_q = torch.where(vol_cpu <= 0.33, 0.0, torch.where(vol_cpu <= 0.67, 0.5, 1.0))
    
    return vol_q.numpy()


def main():
    parser = argparse.ArgumentParser(description='MicroLAD: 3D Microstructure Generation')
    
    # Required
    parser.add_argument('--vae_ckpt', required=True, help='VAE checkpoint path')
    parser.add_argument('--unet_ckpt', required=True, help='UNet checkpoint path')
    parser.add_argument('--training_tpc', required=True, help='Training TPC .npz file')
    
    # Output
    parser.add_argument('--save_dir', default='./output', help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--n_volumes', type=int, default=1, help='Number of volumes')
    
    # Model parameters
    parser.add_argument('--num_samples', type=int, default=16)
    parser.add_argument('--latent_ch', type=int, default=4)
    parser.add_argument('--H', type=int, default=16)
    parser.add_argument('--W', type=int, default=16)
    
    # SDS parameters
    parser.add_argument('--sds_steps', type=int, default=6000, help='SDS optimization steps')
    parser.add_argument('--sds_lr', type=float, default=0.001, help='SDS learning rate')
    parser.add_argument('--t_min', type=int, default=200)
    parser.add_argument('--t_max', type=int, default=500)
    parser.add_argument('--refinement_K', type=int, default=5, help='Refinement iterations')
    
    # Loss weights
    parser.add_argument('--vf_weight', type=float, default=100000, help='Volume fraction weight')
    parser.add_argument('--tpc_weight', type=float, default=10000, help='TPC weight')
    parser.add_argument('--rd_weight', type=float, default=0.0, help='Diffusivity weight')
    parser.add_argument('--sa_weight', type=float, default=0.0, help='Surface area weight')
    
    # Targets
    parser.add_argument('--vf_targets', type=str, default="0:0.35,0.5:0.28,1:0.37", 
                        help='VF targets: "0:0.35,0.5:0.28,1:0.37"')
    parser.add_argument('--rd_targets', type=str, default=None, help='RD targets: "0:0.1,1:0.5,2:0.8"')
    parser.add_argument('--sa_targets', type=str, default=None, help='SA targets: "0:1.0,1:0.5,2:0.3"')
    
    args = parser.parse_args()
    
    # Parse VF targets from string format
    vf_dict = {float(k): float(v) for k, v in (pair.split(':') for pair in args.vf_targets.split(','))}
    args.vf0 = vf_dict.get(0.0, 0.35)
    args.vf05 = vf_dict.get(0.5, 0.28)
    args.vf1 = vf_dict.get(1.0, 0.37)
    
    # Set w_m1 and w_m2 from vf_weight
    args.w_m1 = args.vf_weight
    args.w_m2 = args.vf_weight
    
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load models
    print("Loading models...")
    vae = CustomVAE(latent_ch=args.latent_ch).to(device)
    vae.load_state_dict(torch.load(args.vae_ckpt, map_location=device)['vae'])
    vae.eval()
    [p.requires_grad_(False) for p in vae.parameters()]
    
    unet = TimeUNet(args.latent_ch).to(device)
    unet.load_state_dict(torch.load(args.unet_ckpt, map_location=device))
    unet.eval()
    [p.requires_grad_(False) for p in unet.parameters()]
    
    ddpm = DDPM(timesteps=1000, beta_start=1e-4, beta_end=2e-2, device=device)
    
    # Load TPC targets
    train_data = np.load(args.training_tpc)
    phases = sorted([int(k[1]) for k in train_data if k.endswith('_mean')])
    tpc_targets = {p: train_data[f"S{p}{p}_mean"] for p in phases}
    
    # Parse other targets
    rd_targets = None
    if args.rd_targets:
        rd_targets = {int(p): float(v) for p, v in (pair.split(':') for pair in args.rd_targets.split(','))}
    
    sa_targets = None
    if args.sa_targets:
        sa_targets = {int(p): float(v) for p, v in (pair.split(':') for pair in args.sa_targets.split(','))}
    
    # Setup FEM solver
    fem_solver = None
    if args.rd_weight > 0:
        fem_solver = TorchFEMMesh(M=64, N=64, low_cond=0.001, device=device).to(device)
    
    # Setup TPC bins
    bin_mat, bin_counts = setup_tpc_bins(64, 64, device)
    
    # Print configuration
    print(f"\nGenerating {args.n_volumes} volumes...")
    active_objectives = ['SDS']
    if args.vf_weight > 0:
        active_objectives.append('VF')
    if args.tpc_weight > 0:
        active_objectives.append('TPC')
    if args.rd_weight > 0:
        active_objectives.append('RD')
    if args.sa_weight > 0:
        active_objectives.append('SA')
    print(f"Active objectives: {' + '.join(active_objectives)}")
    print(f"VF targets: Phase 0={args.vf0:.3f}, Phase 0.5={args.vf05:.3f}, Phase 1={args.vf1:.3f}\n")
    
    # Generate volumes
    for iv in range(args.n_volumes):
        print(f"\n{'='*80}")
        print(f"Generating volume {iv+1}/{args.n_volumes}")
        print(f"{'='*80}")
        
        vol_q = generate_single_volume(
            vae, unet, ddpm, fem_solver, bin_mat, bin_counts,
            tpc_targets, sa_targets, rd_targets, phases, args, device
        )
        
        # Create output directory
        out_dir = os.path.join(args.save_dir, f"volume_{iv:03d}")
        os.makedirs(out_dir, exist_ok=True)
        
        # Save volume
        tifffile.imwrite(os.path.join(out_dir, "volume.tiff"), vol_q)
        print(f"Saved volume to {out_dir}/volume.tiff")
        
        # Save 3D visualization
        vis_path = os.path.join(out_dir, "3d_visualization_final.png")
        visualize_3d_microstructure(vol_q, vis_path)
        print(f"Saved 3D visualization")
        
        # Compute and save TPC plots
        vol_lbl = (vol_q * 2).astype(np.int32)
        tpc_results = compute_volume_tpc(vol_lbl, phases)
        
        for axis_name in ['x', 'y', 'z']:
            axis_tpcs = {p: tpc_results[(axis_name, p)] for p in phases}
            save_path = os.path.join(out_dir, f"tpc_compare_{axis_name}.png")
            plot_tpc_comparison(axis_tpcs, tpc_targets, phases, save_path, axis_name)
        
        print(f"Saved TPC comparison plots")
        print(f"Output: {out_dir}")
    
    print(f"\n{'='*80}")
    print(f"Generation complete! Results in: {args.save_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()


