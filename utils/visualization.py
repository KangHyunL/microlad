"""
3D visualization and plotting utilities
"""
import numpy as np
import matplotlib.pyplot as plt

try:
    import pyvista as pv
    PYVISTA_AVAILABLE = True
    pv.global_theme.axes.show = False
except ImportError:
    PYVISTA_AVAILABLE = False


def visualize_3d_microstructure(volume, save_path):
    """
    Create 3D PyVista visualization of microstructure
    
    Args:
        volume: [Z, Y, X] numpy array with values in {0.0, 0.5, 1.0}
        save_path: Path to save PNG screenshot
    """
    if not PYVISTA_AVAILABLE:
        print(f"Warning: PyVista not available. Skipping {save_path}")
        return
    
    # Remap to integer codes: 0.0→0, 0.5→1, 1.0→2
    codes = np.zeros_like(volume, dtype=np.uint8)
    codes[np.isclose(volume, 0.5)] = 1
    codes[np.isclose(volume, 1.0)] = 2
    
    # Create voxel grid
    dims = (np.array(codes.shape) + 1).tolist()
    grid = pv.ImageData(dimensions=dims, spacing=(1, 1, 1), origin=(0, 0, 0))
    grid.cell_data["phase"] = codes.flatten(order="F")
    
    # Setup plotter
    plotter = pv.Plotter(window_size=[800, 800], off_screen=True)
    plotter.background_color = 'white'
    
    # Render each phase with smoothing
    phase_colors = {0: '#444444', 1: 'gray', 2: 'white'}
    for phase, color in phase_colors.items():
        sub = grid.threshold([phase - 0.01, phase + 0.01], scalars="phase")
        surf = sub.extract_surface()
        smooth = surf.smooth(n_iter=50, relaxation_factor=0.1)
        plotter.add_mesh(smooth, color=color, opacity=1.0, show_edges=False)
    
    # Render and save
    plotter.camera_position = 'iso'
    plotter.show(screenshot=save_path, auto_close=True)
    plotter.close()


def plot_tpc_comparison(gen_tpcs, tgt_tpcs, phases, save_path, axis_name):
    """
    Plot TPC comparison for all phases on one axis
    
    Args:
        gen_tpcs: Dict mapping phase -> generated TPC profile
        tgt_tpcs: Dict mapping phase -> target TPC profile  
        phases: List of phase indices
        save_path: Path to save figure
        axis_name: Axis name (x, y, or z)
    """
    plt.figure(figsize=(6, 4))
    
    for p in phases:
        gen_prof = gen_tpcs[p]
        tgt_prof = tgt_tpcs[p]
        L = min(len(gen_prof), len(tgt_prof))
        plt.plot(gen_prof[:L], label=f'Gen p{p}', linewidth=2)
        plt.plot(tgt_prof[:L], '--', label=f'Target p{p}', linewidth=2)
    
    plt.xlabel('r (pixels)', fontsize=11)
    plt.ylabel('S(r)', fontsize=11)
    plt.title(f'{axis_name.upper()}-axis TPC Comparison', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

