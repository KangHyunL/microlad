"""
Differentiable Finite Element Method (FEM) solver for effective diffusivity
"""
import numpy as np
import torch
import torch.nn as nn


class TorchFEMMesh(nn.Module):
    """
    Differentiable 2D FEM solver for steady-state diffusion
    
    Solves: ∇·(σ∇u) = 0
    BC: u=1 (left), u=0 (right), ∂u/∂n=0 (top/bottom)
    
    Computes effective diffusivity for microstructure characterization
    """
    def __init__(self, M: int, N: int, low_cond: float = 0.001, device='cpu'):
        """
        Args:
            M: Number of cells in y-direction
            N: Number of cells in x-direction
            low_cond: Conductivity for void phase
            device: torch device
        """
        super().__init__()
        self.M, self.N = M, N
        self.nx, self.ny = N + 1, M + 1
        self.nn = self.nx * self.ny  # Total nodes
        self.low_cond = low_cond
        self.device = device

        # Reference shape function gradients for linear triangle
        Dhat = np.array([[-1,-1],[1,0],[0,1]], dtype=float)

        # Build element connectivity (2 triangles per quadrilateral cell)
        # The original paper used a rectangular mesh, but for simplicity, a triangular mesh version is provided here.
        # The rectangular mesh version and a more efficient FEM solver will be added soon.

        
        elems = []
        for j in range(M):
            for i in range(N):
                n0 = j*self.nx + i
                n1 = n0 + 1
                n2 = n0 + self.nx
                n3 = n2 + 1
                # Triangle 1: (n0, n1, n2)
                # Triangle 2: (n1, n3, n2)
                elems.append((n0, n1, n2))
                elems.append((n1, n3, n2))
        
        elems = np.array(elems, dtype=np.int64)
        self.register_buffer('elems', torch.from_numpy(elems))

        # Assemble base stiffness matrix (geometry only, no material)
        rows, cols, base_data = [], [], []
        elem_idx = []
        gradT_list = []
        area_list = []
        
        for idx, e in enumerate(elems):
            # Node coordinates
            pts = np.array([[e[k] % self.nx, e[k]//self.nx] for k in range(3)], float)
            
            # Jacobian and shape function gradients
            J = np.vstack((pts[1]-pts[0], pts[2]-pts[0])).T
            detJ = np.linalg.det(J)
            invJT = np.linalg.inv(J).T
            grads = Dhat.dot(invJT)  # Physical space gradients
            gradT = grads.T
            area = abs(detJ) / 2.0
            
            # Element stiffness (without material property)
            ke0 = area * (grads.dot(grads.T))
            
            # Store for sparse assembly
            for a in range(3):
                for b in range(3):
                    rows.append(e[a])
                    cols.append(e[b])
                    base_data.append(ke0[a,b])
                    elem_idx.append(idx)
            
            gradT_list.append(gradT)
            area_list.append(area)

        # Register as buffers (will move with .to(device))
        self.register_buffer('rows', torch.LongTensor(rows))
        self.register_buffer('cols', torch.LongTensor(cols))
        self.register_buffer('base_data', torch.tensor(base_data, dtype=torch.float32))
        self.register_buffer('elem_idx', torch.LongTensor(elem_idx))
        self.register_buffer('elem_gradT', torch.tensor(np.stack(gradT_list), dtype=torch.float32))
        self.register_buffer('elem_area', torch.tensor(area_list, dtype=torch.float32))

        # Dirichlet boundary conditions: left=1, right=0
        bc = []
        for j in range(self.ny):
            bc.append(j*self.nx + 0)              # Left boundary
            bc.append(j*self.nx + (self.nx-1))    # Right boundary
        
        bc_idx = np.unique(bc)
        fc_idx = np.setdiff1d(np.arange(self.nn), bc_idx)  # Free DOFs
        
        self.register_buffer('bc_idx', torch.LongTensor(bc_idx))
        self.register_buffer('fc_idx', torch.LongTensor(fc_idx))
        
        # BC values
        u_c = np.zeros(len(bc_idx), dtype=np.float32)
        coords = np.stack([bc_idx % self.nx, bc_idx // self.nx], -1)
        u_c[coords[:,0] == 0] = 1.0           # Left = 1
        u_c[coords[:,0] == self.nx-1] = 0.0   # Right = 0
        self.register_buffer('u_c', torch.tensor(u_c))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Solve FEM problem and compute effective diffusivity
        
        Args:
            mask: [H, W] conductivity map in [0,1]
        Returns:
            D_eff: scalar effective diffusivity
        """
        # Map mask to element conductivities
        m_flat = mask.reshape(-1)  # [M*N]
        sigma_e = self.low_cond + (1.0 - self.low_cond) * m_flat.repeat_interleave(2)  # [n_elems]
        
        # Assemble global stiffness with material properties
        data = self.base_data * sigma_e[self.elem_idx]
        K = torch.sparse_coo_tensor(
            torch.stack([self.rows, self.cols], 0),
            data,
            (self.nn, self.nn),
            device=self.device
        )
        
        # Convert to dense for BC application
        K_dense = K.to_dense()
        bc, fc = self.bc_idx, self.fc_idx
        K_ff = K_dense[fc][:, fc]   # Free-free partition
        K_fc = K_dense[fc][:, bc]   # Free-constrained partition
        b_f  = - K_fc @ self.u_c    # RHS with BC
        
        # Solve linear system (differentiable!)
        u_f  = torch.linalg.solve(K_ff, b_f)
        
        # Build full solution vector
        u = torch.zeros(self.nn, device=self.device)
        u[fc] = u_f
        u[bc] = self.u_c
        
        # Compute effective diffusivity from energy
        u_e = u[self.elems]  # Element nodal values
        grad_u = torch.matmul(self.elem_gradT, u_e.unsqueeze(-1)).squeeze(-1)  # [n_elems, 2]
        sq = (grad_u**2).sum(dim=1)  # |∇u|²
        
        # D_eff = ∫ σ |∇u|² dΩ
        Deff = (sigma_e * sq * self.elem_area).sum()
        
        return Deff


