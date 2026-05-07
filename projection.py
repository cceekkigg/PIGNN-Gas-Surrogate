
# ============================

import hashlib

import torch


def _resolve_dtype(name: str):
    name = str(name).lower()
    if name in {"float64", "fp64", "double"}:
        return torch.float64
    return torch.float32


class _FixedTopologyDirectSolverCache:
    """
    Caches a reduced Laplacian factorization for batches made of repeated copies
    of the same graph topology. Falls back cleanly when the batch layout is not
    compatible with the fast path.
    """

    def __init__(self, damping: float = 0.0):
        self.damping = float(damping)
        self.signature = None
        self.ready = False
        self.B = None
        self.B_t = None
        self.free_idx = None
        self.chol = None
        self.num_nodes = 0
        self.num_edges = 0

    def _signature_for(self, edge_index_local, slack_local, dtype):
        edge_bytes = edge_index_local.detach().cpu().numpy().tobytes()
        slack_bytes = slack_local.detach().to(torch.uint8).cpu().numpy().tobytes()
        topo_hash = hashlib.sha1(edge_bytes + slack_bytes).hexdigest()
        return (
            str(edge_index_local.device),
            str(dtype),
            self.damping,
            int(edge_index_local.size(1)),
            int(slack_local.numel()),
            topo_hash,
        )

    def prepare(self, edge_index, is_slack_mask, graph_ptr, dtype):
        if graph_ptr is None:
            return False

        ptr = graph_ptr.view(-1)
        if ptr.numel() < 2:
            return False

        num_graphs = int(ptr.numel() - 1)
        node_counts = ptr[1:] - ptr[:-1]
        if int(node_counts.min().item()) != int(node_counts.max().item()):
            return False

        nodes_per_graph = int(node_counts[0].item())
        if nodes_per_graph <= 0:
            return False

        total_edges = int(edge_index.size(1))
        if num_graphs <= 0 or total_edges % num_graphs != 0:
            return False

        edges_per_graph = total_edges // num_graphs
        edge_index_local = edge_index[:, :edges_per_graph] - int(ptr[0].item())
        if edge_index_local.numel() == 0:
            return False

        slack_flat = is_slack_mask.view(-1)
        slack_local = slack_flat[:nodes_per_graph]

        checks = min(num_graphs, 3)
        for graph_id in range(1, checks):
            node_start = int(ptr[graph_id].item())
            node_end = int(ptr[graph_id + 1].item())
            edge_start = graph_id * edges_per_graph
            edge_end = edge_start + edges_per_graph
            local_edges = edge_index[:, edge_start:edge_end] - node_start
            if not torch.equal(local_edges, edge_index_local):
                return False
            if not torch.equal(slack_flat[node_start:node_end], slack_local):
                return False

        signature = self._signature_for(edge_index_local, slack_local, dtype)
        if self.ready and signature == self.signature:
            return True

        device = edge_index.device
        cols = torch.arange(edges_per_graph, device=device)
        u = edge_index_local[0].to(device)
        v = edge_index_local[1].to(device)

        B = torch.zeros((nodes_per_graph, edges_per_graph), dtype=dtype, device=device)
        B[v, cols] = 1.0
        B[u, cols] -= 1.0

        free_idx = (slack_local < 0.5).nonzero(as_tuple=False).view(-1)
        if free_idx.numel() == 0:
            self.ready = False
            return False

        laplacian = B @ B.transpose(0, 1)
        if self.damping > 0.0:
            laplacian = laplacian + self.damping * torch.eye(
                nodes_per_graph, dtype=dtype, device=device
            )
        laplacian_free = laplacian.index_select(0, free_idx).index_select(1, free_idx)
        chol, info = torch.linalg.cholesky_ex(laplacian_free)
        if int(info.item()) != 0:
            self.ready = False
            return False

        self.signature = signature
        self.ready = True
        self.B = B
        self.B_t = B.transpose(0, 1).contiguous()
        self.free_idx = free_idx
        self.chol = chol
        self.num_nodes = nodes_per_graph
        self.num_edges = edges_per_graph
        return True

    def apply_node_accumulation(self, edge_batch):
        return edge_batch @ self.B_t

    def apply_edge_gradient(self, node_batch):
        return node_batch @ self.B

    def solve_free(self, rhs_free):
        solved_t = torch.cholesky_solve(rhs_free.transpose(0, 1).contiguous(), self.chol)
        return solved_t.transpose(0, 1).contiguous()


class MassBalanceProjection:
    """
    Enforces strict nodal mass balance via differentiable projection.

        min ||Q* - q_unprojected||^2  s.t.  A Q* = d

    with KKT solution:
        Q* = q_unprojected - A^T λ
        (A A^T) λ = A q_unprojected - d

    Implementation details:
      - Uses implicit incidence operators via index_add_ (no explicit sparse matrices).
      - Solves (A A^T) λ = b by Conjugate Gradient (CG) on the free-node subspace.
      - Slack node(s) are masked out (λ_slack = 0).
    """

    def __init__(self, max_iter=1000, tolerance=1e-6, solve_dtype="float64", use_direct_cache=True):
        self.max_iter = int(max_iter)
        self.tol = float(tolerance)
        self.solve_dtype = _resolve_dtype(solve_dtype)
        self.use_direct_cache = bool(use_direct_cache)
        self._direct_cache = _FixedTopologyDirectSolverCache(damping=0.0)
    
    def __call__(self, q_unprojected, edge_index, demand, is_slack_mask, graph_ptr=None):
        """
        Args:
            q_unprojected: [E, 1] float32/float64
            edge_index:    [2, E] long
            demand:        [N, 1] float32/float64
            is_slack_mask: [N] or [N,1] float/bool, 1=slack

        Returns:
            q_projected:   [E, 1] (same dtype as q_unprojected input)
            residual_pre:  float
            residual_post: float
        """
        
        device = q_unprojected.device
        orig_dtype = q_unprojected.dtype
        dtype = self.solve_dtype

        q_unprojected = q_unprojected.to(dtype)
        if demand.dim() == 1:
            demand = demand.unsqueeze(1)
        demand = demand.to(dtype)
        demand_rhs = 2.0 * demand

        if is_slack_mask.dim() == 1:
            is_slack_mask = is_slack_mask.unsqueeze(1)
        is_slack_mask = is_slack_mask.to(dtype)

        if self.use_direct_cache and self._direct_cache.prepare(
            edge_index=edge_index,
            is_slack_mask=is_slack_mask,
            graph_ptr=graph_ptr,
            dtype=dtype,
        ):
            num_graphs = int(graph_ptr.numel() - 1)
            num_nodes = self._direct_cache.num_nodes
            num_edges = self._direct_cache.num_edges

            q_batch = q_unprojected.reshape(num_graphs, num_edges)
            demand_batch = demand.reshape(num_graphs, num_nodes)
            slack_batch = is_slack_mask.reshape(num_graphs, num_nodes)
            free_mask_batch = 1.0 - slack_batch

            residual_pre_full = (
                self._direct_cache.apply_node_accumulation(q_batch) - 2.0 * demand_batch
            ) * free_mask_batch
            residual_pre = torch.linalg.vector_norm(residual_pre_full).item()

            rhs_free = residual_pre_full.index_select(1, self._direct_cache.free_idx)
            if rhs_free.numel() == 0 or torch.max(torch.abs(rhs_free)).item() < self.tol:
                q_proj_batch = q_batch
            else:
                lam_free = self._direct_cache.solve_free(rhs_free)
                lam_batch = torch.zeros(
                    (num_graphs, num_nodes), dtype=dtype, device=device
                )
                lam_batch.index_copy_(1, self._direct_cache.free_idx, lam_free)
                q_proj_batch = q_batch - self._direct_cache.apply_edge_gradient(lam_batch)

            residual_post_full = (
                self._direct_cache.apply_node_accumulation(q_proj_batch) - 2.0 * demand_batch
            ) * free_mask_batch
            residual_post = torch.linalg.vector_norm(residual_post_full).item()
            return q_proj_batch.reshape(-1, 1).to(orig_dtype), residual_pre, residual_post

        free_mask = (1.0 - is_slack_mask)  # [N,1]
        num_nodes = demand_rhs.size(0)

        u = edge_index[0].to(device)
        v = edge_index[1].to(device)

        def A_op(q):
            # node divergence: sum_in - sum_out
            out = torch.zeros((num_nodes, 1), dtype=dtype, device=device)
            out.index_add_(0, v, q)
            out.index_add_(0, u, -q)
            return out

        def AT_op(lam):
            # edge gradient
            return lam[v] - lam[u]

        def M_op(x):
            # (A A^T) x = A (A^T x)
            return A_op(AT_op(x))

        r_pre = (A_op(q_unprojected) - demand_rhs) * free_mask

        # residual before
        residual_pre = torch.sqrt((r_pre ** 2).sum()).item()
        b = (A_op(q_unprojected) - demand_rhs) * free_mask

        lam = torch.zeros((num_nodes, 1), dtype=dtype, device=device)
        r = (b - M_op(lam)) * free_mask
        p = r.clone()
        rs_old = (r * r).sum()

        if rs_old.item() < (self.tol ** 2):
            q_proj = q_unprojected
        else:
            for _ in range(self.max_iter):
                Mp = M_op(p) * free_mask
                denom = (p * Mp).sum().clamp_min(1e-30)
                alpha = rs_old / denom

                lam = lam + alpha * p
                r = (r - alpha * Mp) * free_mask

                rs_new = (r * r).sum()
                if rs_new.item() < (self.tol ** 2):
                    break

                beta = rs_new / rs_old.clamp_min(1e-30)
                p = r + beta * p
                rs_old = rs_new

            q_proj = q_unprojected - AT_op(lam)

        # residual after
        r_post = (A_op(q_proj) - demand_rhs) * free_mask
        residual_post = torch.sqrt((r_post ** 2).sum()).item()

        # print(residual_pre, residual_post)
        return q_proj.to(orig_dtype), residual_pre, residual_post


class PressureDropIntegration:
    """
    Edge-first pressure model integration:

        predict dp2_hat on directed edges (u->v): dp2 = p2[v] - p2[u]
        then integrate to nodal p2 by solving the constrained least squares:

            min_p2 || B^T p2 - dp2_hat ||^2    s.t. p2_slack = pset2

    Let delta = p2 - pset2. Because pset2 is constant per-graph (broadcast), B^T pset2 = 0:
        B^T p2 = B^T delta

    The normal equations become a Laplacian solve:
        (B B^T) delta = B dp2_hat
    solved on free nodes with delta(slack)=0.
    """

    def __init__(self, max_iter=2000, tolerance=1e-10, damping=0.0, solve_dtype="float64", use_direct_cache=True):
        self.max_iter = int(max_iter)
        self.tol = float(tolerance)
        self.damping = float(damping)
        self.solve_dtype = _resolve_dtype(solve_dtype)
        self.use_direct_cache = bool(use_direct_cache)
        self._direct_cache = _FixedTopologyDirectSolverCache(damping=self.damping)

    def __call__(self, dp2_hat, edge_index, is_slack_mask, pset2_per_node, graph_ptr=None):
        """
        Args:
            dp2_hat:         [E,1] predicted directed dp2 in physical units (bar^2)
            edge_index:      [2,E]
            is_slack_mask:   [N] or [N,1] (1=slack)
            pset2_per_node:  [N,1] per-node slack^2 broadcast for that graph (bar^2)

        Returns:
            p2_pred:  [N,1] = pset2_per_node + delta
            delta:    [N,1] with delta(slack)=0
        """
        device = dp2_hat.device
        orig_dtype = dp2_hat.dtype
        dtype = self.solve_dtype

        dp2_hat = dp2_hat.to(dtype)
        pset2_per_node = pset2_per_node.to(dtype)

        if is_slack_mask.dim() == 1:
            is_slack_mask = is_slack_mask.unsqueeze(1)
        is_slack_mask = is_slack_mask.to(dtype)

        if self.use_direct_cache and self._direct_cache.prepare(
            edge_index=edge_index,
            is_slack_mask=is_slack_mask,
            graph_ptr=graph_ptr,
            dtype=dtype,
        ):
            num_graphs = int(graph_ptr.numel() - 1)
            num_nodes = self._direct_cache.num_nodes
            num_edges = self._direct_cache.num_edges

            dp2_batch = dp2_hat.reshape(num_graphs, num_edges)
            pset2_batch = pset2_per_node.reshape(num_graphs, num_nodes)
            slack_batch = is_slack_mask.reshape(num_graphs, num_nodes)
            free_mask_batch = 1.0 - slack_batch

            rhs_full = self._direct_cache.apply_node_accumulation(dp2_batch) * free_mask_batch
            rhs_free = rhs_full.index_select(1, self._direct_cache.free_idx)

            delta_batch = torch.zeros((num_graphs, num_nodes), dtype=dtype, device=device)
            if rhs_free.numel() > 0 and torch.max(torch.abs(rhs_free)).item() >= self.tol:
                delta_free = self._direct_cache.solve_free(rhs_free)
                delta_batch.index_copy_(1, self._direct_cache.free_idx, delta_free)

            p2_pred = (pset2_batch + delta_batch).reshape(-1, 1).to(orig_dtype)
            return p2_pred, delta_batch.reshape(-1, 1).to(orig_dtype)

        free_mask = (1.0 - is_slack_mask)  # [N,1]
        num_nodes = pset2_per_node.size(0)

        u = edge_index[0].to(device)
        v = edge_index[1].to(device)

        def BT_op(p2):
            # edge drops: p2[v] - p2[u]
            return p2[v] - p2[u]  # [E,1]

        def B_op(edge_vals):
            # node accumulation: +edge at v, -edge at u
            out = torch.zeros((num_nodes, 1), dtype=dtype, device=device)
            out.index_add_(0, v, edge_vals)
            out.index_add_(0, u, -edge_vals)
            return out

        def L_op(x):
            # Laplacian: B (B^T x)
            y = B_op(BT_op(x))
            if self.damping > 0.0:
                y = y + self.damping * x
            return y

        # Solve L delta = B dp2_hat on free nodes, delta(slack)=0
        b = B_op(dp2_hat) * free_mask

        delta = torch.zeros((num_nodes, 1), dtype=dtype, device=device)

        r = (b - L_op(delta)) * free_mask
        p = r.clone()
        rs_old = (r * r).sum()

        if rs_old.item() >= (self.tol ** 2):
            for _ in range(self.max_iter):
                Lp = L_op(p) * free_mask
                denom = (p * Lp).sum().clamp_min(1e-30)
                alpha = rs_old / denom

                delta = delta + alpha * p
                r = (r - alpha * Lp) * free_mask

                rs_new = (r * r).sum()
                if rs_new.item() < (self.tol ** 2):
                    break

                beta = rs_new / rs_old.clamp_min(1e-30)
                p = r + beta * p
                rs_old = rs_new

        p2_pred = (pset2_per_node + delta).to(orig_dtype)
        return p2_pred, delta.to(orig_dtype)


def validate_mass_balance(demand, threshold: float = 1e-1):
    """
    Used by src/dataset.py during preprocessing.

    Checks if nodal demands satisfy network-wide mass conservation:
        sum(demand) ≈ 0  (within threshold)

    Args:
        demand: [N] or [N,1] tensor
        threshold: absolute tolerance on |sum(demand)| in Nm3/s

    Raises:
        ValueError if |sum(demand)| exceeds threshold
    """
    if demand is None:
        raise ValueError("validate_mass_balance: demand is None")

    if demand.dim() == 2 and demand.size(1) == 1:
        s = demand.sum()
    else:
        s = demand.view(-1).sum()

    total_imbalance = torch.abs(s).item()
    if total_imbalance > float(threshold):
        raise ValueError(
            f"Network mass balance violated: |sum(demand)| = {total_imbalance:.6g} Nm3/s "
            f"(threshold={float(threshold):.6g} Nm3/s)."
        )

class MonotonicityCorrection:
    """
    Post-projection monotonicity correction inside the mass-balance manifold.

    This module:
    - Starts from mass-balanced flow q_proj
    - Performs a few small projected-gradient steps to reduce
      q * (p_u - p_v) < 0 violations
    - Re-projects to Aq=d after each step

    IMPORTANT:
    - Runs under torch.no_grad()
    - Does NOT backpropagate gradients
    """

    def __init__(
        self,
        projection: MassBalanceProjection,
        steps: int = 3,
        step_size: float = 0.15,
        beta: float = 0.3,
        q_quantile: float = 0.20,
        dp_quantile: float = 0.20,
        scale_floor: float = 1e-2,
        accept_tol: float = 7e-2,
        backtracking_steps: int = 4,
        require_improvement: bool = True,
        eps: float = 1e-12,
    ):
        self.projection = projection
        self.steps = steps
        self.step_size = step_size
        self.beta = beta
        self.q_quantile = q_quantile
        self.dp_quantile = dp_quantile
        self.scale_floor = scale_floor
        self.accept_tol = accept_tol
        self.backtracking_steps = backtracking_steps
        self.require_improvement = require_improvement
        self.eps = eps

    @torch.no_grad()
    def __call__(self, q_proj, p2_pred, edge_index, rev_edge_id, demand, is_slack):
        """
        Args:
            q_proj:     [E,1] mass-balanced flow
            p2_pred:    [N,1] predicted p^2
            edge_index: [2,E]
            rev_edge_id:[E]
        Returns:
            q_corr:     [E,1] corrected flow (still mass-balanced)
        """
        device = q_proj.device

        # pressure
        p = torch.sqrt(torch.clamp(p2_pred, min=0.0)).view(-1)
        u, v = edge_index
        dp = (p[u] - p[v]).view(-1)

        q = q_proj.view(-1).clone()
        q0 = q.clone()

        E = q.numel()
        eid = torch.arange(E, device=device)
        phys_mask = eid < rev_edge_id
        phys_idx = phys_mask.nonzero(as_tuple=False).view(-1)

        if phys_idx.numel() == 0:
            return q_proj

        dp_phys_all = dp[phys_idx]

        def violation_score(q_vec: torch.Tensor):
            prod_all = q_vec[phys_idx] * dp_phys_all
            margin = torch.relu(-(prod_all + self.accept_tol))
            return margin.mean(), (prod_all < -self.accept_tol).float().mean()

        for _ in range(self.steps):
            q_phys = q[phys_idx]
            dp_phys = dp[phys_idx]

            q_abs = q_phys.abs()
            dp_abs = dp_phys.abs()

            q_th = torch.quantile(q_abs, self.q_quantile)
            dp_th = torch.quantile(dp_abs, self.dp_quantile)
            sel = (q_abs > q_th) & (dp_abs > dp_th)
            prod_phys = q_phys * dp_phys
            sel = sel & (prod_phys < -self.accept_tol)

            if sel.sum() == 0:
                break

            scale = (q_abs[sel] * dp_abs[sel]).median()
            scale = scale.clamp_min(self.scale_floor)

            z = prod_phys[sel] / (scale + self.eps)
            grad_sel = -torch.sigmoid(-z) * (dp_phys[sel] / (scale + self.eps))

            grad = torch.zeros_like(q)
            grad[phys_idx[sel]] = grad_sel

            current_obj, current_rate = violation_score(q)
            step_size = self.step_size
            accepted = False

            for _ in range(max(1, self.backtracking_steps)):
                q_candidate = q - step_size * ((q - q0) + self.beta * grad)

                # enforce antisymmetry
                q_candidate = 0.5 * (q_candidate - q_candidate[rev_edge_id])

                # re-project to mass balance
                q_candidate, _, _ = self.projection(
                    q_candidate.view(-1, 1), edge_index, demand, is_slack
                )
                q_candidate = q_candidate.view(-1)

                if not self.require_improvement:
                    q = q_candidate
                    accepted = True
                    break

                cand_obj, cand_rate = violation_score(q_candidate)
                improves_obj = cand_obj <= current_obj + 1e-12
                improves_rate = cand_rate <= current_rate + 1e-12
                if improves_obj and improves_rate:
                    q = q_candidate
                    accepted = True
                    break

                step_size *= 0.5

            if not accepted:
                break

        return q.view(-1, 1)

