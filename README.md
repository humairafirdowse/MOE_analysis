## MoE Theoretical Analysis Tool (Phase 1 UI)

This is a React + TypeScript + Vite frontend for exploring Mixture-of-Experts (MoE) model
theory, designed around the architecture you described.

Phase 1 implements:

- **Global reactive sidebar** with:
  - Model architecture inputs (N, d_model, L, heads, vocab, sequence length)
  - MoE configuration (experts, top-K, shared experts, FFN dims, granularity)
  - Training and inference configs (batch, tokens, precision, parallelism, GPU type)
  - Paper/preset loader (DeepSeek-V3, Mixtral 8x7B, and stubs for others)
- **Tab 1 — Model Overview**:
  - KPI cards for:
    - Total parameters
    - Active parameters per token
    - Active / total ratio
    - Equivalent dense model (same active params)
  - Horizontal stacked bar chart (Recharts) showing:
    - Embedding, Attention, Routed Experts, Shared Experts, Output Head
    - Active params (solid) vs inactive expert capacity (ghosted)
- **Tab 7 — Validation**:
  - Uses the same presets as the sidebar (e.g. DeepSeek-V3)
  - Comparison table for:
    - Total parameters
    - Active parameters
  - Delta (%) and status (✅ PASS / ⚠️ WARN / ❌ FAIL) with 5% / 15% thresholds
  - Accuracy scorecard summary

Other tabs (Training Compute, Training Memory, Inference Analysis, Parallelism & Communication,
Routing & Efficiency) are stubbed as “Phase 2+” and will be wired to your Python analysis
engine in later iterations.

### Tech stack

- **Framework**: React 18 + TypeScript + Vite
- **Styling**: Tailwind CSS (dark, dashboard-style UI)
- **State**: Zustand store for global sidebar config + derived metrics
- **Charts**: Recharts for Phase 1 (ECharts deps are installed for radar charts later)

### Getting started

From `c:\Users\Humaira\Documents\MOE_analysis`:

```bash
npm install
npm run dev
```

Then open the printed `http://localhost:5173` URL in your browser.

### Deploying to Vercel

1. **Install Vercel CLI** (optional; you can also use the Vercel dashboard):

   ```bash
   npm i -g vercel
   ```

2. **From the project root**, deploy:

   ```bash
   vercel
   ```

   First run will ask you to log in and link the project. Accept defaults (Vite is auto-detected).

3. **Production deploy**:

   ```bash
   vercel --prod
   ```

4. **Using the Vercel dashboard** (no CLI):

   - Push the repo to GitHub/GitLab/Bitbucket.
   - Go to [vercel.com](https://vercel.com) → **Add New Project** → import the repo.
   - Vercel will detect Vite; use **Build Command**: `npm run build`, **Output Directory**: `dist`.
   - Deploy. Future pushes to the main branch will auto-deploy.

The repo includes a `vercel.json` that sets the output directory and SPA rewrites so client-side routing works.

### How the math is wired (Phase 1)

- **Parameter counts** follow the spec:
  - `Embedding = V × d_model`
  - `Per-layer attention = Q + K + V + O projections`, with GQA using `n_kv_heads`
  - `Per-expert FFN = 3 × d_model × d_ff` (SwiGLU assumption)
  - `Routed expert params = L × E × per_expert_FFN`
  - `Shared expert params = L × N_shared × per_shared_expert_FFN`
  - `Output head = V × d_model`
- **Active parameters per token**:
  - `P_active = P_embedding + P_attention + L × K × P_single_expert + L × N_shared × P_shared_expert + P_output`
  - `L_moe` is assumed equal to `L` for now (all layers MoE); can be exposed later.

The validation tab currently compares parameter counts against hardcoded paper values for:

- **DeepSeek-V3**: 671B total params, ~37B active params
- **Mixtral 8x7B**: 46.7B total, ~12.9B active (for reference)

FLOPs, training memory, inference latency, and communication metrics will be added in
subsequent phases, using your existing Python engine (`main_dummy.py`) as the reference
math implementation.

