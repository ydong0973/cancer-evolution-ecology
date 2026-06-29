using Pkg
Pkg.activate("./EcoEvoSim")
Pkg.instantiate()

using EcoEvoSim, Plots, Random

# %%
# ─── Shared ecological parameters ────────────────────────────────────────────

z0, eta, sigma = 0.5, 0.2, 0.13              #evolving trait (proliferation capacity), smoothness of the transition , transition width (!= 0) 
at, rt         = 1e-14, 1e-8                 #absolute and relative tolerance 
et             = 0.003                       #extThreshold
mv, iv         = 0.001^2, 0.001              #mutation variance, invaderPopsize 
n_steps        = 3_000                       #mutation steps 
z_init         = 0.3
test_seed      = 54321

Q(x)             = (tanh(x) + 1) / 2
growthFn(z)      = Q((sum(z) - z0) / eta) - Q(-z0 / eta)
kernelFn(zi, zj) = -Q((sum(zi) - sum(zj)) / sigma)

function make_config(gFn; alg=DynamicSS(), maxT=Inf)
    EcoEvoConfig(
        ecoDyn            = lotkaVolterra(gFn, kernelFn),
        mutationGenerator = generateMutantWeighted(invaderPopsize=iv, variance=mv),
        integrationParams = IntegrationParams(maxTime=maxT, algorithm=alg, abstol=at, reltol=rt),
        extThreshold      = et
    )
end

# ─── Therapy parameters   ────────────────────────────────────────────────────────

drug_conc_chemo    = 0.3    # Option 1a: linear kill strength against high z-clones

z_target           = 0.6    # Option 1b: trait value the drug targets
kill_width         = 0.08   # Option 1b: width of Gaussian kill window (smaller = more specific)
drug_conc_targeted = 0.9    # Option 1b: Gaussian kill peak height (0-1)

C_max              = 0.8    # Option 3: peak drug concentration at dose time (t=0 of each settling)
k_decay            = 0.05   # Option 3: exponential clearance rate (higher = faster clearance)

# ─── Therapy growth functions — defined once, shared by plots & simulations ───

growthFn_chemo(z)    = growthFn(z) - drug_conc_chemo * sum(z)
kill_term(z)         = drug_conc_targeted * exp(-((sum(z) - z_target)^2) / (2 * kill_width^2))
growthFn_targeted(z) = growthFn(z) - kill_term(z)


# ─── Initial community ────────────────────────────────────────────────────────

Random.seed!(test_seed)
init = ecoDyn(Community([1.0], [z_init]), make_config(growthFn))


# ═══════════════════════════════════════════════════════════════════════════════
# GROWTH FUNCTION PLOTS
# Shows the fitness landscape each option imposes before running simulations.
# ─────────────────────────────────────────────────────────────────────────────

z_plot = range(0.0, 1.2; length=300)

# scalar wrappers so we can broadcast over z_plot (EcoEvoSim functions take vectors)
_b(z)          = growthFn([z])
_b_chemo(z)    = growthFn_chemo([z])
_b_targeted(z) = growthFn_targeted([z])
_kill(z)       = kill_term([z])

pg1 = plot(z_plot, _b.(z_plot);
    label="b(z) — untreated", color=:steelblue, linewidth=2,
    xlabel="Proliferation capacity (z)", ylabel="Per-capita growth rate",
    title="Option 1a — Chemotherapy", legend=:topleft)
plot!(pg1, z_plot, _b_chemo.(z_plot);
    label="b_treated(z)  (drug_conc = $drug_conc_chemo)", color=:crimson, linewidth=2)
hline!(pg1, [0.0]; color=:gray, linestyle=:dash, linewidth=1, label="extinction threshold")

pg2 = plot(z_plot, _b.(z_plot);
    label="b(z) — untreated", color=:steelblue, linewidth=2,
    xlabel="Proliferation capacity (z)", ylabel="Per-capita growth rate",
    title="Option 1b — Targeted therapy", legend=:topleft)
plot!(pg2, z_plot, _b_targeted.(z_plot);
    label="b_treated(z)  (z_target = $z_target, w = $kill_width)", color=:crimson, linewidth=2)
plot!(pg2, z_plot, _kill.(z_plot);
    label="kill term (Gaussian)", color=:darkorange, linewidth=1.5, linestyle=:dash)
hline!(pg2, [0.0]; color=:gray, linestyle=:dash, linewidth=1, label="extinction threshold")
vline!(pg2, [z_target]; color=:darkorange, linestyle=:dot, linewidth=1, label="z_target")

pg = plot(pg1, pg2; layout=(1, 2), size=(1100, 420), margin=5Plots.mm)
savefig(pg, "fig_growth_fns.png")
println("fig_growth_fns.png written.")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 1a — Chemotherapy
# ───────────────────────────────────────────────────────────────────────────────
# Drug kills fast-proliferating cells proportionally to z.
# Selects for slow-growing resistant clones (low z).
# Implementation: subtract a linear death term from growthFn.
#   b_treated(z) = b(z) - drug_conc * z
# ─────────────────────────────────────────────────────────────────────────────

config_chemo = make_config(growthFn_chemo)

Random.seed!(test_seed)
lineage_chemo = evolve(init, config_chemo, n_steps; showProgress=false)

p1a = plotEvo(lineage_chemo;
    title="Option 1a: Chemotherapy (drug_conc = $drug_conc_chemo)",
    xlabel="Proliferation capacity (z)", ylabel="Mutation event")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 1b — Targeted Therapy
# ───────────────────────────────────────────────────────────────────────────────
# Drug kills cells near z_target (e.g. cells expressing a targetable receptor).
# Biologically: oncoprotein inhibitors, HER2 blockers, etc. → selection escapes
# Resistance evolves by trait-space escape away from z_target.
# Implementation: Gaussian kill term centred at z_target.
#   b_treated(z) = b(z) - drug_conc * exp(-(z - z_target)² / 2w²)
# ─────────────────────────────────────────────────────────────────────────────

config_targeted = make_config(growthFn_targeted)

Random.seed!(test_seed)
lineage_targeted = evolve(init, config_targeted, n_steps; showProgress=false)

p1b = plotEvo(lineage_targeted;
    title="Option 1b: Targeted therapy (z_target = $z_target)",
    xlabel="Proliferation capacity (z)", 
    ylabel="Mutation event")


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 2 — Sequential Phases (Adaptive / Intermittent Therapy)
# ───────────────────────────────────────────────────────────────────────────────
# Mechanism: therapy is applied in phases rather than continuously.
# Biologically: models treatment → resistance → drug holiday → re-sensitisation
# (adaptive therapy), or on/off cycles to stabilise the tumour community.
#
# Implementation: evolve! extends the history in place so mutation event numbers
# are continuous across phases. Config is swapped between phases.
#   Phase 1 [1 – n_steps]        : no treatment (baseline evolution)
#   Phase 2 [n_steps+1 – 2*n]   : chemo on  (uses growthFn_chemo from option1a) 
#   Phase 3 [2*n_steps+1 – 3*n] : treatment withdrawn
# ─────────────────────────────────────────────────────────────────────────────

config_off = make_config(growthFn)         # no treatment
config_on  = make_config(growthFn_chemo)   # same chemo as Option 1a

Random.seed!(test_seed)
init2 = ecoDyn(Community([1.0], [z_init]), config_off)

lineage_phases = evolve(init2, config_off, n_steps; showProgress=false)
EcoEvoSim.evolve!(lineage_phases, config_on,  n_steps; showProgress=false)
EcoEvoSim.evolve!(lineage_phases, config_off, n_steps; showProgress=false)

p2 = plotEvo(lineage_phases;
    title="Option 2: Sequential phases (off → on → off)",
    xlabel="Proliferation capacity (z)", ylabel="Mutation event")
hline!(p2, [n_steps, 2*n_steps];
    linestyle=:dash, color=:black, linewidth=1.5, label=["Treatment on" "Treatment off"])


# ═══════════════════════════════════════════════════════════════════════════════
# OPTION 3 — Time-Varying Drug (Pharmacokinetics)
# ───────────────────────────────────────────────────────────────────────────────
# Drug decays exponentially within each ecological settling period. (peak dose → clearance).
# Biologically: captures dosing and clearance between treatment cycles; cells
# experience high drug early in each settling window, lower drug later.
#
# Implementation: unstructuredModel with explicit t in the ODE.
#   C(t) = C_max * exp(-k_decay * t)
#   dn_i/dt = n_i * [b(z_i) - C(t)*z_i + Σ_j a(z_i,z_j)*n_j]
#
# Note: DynamicSS() cannot be used here (no fixed point with time-varying input).
# — use Rodas5() with finite maxTime.
# ─────────────────────────────────────────────────────────────────────────────

ecology_pk = unstructuredModel() do i, n, z, aux, S, t
    drug_effect = C_max * exp(-k_decay * t) * sum(z[i])
    b_i = growthFn(z[i]) - drug_effect
    n[i] * (b_i + sum(kernelFn(z[i], z[j]) * n[j] for j in 1:S))
end

config_pk = EcoEvoConfig(
    ecoDyn            = ecology_pk,
    mutationGenerator = generateMutantWeighted(invaderPopsize=iv, variance=mv),
    integrationParams = IntegrationParams(maxTime=200.0, algorithm=Rodas5(), abstol=at, reltol=rt),
    extThreshold      = et
)

Random.seed!(test_seed)
init3 = ecoDyn(Community([1.0], [z_init]), config_pk)
lineage_pk = evolve(init3, config_pk, n_steps; showProgress=false)

p3 = plotEvo(lineage_pk;
    title="Option 3: PK drug (C_max=$C_max, k_decay=$k_decay)",
    xlabel="Proliferation capacity (z)",
    ylabel="Mutation event")


# ─── Summary figure ───────────────────────────────────────────────────────────

p_all = plot(p1a, p1b, p2, p3;
            layout=(2, 2), 
            size=(1200, 900), 
            margin=5Plots.mm)

savefig(p_all, "fig_therapy.png")
println("fig_therapy.png written.")
