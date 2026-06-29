#=
make_figures.jl
Reproduce Figure 2 and Figure 3 from the EcoEvoSim manuscript.

Figure 2: Eco-evolutionary trajectory of the competition-proliferation tradeoff model.
Figure 3: Community matrix M and its inverse M^{-1} at the evolutionary endpoint.

Run time: ~5-15 minutes depending on hardware (10 000 DynamicSS steps).
=#

using Pkg
Pkg.activate("./EcoEvoSim")
Pkg.instantiate()

using EcoEvoSim, Plots, Random, LinearAlgebra, Printf

# ─── Model definition (Equations 4–6 in manuscript) ──────────────────────────

Q(x) = (tanh(x) + 1) / 2          # sigmoid (Eq. 4)

z0    = 0.5   # proliferation threshold
eta   = 0.2   # transition smoothness for growth function
sigma = 0.13  # competitive transition width

growthFn(z) = Q((sum(z) - z0) / eta) - Q(-z0 / eta)   # b(z), Eq. 5
kernelFn(zi, zj) = -Q((sum(zi) - sum(zj)) / sigma)     # a(z,z'), Eq. 6

config = EcoEvoConfig(
    ecoDyn = lotkaVolterra(growthFn, kernelFn),
    mutationGenerator = generateMutantWeighted(
        invaderPopsize = 0.001,
        variance       = 0.001^2
    ),
    integrationParams = IntegrationParams(
        maxTime   = Inf,
        algorithm = DynamicSS(),
        abstol    = 1e-14,
        reltol    = 1e-8
    ),
    extThreshold = 0.003
)

# ─── Run simulation ───────────────────────────────────────────────────────────

println("Starting simulation (10 000 mutation events)…")
Random.seed!(54321)
t0 = time()

initCommunity = ecoDyn(Community([1.0], [0.3]), config)
lineage = evolve(initCommunity, config, 10_000)

elapsed = round(time() - t0; digits=1)
println("Done in $(elapsed) s.")

# ─── Figure 2: evolutionary trajectory ───────────────────────────────────────

p2 = plotEvo(lineage;
    xlabel    = "Proliferation capacity (z)",
    ylabel    = "Mutation event",
    markersize = 2.5,
    alpha      = 0.55,
    size       = (720, 520)
)

savefig(p2, "fig2.png")
println("fig2.png written.")

# ─── Final community (ordered by trait for Figure 3) ─────────────────────────

finalComm = orderByTrait(lastCommunity(lineage))
nClones   = numSpecies(finalComm)
n_star    = [popsizes(finalComm, i)[1] for i in 1:nClones]
z_vals    = [traits(finalComm, i)[1]   for i in 1:nClones]

println("\nFinal community: $nClones clones")
for i in 1:nClones
    @printf "  clone %d  z = %+.4f   n* = %.4f\n" i z_vals[i] n_star[i]
end

# ─── Community matrix (Equation 7): M_ij = n*_i · a(z_i, z_j) ───────────────
# At equilibrium the per-capita growth rates are zero, so the diagonal term
# b(z_i) + Σ_k a(z_i,z_k)·n*_k vanishes and only n*_i·a(z_i,z_j) remains.

M    = [n_star[i] * kernelFn([z_vals[i]], [z_vals[j]]) for i in 1:nClones, j in 1:nClones]
Minv = inv(M)

println("\nCommunity matrix M (rows = clone i, cols = clone j):")
display(round.(M; digits=4))

println("\nInverse community matrix M⁻¹:")
display(round.(Minv; digits=4))

# ─── SVD analysis: most-leveraged perturbation ───────────────────────────────
U, sv, V = svd(Minv)
println("\nSingular values of M⁻¹: $(round.(sv; digits=3))")
println("Leading right singular vector (perturbation direction):")
println(round.(V[:, 1]; digits=4))
println("(entry with largest magnitude identifies the highest-leverage clone)")

# ─── Figure 3: M and M⁻¹ as heatmaps ────────────────────────────────────────

tick_labels = string.(1:nClones)

function community_heatmap(mat, ttl)
    lim = maximum(abs.(mat))
    heatmap(mat;
        xticks         = (1:nClones, tick_labels),
        yticks         = (1:nClones, tick_labels),
        xlabel         = "Clone j",
        ylabel         = "Clone i",
        title          = ttl,
        yflip          = true,
        c              = :RdBu,
        clims          = (-lim, lim),
        aspect_ratio   = :equal,
        colorbar_title = "",
        framestyle     = :box
    )
end

p3a = community_heatmap(M,    "Community matrix M")
p3b = community_heatmap(Minv, "Inverse community matrix M⁻¹")

p3 = plot(p3a, p3b;
    layout = (1, 2),
    size   = (950, 420),
    margin = 5Plots.mm
)

savefig(p3, "fig3.png")
println("fig3.png written.")
println("\nAll figures produced successfully.")
