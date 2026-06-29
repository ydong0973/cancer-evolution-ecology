
using Pkg
# Pkg.activate("/Users/dy/Desktop/cancer evolution ecology/EcoEvoSim")
Pkg.activate("./EcoEvoSim")
Pkg.instantiate()

# Pkg.add("Plots")
# Pkg.add("CSV")
# Pkg.add("Random")
# Pkg.status()

using EcoEvoSim, Plots, Random, CSV


# %%
z0, eta, sigma = 0.5, 0.2, 0.13 #evolving trait (proliferation capacity), smoothness of the transition , transition width (!= 0) -- default is 0.5, 0.2, 0.13
at, rt = 1e-14, 1e-8 #absolute and relative tolerance 
et = 0.003 #extThreshold
mv, iv, n= 0.001^2, 0.001, 10_000 #mutation variance, invaderPopsize, mutation steps 
z_init = 0.3
test_seed = 54321 #45/ 54321

# different models? 


println("z0 = ", z0, ", eta = ", eta, ", sigma = ", sigma, 
", absolute tolerance = ", at, ", relative tolerance = ", rt,", extinction threshold = ", et,
", mutation variance = ", mv, ", invaderPopsize = ", iv, ", mutation steps = ", n,
", initial trait value = ", z_init,
", seed = ", test_seed)

Q(x) = (tanh(x) + 1) / 2
growthFn(z) = Q((sum(z) - z0) / eta) - Q(-z0 / eta)
kernelFn(zi, zj) = -Q((sum(zi) - sum(zj)) / sigma) 

config = EcoEvoConfig(
    ecoDyn = lotkaVolterra(growthFn, kernelFn),
    # ecology = unstructuredModel() do i, n, z, aux, S, t
    #     n[i] * (b(z[i]) + sum(a(z[i], z[j]) * n[j] for j in 1:S))

    mutationGenerator = generateMutantWeighted(
        invaderPopsize = iv,
        variance = mv 
    ),
    # mutationGenerator = noMutation,
    integrationParams = IntegrationParams(
        maxTime = Inf,
        algorithm = DynamicSS(), #Rodas5() or Tsit5() or FunctionMap() or DiscreteSS() for recursions
        abstol = at,
        reltol = rt
    ),
    extThreshold = et
)

Random.seed!(test_seed) 

initCommunity = ecoDyn(Community([1.0], [z_init]), config)
@time lineage = evolve(initCommunity, config, n)
lineageTab = historyToTable(lineage)
# CSV.write("evo-table.csv", lineageTab)

# lineageTab["mutNo"]      # vector of mutation event numbers
# lineageTab["time"]       # vector of elapsed times
# lineageTab["species"]    # vector of species indices
# lineageTab["popsize_1"]  # vector of population sizes
# lineageTab["trait_1"]    # vector of trait values

# println("mutNo: ", lineageTab["mutNo"][5])


# %%
plotEvo(lineage)
# @btime 


