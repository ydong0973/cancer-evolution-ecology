
using Pkg
# Pkg.activate("/Users/dy/Desktop/cancer evolution ecology/EcoEvoSim")
Pkg.activate("./EcoEvoSim")
Pkg.instantiate()

# Pkg.add("Plots")
# Pkg.add("CSV")
# Pkg.add("Random")
using EcoEvoSim, Plots, Random, CSV


z0, eta, sigma = 0.5, 0.2, 0.13 #evolving trait (proliferation capacity), smoothness of the transition , transition width (!= 0) -- default is 0.5, 0.2, 0.13
println("z0 = ", z0, ", eta = ", eta, ", sigma = ", sigma)

Q(x) = (tanh(x) + 1) / 2
growthFn(z) = Q((sum(z) - z0) / eta) - Q(-z0 / eta)
kernelFn(zi, zj) = -Q((sum(zi) - sum(zj)) / sigma) 

config = EcoEvoConfig(
    ecoDyn = lotkaVolterra(growthFn, kernelFn),
    mutationGenerator = generateMutantWeighted(
        invaderPopsize = 0.001,
        variance = 0.001^2
    ),
    integrationParams = IntegrationParams(
        maxTime = Inf,
        algorithm = DynamicSS(),
        abstol = 1e-14,
        reltol = 1e-8
    ),
    extThreshold = 0.003
)

Random.seed!(54321) #45/ 54321

initCommunity = ecoDyn(Community([1.0], [0.3]), config)
lineage = evolve(initCommunity, config, 10_000)
lineageTab = historyToTable(lineage)
# CSV.write("evo-table.csv", lineageTab)
plotEvo(lineage)



