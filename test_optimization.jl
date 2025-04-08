
using DifferentialEquations, RecursiveArrayTools, DiffEqParamEstim
using Optimization, ForwardDiff, OptimizationOptimJL, OptimizationBBO
using Distributions
using Base

#x[0]=W
#x[1]=WE1
#x[2]=W*
#x[3]=W*E2
#x[4]=E1=E1T-x[1]
#x[5]=E2=E2T-x[2]

#define reaction network equations
function rxns(dxdt, x, p, t)
    
    a1=p[1]
    a2=p[2]
    d1=p[3]
    d2=p[4]
    k1=p[5]
    k2=p[6]

    #dw/dt=-a1(W)*(E1) + d1(WE1) + k2(W*E2)
    dxdt[1]=-a1*x[1]*x[5] + d1*x[2] + k2*x[4]
    #dWE1/dt=a1(W)*(E1) - (d1+k1)*(WE1)
    dxdt[2]=a1*x[1]*x[5]-(d1+k1)*x[2]
    #dW*/dt=-a2(W*)(E2) + d(2W*E2) + k1(WE1)
    dxdt[3]=-a2*x[3]*x[6] + d2*x[4] + k1*x[2]
    #d(W*E2)/dt=a2(W*)(E2) - (d2+k2)(W*E2)
    dxdt[4]=a2*x[3]*x[6] - (d2+k2)*x[4]
    dxdt[5]=-dxdt[2]
    dxdt[6]=-dxdt[3]

end

#get solutions for a set of parameters
p = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
u0 = [50; 0; 0; 0; 50; 50]
tspan = (0.0, 10.0)
prob1 = ODEProblem(rxns, u0, tspan, p)
sol = solve(prob1, Tsit5())

#print(sol.t)
#print(sol.u)


# generate a training data set 
t = collect(range(0, stop = 10, length = 200))
function generate_data(sol, t)
    for i in 1:length(t)
        print(sum(sol(t[i])))
        print('\n')
    end
    #randomized=VectorOfArray([sum(sol(t[i])) for i in 1:length(t)])
    #print(randomized)
    #draw from a multinomial distribution at each timepoint with the probabilities given by the solution at that timepoint 
    #randomized = VectorOfArray([Multinomial(sum(sol(t[i])), sol(t[i])/sum(sol(t[i]))) for i in 1:length(t)])
    #data = convert(Array, randomized)
end

aggregate_data = convert(Array, VectorOfArray([generate_data(sol, t) for i in 1:100]))

#=
#we are using cross entropy = log likelihood for multinomial distribution
distributions = [fit_mle(Multinomial, aggregate_data[i, j, :]) for i in 1:2, j in 1:200]

#define the cost function
cost_function = build_loss_objective(prob, Tsit5(), LogLikeLoss(t, distributions), Optimization.AutoForwardDiff(),maxiters = 10000, verbose = false)
optprob = Optimization.OptimizationProblem(cost_function, [1.0, 0.5])
result = solve(optprob, Optim.BFGS())

=#