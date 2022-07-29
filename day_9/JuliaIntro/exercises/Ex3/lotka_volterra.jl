cd(@__DIR__)
include("dual_number_arithmetic.jl")

using CairoMakie
function lotka_volterra(x,α)
    # right-hand-side of lotka-volterra system
    # inputs:
    #       - x : current state x[1] = population of prey, 
    #                           x[2] = populaiton of predator 
    #       - α : hunting efficiency of predator species
    # returns:
    #       - dx/dt : rate of change of population sizes
    
    dx = [x[1] - x[1]*x[2];
          -2*x[2] + α*x[1]*x[2]]
    return dx
end

# or when messing around with unicode symbols ....
function lotka_volterra(x,α)
    # right-hand-side of lotka-volterra system
    # inputs:
    #       - x : current state x[1] = population of prey, 
    #                           x[2] = populaiton of predator 
    #       - α : hunting efficiency of predator species
    # returns:
    #       - dx/dt : rate of change of population sizes

    🐇, 🐺 = x
    d🐇 = 🐇 - α*🐇*🐺
    d🐺 =  -🐺 + α*🐇*🐺
    return [d🐇, d🐺]
end

function simulate_lotka_volterra(x0,α,h,N)
    # simulator for lotka volterra system (explicit Euler)
    # inputs:
    #       - x0 : initial condition 
    #       - α  : hunting efficiency of predator
    #       - h  : step size 
    #       - N  : number of steps 
    # outputs:
    #       - trajectory : Vector of population sizes, entry for each time point visisted 
    #                      during integration
    #       - time_points: Vector of time points visisted during integration
    trajectory = [x0]
    time_points = range(0, N*h, length=N+1)
    for i in 1:N
        x_current = trajectory[end] # current state
        x_hat = x_current + h*lotka_volterra(x_current,α) # explicit euler prediction
        push!(trajectory, x_hat)
    end
    return trajectory, time_points
end

# initial condition for the simulation
x0 = [1.0, 0.25]
# step size
h = 0.01
# time horizon (0.0, N*h)
N = 1500
# parameter range for the predator efficiency
α_range = 1.2:0.01:2.8

# gradient prediction by automatic differentiation
# 
# please complete the
# to that end, recall that 
#       x0_dual = [x0[1] + ∂x0[1]/∂α * ϵ, x0[2] + ∂x0[2]/∂α * ϵ]
#       α_dual = [α + ∂α/∂α * ϵ]
# since we want to compute the sensitivity of the final state wrt. parameter α
α = DualNumber(1.5,1.0)
x0_dual = [DualNumber(x0[1], 0.0), DualNumber(x0[2], 0.0)]


# propagate dual numbers through simulation:
dual_trajectory, time_points = simulate_lotka_volterra(x0_dual, α, h, N)

##############
### No more modifications needed beyond this point 
##############

# General Lotka-Volterra system oscilaltions
trajectory, time_points = simulate_lotka_volterra(x0, 0.25, h, N)

fig = Figure(fontsize=18); 
ax_prey = Axis(fig[1,1], ylabel = "prey population", xlabel = "time")
ax_predator = Axis(fig[2,1], ylabel = "predator population", xlabel = "time")
lines!(ax_prey, time_points, [x[1].val for x in dual_trajectory])
lines!(ax_predator, time_points, [x[2].val for x in dual_trajectory])
display(fig)

# parameter dependence of the final state x(N*h) on α
x_final = []
for α in α_range
    trajectory, time_points = simulate_lotka_volterra(x0, α, h, N)
    push!(x_final, trajectory[end])
end

fig = Figure(fontsize=18); 
ax_prey = Axis(fig[1,1], ylabel = "prey population at $(N*h) months", xlabel = "α")
ax_predator = Axis(fig[2,1], ylabel = "predator population at $(N*h) months", xlabel = "α")
lines!(ax_prey, α_range, [x[1] for x in x_final])
lines!(ax_predator, α_range, [x[2] for x in x_final])
scatter!(ax_prey, [α.val], [dual_trajectory[end][1].val], color = :red)
lines!(ax_prey, [α.val-0.2, α.val+0.2], 
                [dual_trajectory[end][1].val - 0.2*dual_trajectory[end][1].grad,
                dual_trajectory[end][1].val + 0.2*dual_trajectory[end][1].grad],
                color = :red)

scatter!(ax_predator, [α.val], [dual_trajectory[end][2].val], color = :red)
lines!(ax_predator, [α.val-0.2, α.val+0.2], 
                    [dual_trajectory[end][2].val - 0.2*dual_trajectory[end][2].grad,
                    dual_trajectory[end][2].val + 0.2*dual_trajectory[end][2].grad],
                    color = :red)
display(fig)

 
