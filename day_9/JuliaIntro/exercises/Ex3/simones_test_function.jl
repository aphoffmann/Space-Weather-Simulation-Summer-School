cd(@__DIR__)
include("dual_number_arithmetic.jl")
using CairoMakie

# Simone's test function
func(x) = cos(x)+x*sin(x)
func_dot(x) = x*cos(x)

# points at which we want to evaluate Simone's test function and its derivative 
x_range = range(-6, 6, length = 100) 

# evaluate Simone's test function for each x in x_range
vals = func.(x_range)

# evaluate Simeone's test function derivative for each x in x_range
grads = func_dot.(x_range)

# Now we initialize the dual numbers for the evaluation of the derivative 
# of Simone's test function as
x_dual = [DualNumber(x,1.0) for x in x_range]

# evaluate Simone's test function for each x in x_dual and save the result in dual_result
dual_result = func.(x_dual)

#############################################################
# plotting 
#############################################################
fig = Figure(fontsize=18);
axs = [Axis(fig[1,1], xlabel = L"x", ylabel = L"f(x)"),
       Axis(fig[1,2], xlabel = L"x", ylabel = L"\frac{df}{dx}(x)")]

lines!(axs[1], x_range, func.(x_range), color = :black, linestyle=:dash, linewidth=4,label="Analytical")
lines!(axs[2], x_range, func_dot.(x_range), color = :black, linestyle=:dash, linewidth=4)
scatter!(axs[1], [x.val for x in x_dual], [x.val for x in dual_result], color = :red, linewidth=2, label = "AD")
scatter!(axs[2], [x.val for x in x_dual], [x.grad for x in dual_result], color = :red, linewidth=2)
axislegend(axs[1])
display(fig)
