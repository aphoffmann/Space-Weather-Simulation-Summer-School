##################################
#### assignment and unicode symbols 🔥🔥🔥
##################################

# silly example 
🔥 = - 42
💧 = 42
🔥 * 💧
exp(🔥)

# Golden ratio
ϕ = (sqrt(5)+1)/2
φ = (√5 + 1)/2 
phi = (sqrt(5) + 1)/2 

# Navier stokes 
NS = ρ * (∂u + (u ⋅ ∇*u)) + 𝒫

# other useful things 
a = π
unit_disc_area = 2*a

##################################
#### Arrays/Vectors/Matrices
##################################

# defining a vector
powers_of_two = [1, 2, 4]
some_random_stuff = ["flemming", 43.5]

# appending stuff: push!, append!
push!(powers_of_two, 8)
append!(powers_of_two, [16, 32])

# defining a matrix
van_der_monde = [1 2 4 8; # first row
                 1 3 9 27] # second row

# concatenating 
# adding rows 
add_a_row = [van_der_monde;
             1 4 16 64]
stacked_van_der_monde = [van_der_monde; 
                         van_der_monde]

# adding columns
van_der_monde_bigger = [van_der_monde van_der_monde]

# indexing starts at 1!
van_der_monde[1,3]
# slicing
van_der_monde[1:2,3]
van_der_monde[1:2,2:3]

# last element is indexed by end keyword
powers_of_two[end]
powers_of_two[end-1]

##################################
#### loops
##################################
for power in powers_of_two 
    println(power)
end

for i in 1:length(powers_of_two) # discouraged!
    println(powers_of_two[i])
end

for i in eachindex(powers_of_two)
    println(powers_of_two[i])
end

i = 0
while i <= 10
    println(i)
    i += 1
end

#in particular ranges are written with : instead of range function
#range(5) <=> 0:4 

##################################
#### if-elseif-else 
##################################
a = 5.0
if a < 2
    println("a is less than 2")
elseif a < 3
    println("a is less than 3")
else 
    println("a is greater or equal to 3")
end

##################################
#### functions
##################################

# functional programming style
function my_add(a, b)
    # function body
    # in here you can do whatever you want with the arguments a, b
    c = a + b
    return c
    # or you could just do
    # return a + b
end

Σ = my_add(5, 3)

# for simple functions we may prefer the assignment form 
# to resemble standard math notation more closely
f(x) = 1/(2π)*exp(-x^2)
# evaluation 
p = f(0.5)

# vectorization/(map-reduce)
# evaluates our function at every element of the supplied 
# vector/array and returns the result in the same shape!
p = f.([0.5, 2.0])

# differences between python and julia
# Why Julia was created
# https://julialang.org/blog/2012/02/why-we-created-julia/# 
# Julia documentation: Noteworthy differences to other common languages
# https://docs.julialang.org/en/v1/manual/noteworthy-differences/
# Julia for data science
# https://www.imaginarycloud.com/blog/julia-vs-python/#future