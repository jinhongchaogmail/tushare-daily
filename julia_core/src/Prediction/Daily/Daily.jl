module Daily
using Reexport

include("RunStrategy.jl")
@reexport using .RunStrategy

end
