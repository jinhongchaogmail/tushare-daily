module Min5
using Reexport
include("Train.jl")
@reexport using .Train
end
