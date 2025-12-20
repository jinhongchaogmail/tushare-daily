module Prediction
using Reexport

include("Daily/Daily.jl")
@reexport using .Daily

include("Min5/Min5.jl")
@reexport using .Min5

end

