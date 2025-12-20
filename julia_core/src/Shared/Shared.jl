module Shared
using Reexport

# 核心类型定义 (必须最先加载)
include("Types.jl")
@reexport using .Types

include("ChanLun.jl")
@reexport using .ChanLun

include("DataFetcher.jl")
@reexport using .DataFetcher

include("Model.jl")
@reexport using .Model

include("Features/Features.jl")
@reexport using .Features

# 目标生成 (依赖 Types 和 Features)
include("Targets.jl")
@reexport using .Targets

end
