module Model

using PythonCall
using DataFrames

export train_catboost, predict_catboost, save_model, load_model

# 声明将要导入的Python模块对象
const cb = PythonCall.pynew()
const pd = PythonCall.pynew()
const np = PythonCall.pynew()

"""
    __init__()

模块初始化函数，用于建立与Python环境的连接，并导入catboost、pandas和numpy库。
"""
function __init__()
    PythonCall.pycopy!(cb, pyimport("catboost"))
    PythonCall.pycopy!(pd, pyimport("pandas"))
    PythonCall.pycopy!(np, pyimport("numpy"))
end

"""
    train_catboost(X_train, y_train, params::Dict)

使用CatBoost库训练一个分类模型。

参数:
- X_train: 训练特征矩阵，可以是DataFrame或Matrix
- y_train: 训练标签向量
- params: 一个字典，包含CatBoost模型的所有超参数（如iterations, learning_rate等）

返回值:
- 训练好的CatBoost模型对象

说明:
- 函数内部会合并用户提供的参数和默认参数。
- 使用cb.Pool将数据打包，然后调用model.fit进行训练。
"""
function train_catboost(X_train, y_train, params::Dict)
    # 转换为 CatBoost Pool
    # 关键修复: Julia 数组必须转换为 NumPy 数组才能被 CatBoost 接受
    # 如果 X_train 是 DataFrame，先转为 Matrix
    X_np = if X_train isa DataFrame
        np.array(Matrix(X_train))
    elseif X_train isa AbstractMatrix
        np.array(X_train)
    else
        X_train  # 假设已经是 NumPy 数组
    end
    
    y_np = if y_train isa AbstractVector
        np.array(y_train)
    else
        y_train  # 假设已经是 NumPy 数组
    end
    
    pool = cb.Pool(X_np, label=y_np)
    
    # 设置默认参数 - 使用 Symbol key 以便 splatting
    default_params = Dict{Symbol, Any}(
        :iterations => 1000,
        :learning_rate => 0.03,
        :depth => 6,
        :loss_function => "Logloss",
        :verbose => 100,
        :task_type => "CPU" # 如果有 GPU 可改为 "GPU"
    )
    
    # 将用户传入的 String key 转换为 Symbol key
    for (k, v) in params
        default_params[Symbol(k)] = v
    end
    
    # 使用 Symbol key splatting 传递参数给 CatBoost
    model = cb.CatBoostClassifier(; default_params...)
    model.fit(pool)
    
    return model
end

"""
    predict_catboost(model, X)

使用训练好的CatBoost模型对新数据进行预测（预测类别）。

参数:
- model: 已训练的CatBoost模型
- X: 待预测的特征数据

返回值:
- 预测的类别标签数组
"""
function predict_catboost(model, X)
    # 关键修复: 输入数据转换为 NumPy，输出转换回 Julia
    X_np = if X isa DataFrame
        np.array(Matrix(X))
    elseif X isa AbstractMatrix
        np.array(X)
    else
        X  # 假设已经是 NumPy 数组
    end
    
    preds_py = model.predict(X_np)
    return pyconvert(Vector{Int}, preds_py)
end

"""
    predict_proba_catboost(model, X)

预测每个样本属于各个类别的概率。

参数:
- model: 已训练的CatBoost模型
- X: 待预测的特征数据

返回值:
- 一个二维数组，每行代表一个样本，每列代表一个类别的概率。
"""
function predict_proba_catboost(model, X)
    # 关键修复: 输入数据转换为 NumPy，输出转换回 Julia
    X_np = if X isa DataFrame
        np.array(Matrix(X))
    elseif X isa AbstractMatrix
        np.array(X)
    else
        X  # 假设已经是 NumPy 数组
    end
    
    proba_py = model.predict_proba(X_np)
    return PyArray(proba_py)
end

"""
    save_model(model, path::String)

保存模型。

参数:
- model: 已训练的CatBoost模型
- path: 保存模型的文件路径

返回值:
- 无
"""
function save_model(model, path::String)
    model.save_model(path)
end

"""
    load_model(path::String)

加载模型。

参数:
- path: 保存模型的文件路径

返回值:
- 加载好的CatBoost模型对象
"""
function load_model(path::String)
    model = cb.CatBoostClassifier()
    model.load_model(path)
    return model
end

end # module
