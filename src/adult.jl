using Optim
using DataFrames

require("preprocess.jl")

dataset = :adult

df = readtable("adult.data")
# df = readtable("census-income.data")

# complete_cases!(df)

X = featurize(df[1:end-1])
y = class(df[end])

X = X[1:10000,:]
y = y[1:10000,:]

data = cat(2, X, y)

num_features = size(X, 2)

data = data'

theta = randn(num_features)

lambda = 0.001

function func(x::Vector)
  return loss(data, x, lambda)
end

function gradient!(x::Vector, storage::Vector)
  copy!(storage, full_grad(grad!, data, x, lambda))
end

res = optimize(func, gradient!, theta, method = :l_bfgs, show_trace=true, grtol=1e-5)

dataset_info = ["name" => "UCI", "theta_star" => res.minimum]
