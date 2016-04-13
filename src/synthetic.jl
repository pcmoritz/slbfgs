using Regression
using SVM
using Optim

require("preprocess.jl")

require("svm.jl")

X = randn(100, 5000)

I = randbool(100, 5000)
X[I] = 0.0

X[1,:] = 1.0

theta_star = randn(100)

y = 1.0*((X' * theta_star + 0.05 * randn(5000)) .>= 0.0)

data = [X; y']

num_features = size(X, 1)

theta = randn(num_features)/100

lambda = 0.001

function func(x::Vector)
  return loss(data, x, lambda)
end

function gradient!(x::Vector, storage::Vector)
  copy!(storage, full_grad(grad!, data, x, lambda))
end

res = optimize(func, gradient!, theta, method = :l_bfgs; grtol=1e-13)

dataset_info = ["name" => "synthetic", "theta_star" => res.minimum]using Regression
