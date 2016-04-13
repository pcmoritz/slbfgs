# Loss function, gradients and hessian vector products for SVMs with quadratic loss

println("Loading SVM module")

function loss(data::Matrix, theta::Vector, lambda::Float64)
  result = 0.0
  n = size(data, 2)
  for i = 1:n
    x = data[1:end-1,i]
    y = data[end,i]
    result += 0.5 * max(0.0, 1 - y * dot(vec(x), theta))^2/n
  end
  result += lambda/2 * dot(theta, theta)
  return result
end

function grad!(datapoint::Vector, theta::Vector, lambda::Float64, grad::Vector)
  x = datapoint[1:end-1]
  y = datapoint[end]
  alpha = dot(x, theta)
  if y * alpha < 1.0
    BLAS.axpy!(1.0, lambda * theta -  (1 - y*alpha) * y * x, grad)
  else
    BLAS.axpy!(1.0, lambda * theta, grad)
  end
end

function hvp!(datapoint::Vector, theta::Vector, v::Vector, lambda::Float64, result::Vector)
  x = datapoint[1:end-1]
  y = datapoint[end]
  alpha = dot(x, theta)
  if y * alpha < 1.0
    BLAS.axpy!(1.0, x * dot(x, v) + lambda * v, result)
  else
    BLAS.axpy!(1.0, lambda * v, result)
  end
end

require("checks.jl")
ad_hoc_check("svm")
