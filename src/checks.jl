# Checking gradients and Hessian vector products

function check_single_grad(loss::Function, grad!::Function, datapoint::Vector, theta::Vector, lambda::Float64, eps::Float64)
  p = length(theta)
  delta = randn(p)
  deriv_approx = (loss(datapoint, theta+eps*delta, lambda) - loss(datapoint, theta-eps*delta, lambda))/(2*eps)
  grad = zeros(p)
  grad!(datapoint, theta, lambda, grad)
  deriv_exact = dot(grad, delta)
  return norm(deriv_approx - deriv_exact)
end

function check_grad(loss::Function, grad!::Function, data::Matrix, theta::Vector, lambda::Float64, eps::Float64)
  n = size(data, 2)
  p = length(theta)
  delta = randn(p)
  deriv_approx = (loss(data, theta+eps*delta, lambda) - loss(data, theta-eps*delta, lambda))/(2*eps)
  grad = zeros(p)
  for i = 1:n
    grad!(vec(data[:,i]), theta, lambda, grad)
  end
  deriv_exact = dot(grad/n, delta)
  return norm(deriv_approx - deriv_exact)
end

# Compute the full gradient over the whole data matrix
function full_grad(grad!::Function, data::Matrix, theta::Vector, lambda::Float64)
  n = size(data, 2)
  result = zeros(length(theta))
  for i = 1:n
    grad!(vec(data[:,i]), theta, lambda, result)
  end
  return result
end

function single_grad(grad!::Function, datapoint::Vector, theta::Vector, lambda::Float64)
  result = zeros(length(theta))
  grad!(datapoint, theta, lambda, result)
  return result
end

# plase check with check_gradient if the gradient function is correct before you use this function
function check_hvp(grad!::Function, hvp!::Function, data::Matrix, theta::Vector, lambda::Float64, eps::Float64)
  n = size(data, 2)
  p = length(theta)
  delta = randn(p)
  hvp_exact = zeros(p)
  for i = 1:n
    hvp!(vec(data[:,i]), theta, delta, lambda, hvp_exact)
  end
  hvp_approx = (full_grad(grad!, data, theta+eps*delta, lambda) - full_grad(grad!, data, theta-eps*delta, lambda))/(2*eps)
  return norm(hvp_exact - hvp_approx)
end

function check_single_hvp(grad!::Function, hvp!::Function, datapoint::Vector, theta::Vector, lambda::Float64, eps::Float64)
  p = length(theta)
  delta = randn(p)
  hvp_exact = zeros(p)
  hvp!(datapoint, theta, delta, lambda, hvp_exact)
  hvp_approx = (single_grad(grad!, datapoint, theta+eps*delta, lambda) - single_grad(grad!, datapoint, theta-eps*delta, lambda))/(2*eps)
  return norm(hvp_exact - hvp_approx)
end

function ad_hoc_check(mode::String)
  println("Checking gradients and hessian vector products")
  data = randn(100, 100)
  if mode == "svm"
    data[100,:] = 2.0*randbool(100) - 1
  end
  if mode == "logistic regression"
    data[100,:] = 2.0*randbool(100) - 1
  end
  theta = randn(99)
  gnorm = norm(check_grad(loss, grad!, data, theta, 1e-3, 1e-4))
  hnorm = norm(check_hvp(grad!, hvp!, data, theta, 1e-3, 1e-4))
  println(gnorm)
  @assert gnorm <= 1e-4
  println(hnorm)
  @assert hnorm <= 1e-4
end
