# p is number of features
# S is number of epochs
function sgd(loss::Function, grad!::Function, data::Matrix, theta::Vector, config::Dict, diagnostics::HDF5Group)
  @printf("===== running the stochastic gradient method =====\n")
  n = size(data, 2)
  theta = copy(theta)
  dtheta = Array(Float64, length(theta))
  eta = config["stepsize"]
  S = config["epochs"]
  verbose = true
  T = S*n
  for t in 1:T
    eta_t = eta / sqrt(t)
    fill!(dtheta, 0.0)
    index = rand(1:n)
    grad!(vec(data[:,index]), theta, config["lambda"], dtheta)
    if verbose && mod(t, n) == 0
      l = loss(data, theta, config["lambda"])
      @printf "%4d | %9.4E \n" t/n l
      diagnostics[@sprintf "%04d" t/n] = l
    end
    theta -= eta_t * dtheta
  end
  return theta
end
