@everywhere using DataStructures
using Base.Test
using ArrayViews

type InverseHessian
  syrhos::Deque{(Vector{Float64}, Vector{Float64}, Float64)}
  tau::Int
end

function InverseHessian(tau::Int)
  return InverseHessian(Deque{(Vector{Float64}, Vector{Float64}, Float64)}(), tau)
end

function add!(self::InverseHessian, s::Vector{Float64}, y::Vector{Float64})
  rho = 1./dot(s, y)
  push!(self.syrhos, (s, y, rho))
  if length(self.syrhos) > self.tau
    shift!(self.syrhos)
  end
end

function test_mvp(self::InverseHessian, g::Vector{Float64}, delta::Float64)
  syrhos = collect(self.syrhos)
  (s, y, rho) = syrhos[end]
  p = length(s)
  gamma = dot(s, y)/dot(y, y)
  H = gamma / (1 + delta*gamma) * eye(p)
  for (s, y, rho) in syrhos
    V = eye(p) - rho * y * s'
    H = V' * H * V + rho * s * s'
  end
  return H*g
end

function mvp(self::InverseHessian, g::Vector{Float64}, delta::Float64)
  @assert length(self.syrhos) > 0
  q = copy(g)
  alphas = zeros(length(self.syrhos))
  for (i, (s, y, rho)) in enumerate(reverse(collect(self.syrhos))) # TODO: Implement reverse iterator for Deques
    alpha = rho * dot(s, q)
     alphas[length(self.syrhos) - i + 1] = alpha # TODO: implement reverse(enumerate)
     q -= alpha * y
  end
  (s, y, rho) = back(self.syrhos) # current iterate
  gamma = dot(s, y)/dot(y, y)
  r = gamma/(1 + delta*gamma)*q
  for (i, (s, y, rho)) in enumerate(collect(self.syrhos)) # TODO: Remove collect overhead
    beta = rho * dot(y, r)
    r += s * (alphas[i] - beta)
  end
  return r
end

function test_inverse_hessian()
  self = InverseHessian(3)
  for i = 1:4
    s = rand(2)
    y = rand(2)
    add!(self, s, y)
  end
  g = rand(2)
  first = mvp(self, g, 0.0)
  second = test_mvp(self, g, 0.0)
  @test norm(first - second) <= 1e-10
end

test_inverse_hessian()

# S is the number of epochs
function slbfgs(loss::Function, grad!::Function, hvp!::Function, data::Matrix, theta::Vector, config::Dict, diagnostics::HDF5Group)
  tau = config["memory_size"]
  delta = config["delta"]
  eta = config["stepsize"]
  S = config["epochs"]
  verbose = true
  println("===== running stochastic lbfgs =====\n")
  println("iter | lbfgs error")
  n = size(data, 1)
  m = 2*n
  p = length(theta)
  H = InverseHessian(tau)
  thetaold = copy(theta)
  dtheta = zeros(p)
  mu = zeros(p)
  z = zeros(p)
  zold = zeros(p)
  P = zeros(p)

  t = -1 # number of correction pairs currently computed
  stheta = zeros(p)
  sthetaold = zeros(p)

  hdf_loss = g_create(diagnostics, "loss")
  hdf_theta = g_create(diagnostics, "theta")

  if config["use_gradient_step"]
    println("warning: using gradient steps")
  end

  for s in 1:S
    if verbose
      l = loss(data, theta, config["lambda"])
      @printf "%4d | %9.4E \n" s l
      hdf_loss[@sprintf "%04d" s] = l
      hdf_theta[@sprintf "%04d" s] = theta
    end

    ctheta = copy(theta) # theta checkpoint
    # \tilde \mu = 1/n \sum \nable f_i(\tilde theta)
    fill!(mu, 0.0)
    for i = 1:n
      grad!(vec(data[:,i]), ctheta, config["lambda"], mu)
    end
    mu /= n
    if s == 1
      # Do a gradient step to get LBFGS started
      copy!(P, -mu/norm(mu))
    end
    copy!(theta, ctheta)
    for k in 1:m
      index = rand(1:n)
      copy!(zold, z)
      fill!(z, 0.0)
      # z_k = \nabla f_{i_k}(\theta) - \nabla f_{i_k}(\tilde\theta) + \tilde \mu
      fill!(dtheta, 0.0)
      grad!(vec(data[:,index]), theta, config["lambda"], dtheta)
      z += dtheta
      fill!(dtheta, 0.0)
      grad!(vec(data[:,index]), ctheta, config["lambda"], dtheta)
      if config["reduce_grad_var"]
        z -= dtheta
        z += mu
      end

      stheta += theta

      if k > 1 # neccessary?
        if mod(k, config["hess_period"]) == 0
          stheta /= config["hess_period"]
          t += 1
        end
        if mod(k, config["hess_period"]) == 0 && t >= 1
          svec = stheta - sthetaold
          r = zeros(p)
          if config["use_hvp"]
            for i = 1:config["hess_samples"]
              index = rand(1:n)
              hvp!(vec(data[:,index]), stheta, svec, config["lambda"], r)
            end
            r /= config["hess_samples"]
          else
                # delta a regularization parameter
                r = z - zold + delta*svec
            end
            add!(H, svec, r)
            copy!(sthetaold, stheta)
        end
        if t > 1
          P = -eta * mvp(H, z, delta)
        else
          P = -eta*z
        end
      end
      # take step
      copy!(thetaold, theta)
      if config["use_gradient_step"]
        theta -= eta*z
      else
        theta += P
      end
    end
  end
  return theta
end
