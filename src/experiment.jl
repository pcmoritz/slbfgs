using HDF5
using ArgParse

require("slbfgs.jl")
require("sgd.jl")

s = ArgParseSettings()

@add_arg_table s begin
  "--hdf"
  help = "Name of the HDF5 where results should be stored"
  arg_type = String
  "--seed"
  help = "Random seed"
  arg_type = Int
  default = -1
  "--exp"
  help = "Filename of a Julia file describing the experiment"
  # A Julia file describing an experiment must export the following symbols:
  # data for the dataset, loss for the loss function, grad! for the gradient
  # and hvp! for Hessian vector products  and theta for the initial
  # parameter vector (see blog.jl)
  arg_type = String
  default = "blog.jl"
  "--method"
  help = "The optimization method to be used, can be 'lbfgs' or 'sgd'"
  arg_type = String
  default = "lbfgs"
  "--epochs"
  help = "Number of passes through the data"
  arg_type = Int
  default = 20
  "--stepsize"
  help = "The stepsize to be used"
  arg_type = Float64
  default = 0.0001
  "--lambda"
  help = "Regularization parameter"
  arg_type = Float64
  default = 0.001
  "--memory_size"
  help = "The memory size of the LBFGS matrix"
  arg_type = Int
  default = 10
  "--delta"
  help = "Regularization parameter for LBFGS matrix, think B_k^{-1} is replaced by (B_k + delta I)^{-1}"
  arg_type = Float64
  default = 0.0
  "--hess_samples"
  help = "Number of samples used to estimate the Hessian matrix"
  arg_type = Int
  default = 20
  "--hess_period"
  help = "Number of stochastic steps before a new vector pair is added to the LBFGS matrix"
  arg_type = Int
  default = 10
  "--reduce_grad_var"
  help = "Whether variance reduced gradients are used"
  arg_type = Bool
  default = true
  "--use_gradient_step"
  help = "When updating theta, take B_k as the identity matrix"
  arg_type = Bool
  default = false
  "--use_hvp"
  help = "If true, use hessian vector products to approximate the LBFGS matrix"
  arg_type = Bool
  default = true
end

parsed_args = parse_args(ARGS, s)

println(parsed_args)

function save_dict(dict::Dict, group::HDF5Group)
  for (key, value) in dict
    group[key] = string(value)
  end
end

function do_experiment(config::Dict)
  if config["seed"] == -1
    config["seed"] = time_ns()
  end
  srand(config["seed"])

  require(config["exp"])

  h5open(config["hdf"], "w") do hdf
      params = g_create(hdf, "params")
      save_dict(config, params)
      diagnostics = g_create(hdf, "diagnostics")
      dataset = g_create(hdf, "dataset")
      save_dict(dataset_info, dataset)
      if config["method"] == "sgd"
        sgd(loss, grad!, data, theta, config, diagnostics)
      end
      if config["method"] == "lbfgs"
        res = slbfgs(loss, grad!, hvp!, data, theta, config, diagnostics)
      end
  end
end

do_experiment(parsed_args)
