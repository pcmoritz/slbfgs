mkdir -p $HOME/output/adult/nored/
mkdir -p $HOME/output/adult/red/

julia experiment.jl --hdf $HOME/output/adult/nored/0.00001.h5 --exp adult-svm.jl --stepsize 0.0001 --memory_size 10 --reduce_grad_var false --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
julia experiment.jl --hdf $HOME/output/adult/nored/0.00005.h5 --exp adult-svm.jl --stepsize 0.0005 --memory_size 10 --reduce_grad_var false --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
julia experiment.jl --hdf $HOME/output/adult/nored/0.0001.h5 --exp adult-svm.jl --stepsize 0.001 --memory_size 10 --reduce_grad_var false --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
julia experiment.jl --hdf $HOME/output/adult/nored/0.0005.h5 --exp adult-svm.jl --stepsize 0.005 --memory_size 10 --reduce_grad_var false --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
julia experiment.jl --hdf $HOME/output/adult/nored/0.001.h5 --exp adult-svm.jl --stepsize 0.01 --memory_size 10 --reduce_grad_var false --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &

julia experiment.jl --hdf $HOME/output/adult/red/0.00001.h5 --exp adult-svm.jl --stepsize 0.00001 --memory_size 10 --reduce_grad_var true --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
julia experiment.jl --hdf $HOME/output/adult/red/0.00005.h5 --exp adult-svm.jl --stepsize 0.00005 --memory_size 10 --reduce_grad_var true --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
julia experiment.jl --hdf $HOME/output/adult/red/0.0001.h5 --exp adult-svmjl --stepsize 0.0001 --memory_size 10 --reduce_grad_var true --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
julia experiment.jl --hdf $HOME/output/adult/red/0.0005.h5 --exp adult-svm.jl --stepsize 0.0005 --memory_size 10 --reduce_grad_var true --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
julia experiment.jl --hdf $HOME/output/adult/red/0.001.h5 --exp adult-svm.jl --stepsize 0.001 --memory_size 10 --reduce_grad_var true --delta 0.00 --hess_period 10 --hess_samples 100 --epochs 30 &
