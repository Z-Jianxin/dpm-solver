devices="0"

steps="20"
eps="1e-6"
skip="time_uniform"
# skip="logSNR"
method="singlestep"
#method="adaptive"
order="1"
# dir="/scratch/clayscot_root/clayscot0/jianxinz/generated/test/cifar10_ddpmpp_deep_continuous/"
# dir="/scratch/clayscot_root/clayscot0/jianxinz/generated/dpm_solver_sde/cifar10_ddpmpp_deep_continuous"
dir="/scratch/clayscot_root/clayscot0/jianxinz/generated/test/cifar10_ddpmpp_deep_continuous"
eval_folder="eval_test_$(date "+%Y-%m-%d_%H-%M-%S")"

export LD_LIBRARY_PATH=/scratch/clayscot_root/clayscot0/jianxinz/conda_envs/dpm_solver1/lib:$LD_LIBRARY_PATH
CUDA_VISIBLE_DEVICES=$devices python main.py --config "configs/vp/cifar10_ddpmpp_deep_continuous.py" \
    --mode "eval" --workdir $dir --eval_folder $eval_folder \
    --config.sampling.eps=$eps --config.sampling.method="dpm_solver" \
    --config.sampling.steps=$steps --config.sampling.skip_type=$skip --config.sampling.dpm_solver_order=$order \
    --config.sampling.dpm_solver_method=$method \
    --config.eval.batch_size=512 --config.eval.num_samples=512 \
    --config.eval.begin_ckpt=19 --config.eval.end_ckpt=19 \
    --config.sampling.algorithm_type="dpmsolver_approx" \
    --config.sampling.atol=1e-2 --config.sampling.rtol=5e-2
