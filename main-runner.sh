
#n_runs=400
#python plot_figures_misc.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta 0.01


#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta 0.007;
#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 1 --theta 0.007;
#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta 0.009;

#for th in $(seq 0.001 0.001 0.03)
#do
#    echo $th;
#    python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 1 --theta $th;
#done


#for th in $(seq 0.001 0.002 0.04)
#do
#    echo $th;
#    python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta $th;
#done

#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta 0.001

n_runs=100 #4
python process_data_varying_p_theta_exp.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2
#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta 0.01
#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta 0.02
#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta 0.04
#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2 --theta 0.08

#'''
#n_runs=100 #4
#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 1
#python main_runner_varying_Ts_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2

#python main_runner_varying_prob_p_theta_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 1
#python main_runner_varying_prob_p_theta_experiment.py --n_runs $n_runs --dataset cifar --data ../data/ --scheme 2
#'''

#python main_runner_varying_prob_p_theta_experiment.py --n_runs 4 --dataset cifar --data ../data/ --scheme 1
#python main_runner_varying_prob_p_theta_experiment.py --n_runs 4 --dataset cifar --data ../data/ --scheme 2

#python main_runner_varying_prob_p_theta_experiment.py --n_runs 4 --dataset cifar --data ../data/
#python main_runner_varying_prob_p_theta_experiment.py --n_runs 4 --dataset gas --data ../data/GAS/

#python main_runner_varying_prob_p_theta_experiment.py --n_runs 100 --dataset cifar --data ../data/
#python main_runner_varying_prob_p_theta_experiment.py --n_runs 100 --dataset gas --data ../data/GAS/

#python main_runner_varying_Ts_experiment.py --n_runs 4 --dataset cifar --data ../data/
#python main_runner_varying_Ts_experiment.py --n_runs 4 --dataset gas --data ../data/GAS/


#python main_runner_varying_Ts_experiment.py --n_runs 100 --dataset cifar --data ../data/
#python main_runner_varying_Ts_experiment.py --n_runs 100 --dataset gas --data ../data/GAS/
