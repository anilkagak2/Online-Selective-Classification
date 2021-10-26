
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

# Final results with this setting
#n_runs=100 #4
#python process_data_varying_p_theta_exp.py --n_runs 200 --dataset gas --data ../data/GAS/ --scheme 1
#python process_data_varying_p_theta_exp.py --n_runs 200 --dataset gas --data ../data/GAS/ --scheme 1 --ymax 0.12
#python process_data_varying_p_theta_exp.py --n_runs 200 --dataset gas --data ../data/GAS/ --scheme 2 --ymax 0.12
#python process_data_varying_p_theta_exp.py --n_runs 100 --dataset cifar --data ../data/ --scheme 2
#python process_data_varying_p_theta_exp.py --n_runs 100 --dataset cifar --data ../data/ --scheme 1 --ymax 0.09
#python process_data_varying_Ts_exp.py --n_runs 100 --dataset cifar --data ../data/ --scheme 2 --incr_th 0.002 --result_lbl "" 
#python process_data_varying_Ts_exp.py --n_runs 400 --dataset cifar --data ../data/ --scheme 1 --incr_th 0.001  


python process_data_varying_Ts_exp.py --n_runs 100 --dataset gas --data ../data/GAS/ --scheme 1 --incr_th 0.001 --max_th 0.02  
python process_data_varying_Ts_exp.py --n_runs 100 --dataset gas --data ../data/GAS/ --scheme 2 --incr_th 0.001 --max_th 0.02  
python process_data_varying_Ts_exp.py --n_runs 100 --dataset cifar --data ../data/ --scheme 1 --incr_th 0.001 --max_th 0.02  
python process_data_varying_Ts_exp.py --n_runs 100 --dataset cifar --data ../data/ --scheme 2 --incr_th 0.001 --max_th 0.02  

#python process_data_varying_Ts_exp.py --n_runs 100 --dataset gas --data ../data/GAS/ --scheme 1 --incr_th 0.001 --max_th 0.005  
#python process_data_varying_Ts_exp.py --n_runs 100 --dataset gas --data ../data/GAS/ --scheme 2 --incr_th 0.001 --max_th 0.006  

#python process_data_varying_Ts_exp.py --n_runs 200 --dataset gas --data ../data/GAS/ --scheme 1 --incr_th 0.001 --max_th 0.016 
#python process_data_varying_Ts_exp.py --n_runs 200 --dataset gas --data ../data/GAS/ --scheme 2 --incr_th 0.001 --max_th 0.016 

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
