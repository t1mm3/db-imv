#!/bin/bash

processor="SKY"

#data size
r_size_set=("1M")
s_size_set=("50M")
#data skew
r_skew_set=(0 0.5 1)
s_skew_set=(0 0.5 1)
#scalability
thread_nums=(1)

# range=1~40, +4 scalarstatesize
sstatestart=20
sstateend=20
# range=1~20, +1 simdstatesize=5
statestart=5
stateend=5
# paramter 2: sequential prefetch distance
disstart=320
disend=320

# paramter 3:application name
app="NPO"
#app="PIPELINE"
#app="BTS"

dir_name="results"

core_set="16 0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15,"

#keep the same with REPEAT_PROBE
repeat=10

prefetch_path="../include/imv-operator/prefetch.hpp"
#set repeat time
sed -i "/#define\ REPEAT_PROBE/c\#define\ REPEAT_PROBE\ ${repeat}" $prefetch_path


numa_config="-m 0"

SKX_core_set=("1 0," "4 0,2,4,6," "8 0,2,4,6,8,10,12,14," "12 0,2,4,6,8,10,12,14,1,3,5,7," "16 0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15,"  "2 0,16," "8 0,16,2,18,4,20,6,22," "16 0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30," "24 0,16,2,18,4,20,6,22,8,24,10,26,12,28,14,30,1,17,3,19,5,21,7,23," "-32 0-31,")
SKX_core_set_num=(1 4 8 12 16 2 8 16 24 32)

KNL_core_set=("1 0," "-16 0-15," "-32 0-31," "-48 0-47," "-64 0-63," "2 0,64," "-32 0-15,64-79," "-64 0-31,64-95," "-96 0-47,64-111," "-128 0-63,64-127," "4 0,64,128,192," "-64 0-15,64-79,128-143,192-207," "-128 0-31,64-95,128-159,192-223," "-192 0-47,64-111,128-175,192-239," "-256 0-255,")
KNL_core_set_num=(1 16 32 48 64 2 32 64 96 128 4 64 128 192 256)

sudo_passward="claims"

function reset_default_param() {
	#data size
	r_size_set=("1M")
	s_size_set=("50M")
	#data skew
	r_skew_set=(1)
	s_skew_set=(1)
	#scalability
	thread_nums=(1)

	# range=1~40, +4 scalarstatesize
	sstatestart=20
	sstateend=20
	# range=1~20, +1 simdstatesize=5
	statestart=5
	stateend=5

	# paramter 2: sequential prefetch distance
	disstart=320
	disend=320

	# paramter 3:application name
	#app="NPO"
	#app="PIPELINE"
	#app="BTS"

	dir_name="results"

	if [[ $processor == "KNL" ]]; then
		core_set="-64 0-63,"
	else
		core_set="16 0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15,"
	fi
	sed -i "1s/.*/$core_set/" cpu-mapping.txt
sudo -S su << EOF
$sudo_passward
        free -g
        sync
        echo 3 > /proc/sys/vm/drop_caches
        free -g
EOF
        echo "##drop caches##"
}

function run_a_cycle(){
	#################NOTE: 请不要注释掉以下循环，修改参数请在上面修改###########################
	for ((k=0;k<${#r_skew_set[@]};k++)) do
#		for ((l=0;l<${#s_skew_set[@]};l++)) do
		l=$k
			for ((t=0;t<${#thread_nums[@]};t++)) do
				for ((i=0;i<${#r_size_set[@]};i++)) do
					for ((j=0;j<${#s_size_set[@]};j++)) do
						echo "thread_num: ${thread_nums[t]}"
						echo "r_size: ${r_size_set[i]}"
						echo "s_size: ${s_size_set[j]}"
						echo "r_skew: ${r_skew_set[k]}"
						echo "s_skew: ${s_skew_set[l]}"
						#########################################
						echo "ARGS dis: $2 SIMDstatesize: $3 scalarstatesize: $4 thread_num: ${thread_nums[t]} r_size: ${r_size_set[i]} s_size: ${s_size_set[j]} r_skew: ${r_skew_set[k]} s_skew: ${s_skew_set[l]}" >> $1
						echo "time  numactl  ${numa_config}  ./operator -a $app -n ${thread_nums[t]} --r-file=r_skew=${r_skew_set[k]}_size=${r_size_set[i]} --s-file=s_skew=${s_skew_set[l]}_size=${s_size_set[j]}_max=${r_size_set[i]}"
						time  numactl  ${numa_config}  ./operator -a $app -n ${thread_nums[t]} --r-file=r_skew=${r_skew_set[k]}_size=${r_size_set[i]} --s-file=s_skew=${s_skew_set[l]}_size=${s_size_set[j]}_max=${r_size_set[i]}  >> $1
#					done;
				done;
			done;
		done;
#		echo "!!!" >> $1
	done;
}
function file_loop() {
	for((pdis=$disstart;pdis<=$disend;pdis+=64)) do
		#############修改prefetch.h文件##################
		sed -i "/#define\ PDIS/c\#define\ PDIS\ ${pdis}" $prefetch_path
		for((scalarstatesize=$sstatestart,simdstatesize=$statestart;scalarstatesize<=$sstateend;scalarstatesize+=3,simdstatesize+=1)) do
		        output_file=${dir_name}/${app}_pdis_${pdis}_simdstatesize_${simdstatesize}_scalarstatesize_${scalarstatesize}.txt
		        echo $output_file
		        #############修改prefetch.h文件##############
		        sed -i "/#define\ SIMDStateSize/c\#define\ SIMDStateSize\ ${simdstatesize}" $prefetch_path
		        sed -i "/#define\ ScalarStateSize/c\#define\ ScalarStateSize\ ${scalarstatesize}" $prefetch_path
			echo "pdis_${pdis}_simdstatesize_${simdstatesize}_scalarstatesize_${scalarstatesize}" >> chg_prefetch_h.log
		        cat $prefetch_path >> chg_prefetch_h.log
		        ##########################
		        make
		        echo "start to run..."
		        run_a_cycle $output_file ${pdis} ${simdstatesize} ${scalarstatesize}
		        echo "end run..."
		        results_file=${dir_name}/${app}_results_pdis_${pdis}_simdstatesize_${simdstatesize}_scalarstatesize_${scalarstatesize}.csv
		        echo "start to parse..."
		        echo $output_file
		        #############解析结果的脚本，第一个参数是源文件，第二个参数是结果文件，第三个参数是你repeat的次数################
		        python test_results_parser.py $output_file $results_file $repeat
			rm ${output_file}
		        echo "end parse..."
		done;
	done;
}
## change range and step in file_loop()
function expr_group_size() {
	reset_default_param
	# set paramaters for this experiment
	sstatestart=1
	sstateend=23
	statestart=1
	stateend=10
	# paramter 3:application name
	dir_name="results_group"_$(date +%F-%T)

	mkdir $dir_name

	file_loop
	python test_results_merge.py $dir_name merged_results.csv
}
function expr_pdis() {
	reset_default_param
	# set parameters for this experiemnt 
	# paramter 2: sequential prefetch distance
	disstart=0
	disend=320
	dir_name="results_pdis"_$(date +%F-%T)

	mkdir $dir_name
	file_loop
	python test_results_merge.py $dir_name merged_results.csv
}
function expr_scale() {
	reset_default_param
	dir_name="results_scale"_$(date +%F-%T)
	mkdir $dir_name
	if [[ $processor == "SKX" ]]; then
		for ((ct=0;ct<${#SKX_core_set[@]};ct++)) do
			core_set=${SKX_core_set[ct]}
			thread_nums=(${SKX_core_set_num[ct]})
			sed -i "1s/.*/$core_set/" cpu-mapping.txt
			file_loop
		done;
	else
		for ((ct=0;ct<${#KNL_core_set[@]};ct++)) do
			core_set=${KNL_core_set[ct]}
			thread_nums=(${KNL_core_set_num[ct]})
			sed -i "1s/.*/$core_set/" cpu-mapping.txt
			file_loop
		done;
	fi	
	python test_results_merge.py $dir_name merged_results.csv
}
function expr_skew() {
	reset_default_param
	dir_name="results_skew"_$(date +%F-%T)
	mkdir $dir_name
	r_skew_set=(0 0.5 1)
	s_skew_set=(0 0.5 1)
	file_loop
	python test_results_merge.py $dir_name merged_results.csv
}

function expr_data_size() {
	reset_default_param
	dir_name="results_data_size"_$(date +%F-%T)
	mkdir $dir_name

	r_size_set=("16K" "64K" "256K" "512K" "1M" "4M" "16M" "64M")
	file_loop
	python test_results_merge.py $dir_name merged_results.csv
}
##compare with skew
function expr_huge_page() {
	reset_default_param
	#enable huge page
	dir_name="results_huge_enable"_$(date +%F-%T)
	mkdir $dir_name
	r_skew_set=(1)
	s_skew_set=(1)
	r_size_set=("1M" "64M")
	file_loop
	python test_results_merge.py $dir_name merged_results.csv

	#disable huge page
	dir_name="results_huge_disable"_$(date +%F-%T)
	mkdir $dir_name
	# disable huge pages
sudo -S su << EOF
$sudo_passward
	echo never > /sys/kernel/mm/transparent_hugepage/defrag
	echo never > /sys/kernel/mm/transparent_hugepage/enabled
EOF
	echo "##after disable##"
	# comare with expr_skew under enabling huge pages
	cat /sys/kernel/mm/transparent_hugepage/defrag 
	cat /sys/kernel/mm/transparent_hugepage/enabled 
	
	file_loop
	python test_results_merge.py $dir_name merged_results.csv

	# enalbe huge pages
sudo -S su << EOF
$sudo_passward
	echo always > /sys/kernel/mm/transparent_hugepage/enabled
	echo always > /sys/kernel/mm/transparent_hugepage/defrag
EOF
	echo "##re enable##"
	# comare with expr_skew under enabling huge pages
	cat /sys/kernel/mm/transparent_hugepage/defrag 
	cat /sys/kernel/mm/transparent_hugepage/enabled 
}

function expr_smt() {
	reset_default_param
	dir_name="results_smt"_$(date +%F-%T)
	mkdir $dir_name
	if [[ $processor == "SKX" ]]; then
                # one physical core with out SMT, but SMV
		ct=0
		core_set=${SKX_core_set[ct]}
		thread_nums=(${SKX_core_set_num[ct]})
		sed -i "1s/.*/$core_set/" cpu-mapping.txt
		file_loop

                # one physical core with SMT and SMV
		ct=5
		core_set=${SKX_core_set[ct]}
		thread_nums=(${SKX_core_set_num[ct]})
		sed -i "1s/.*/$core_set/" cpu-mapping.txt
		file_loop

                # disable SMV
		sstatestart=1
		sstateend=1
		statestart=1
		stateend=1
		disstart=0
		disend=0
		# one physical core, no SMV, SMT
                ct=0
                core_set=${SKX_core_set[ct]}
                thread_nums=(${SKX_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop
		# one physical core, SMV + SMT
		ct=5
		core_set=${SKX_core_set[ct]}
		thread_nums=(${SKX_core_set_num[ct]})
		sed -i "1s/.*/$core_set/" cpu-mapping.txt
		file_loop		
		
	else
		# one physical core with optimal group size
                ct=0
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop

                # one physical core with 2 SMT, and SMV
                ct=5
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop
                # one physical core with 4 SMT, and SMV
                ct=10
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop

		# disable SMV
                sstatestart=1
                sstateend=1
                statestart=1
                stateend=1
                disstart=0
                disend=0
                # one physical core with 1 SMT, but withou SMV
                ct=0
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop
                # one physical core with 2 SMT, but withou SMV
                ct=5
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop
                # one physical core with 4 SMT, but withou SMV
                ct=10
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop

	fi	
	python test_results_merge.py $dir_name merged_results.csv
}
function expr_perf() {
	reset_default_param
	dir_name="results_perf"_$(date +%F-%T)
	mkdir $dir_name
	r_skew_set=(0 0.5 1)
	s_skew_set=(0 0.5 1)

	if [[ $processor == "SKX" ]]; then
         # all threads
		ct=9
		core_set=${SKX_core_set[ct]}
		thread_nums=(${SKX_core_set_num[ct]})
		sed -i "1s/.*/$core_set/" cpu-mapping.txt
		numa_config="-m 0,1"
		file_loop	
		# one socket
		ct=7
		core_set=${SKX_core_set[ct]}
		thread_nums=(${SKX_core_set_num[ct]})
		sed -i "1s/.*/$core_set/" cpu-mapping.txt
		numa_config="-m 0"
		file_loop	
		# all physical cores
		ct=4
		core_set=${SKX_core_set[ct]}
		thread_nums=(${SKX_core_set_num[ct]})
		sed -i "1s/.*/$core_set/" cpu-mapping.txt
		numa_config="-m 0,1"
		file_loop	
	else
		# single
                ct=0
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop
		#all
                ct=14
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop
	fi	
	python test_results_merge.py $dir_name merged_results.csv
}
function expr_build_agg() {
	reset_default_param
	dir_name="results_build_agg"_$(date +%F-%T)
	mkdir $dir_name
	r_skew_set=(0 0.5 1)
	s_skew_set=(0 0.5 1)
	r_size_set=("1M")
	s_size_set=("50M")
	if [[ $processor == "SKX" ]]; then
         # single thread
		ct=0
		core_set=${SKX_core_set[ct]}
		thread_nums=(${SKX_core_set_num[ct]})
		sed -i "1s/.*/$core_set/" cpu-mapping.txt
		numa_config="-m 0"
		file_loop	
		# one socket
		ct=9
		core_set=${SKX_core_set[ct]}
		thread_nums=(${SKX_core_set_num[ct]})
		sed -i "1s/.*/$core_set/" cpu-mapping.txt
		numa_config="-m 0,1"
		file_loop	

	else
		# single
                ct=0
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop
		#all
                ct=14
                core_set=${KNL_core_set[ct]}
                thread_nums=(${KNL_core_set_num[ct]})
                sed -i "1s/.*/$core_set/" cpu-mapping.txt
                file_loop
	fi	
	python test_results_merge.py $dir_name merged_results.csv
}
function gen_data() {
	reset_default_param
	thread_nums=(8)
	r_size_set=("16384" "65536" "262144" "524288" "1048576" "4194304" "16777216" "67108864")
	# make sure |r_size_set| < 10
	t=0
	for ((i=0;i<${#r_size_set[@]};i++)) do 
		numactl -C 0-16 ./operator -a GEN -n 16  --r-size=${r_size_set[i]}  --s-size=52428800 --r-skew=0 --s-skew=0 &
		((t++))
		numactl -C $t ./operator -a GEN -n 16  --r-size=${r_size_set[i]}  --s-size=52428800 --r-skew=0.5 --s-skew=0.5 &
		((t++))
		numactl -C $t ./operator -a GEN -n 16  --r-size=${r_size_set[i]}  --s-size=52428800 --r-skew=1 --s-skew=1 
		((t++))
	done;
}
function run_all() {
	#expr_smt
	expr_group_size
	expr_data_size
	expr_skew
	expr_huge_page
	#expr_pdis
	expr_scale
}
function expr_write_apps() {
	app="BUILD"
	expr_build_agg
	app="AGG"
	expr_build_agg
}	
function expr_apps() {
	app="BTS"
	run_all
	app="NPO"
	run_all
}		

echo "What is the processor ? SKX : KNL?"

read processor

if [[ $processor == "SKX" ||  $processor == "KNL" ]]; then
	echo "the processor is $processor, valid"
else 
	echo "the processor is $processor, but invalid"
	exit
fi


if [[ $processor == "KNL" ]]; then
	sudo_passward="hsdzhfang"
	core_set="-64 0-63,"
	# 1 for MCDRAM
	numa_config="-m 0"
else
	sudo_passward="claims"
	core_set="16 0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15,"
	numa_config="-m 0"
fi

sed -i "1s/.*/$core_set/" cpu-mapping.txt

echo "input expr name : 
GROUP: group size
DIS: prefetch distance
SCALE: scalability
SKEW: data skew
GEN: generate data
DATA: data size
PAGE: huge page
SMT: smt
PERF: compare using all skew all cores
APP: all applications, NPO+BTS
ALL: all experiments, default NPO
------------------"
read expr_name

# enalbe huge pages
sudo -S su << EOF
$sudo_passward
        echo always > /sys/kernel/mm/transparent_hugepage/enabled
        echo always > /sys/kernel/mm/transparent_hugepage/defrag
EOF
        echo "##re enable huge page##"
        # comare with expr_skew under enabling huge pages
        cat /sys/kernel/mm/transparent_hugepage/defrag
        cat /sys/kernel/mm/transparent_hugepage/enabled


if [[ ${expr_name} == 'DIS' ]]; then 
	expr_pdis
elif [[ ${expr_name} == 'GROUP' ]]; then	
	expr_group_size
elif [[ ${expr_name} == 'GEN' ]]; then	
	gen_data
elif [[ ${expr_name} == 'DATA' ]]; then	
	expr_data_size
elif [[ ${expr_name} == 'SCALE' ]]; then	
	expr_scale
elif [[ ${expr_name} == 'SKEW' ]]; then	
	expr_skew
elif [[ ${expr_name} == 'PAGE' ]]; then	
	expr_huge_page
elif [[ ${expr_name} == 'SMT' ]]; then	
	expr_smt
elif [[ ${expr_name} == 'APP' ]]; then	
	expr_apps
elif [[ ${expr_name} == 'WRITE' ]]; then	
	expr_write_apps
elif [[ ${expr_name} == 'PERF' ]]; then	
	app="NPO"
	expr_perf
	app="BTS"
	expr_perf
elif [[ ${expr_name} == 'ALL' ]]; then
	expr_scale
#	expr_smt
#	expr_group_size
#	expr_pdis
	expr_perf
	expr_skew
	expr_data_size
	expr_huge_page
else 
	echo -n ${expr_name}
	echo " not found"
fi


