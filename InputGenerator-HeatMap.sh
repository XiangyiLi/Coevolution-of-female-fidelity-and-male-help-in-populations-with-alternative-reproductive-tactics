# Do check the type of output files -- e.g. suppress the trajectories.
declare -a deltaArray=($(seq 0 0.02 1.001))
declare -a rEPOArray=($(seq 0.8 0.01 1.201))
declare -a repArray=($(seq 1 1 1.001))

for delta in "${deltaArray[@]}"
do
	for rEPO in "${rEPOArray[@]}"
	do
		for rep in "${repArray[@]}"
		do
			a="SimuBirdEPC.SimuBirdEPC(sexRatio, "
			b=", femaleAlpha, maleBetaPairing, maleBetaEPC, init_a, "
			c=", nb, popSize, mutationRate, mutationSize, tMax, "
			d=", SurvivalFunc, NoSneaker, TraitMode)"
			inputLine=$a$delta$b$rEPO$c$rep$d
			echo $inputLine
			sed "18s/.*/$inputLine/" InputTemplate1.py > "delta-$delta-rEPO-$rEPO.py"
			echo "#!/bin/bash" > "job_delta_$delta-rEPO-$rEPO.sh"
			echo "#SBATCH -n 1" >> "job_delta_$delta-rEPO-$rEPO.sh"
			echo "#SBATCH --mem 2000" >> "job_delta_$delta-rEPO-$rEPO.sh"
			echo "#SBATCH --time 20:0:0" >> "job_delta_$delta-rEPO-$rEPO.sh"
			echo "#SBATCH --exclude=node07,node08,node09,node10" >> "job_delta_$delta-rEPO-$rEPO.sh"
			echo "python3 delta-$delta-rEPO-$rEPO.py" >> "job_delta_$delta-rEPO-$rEPO.sh"
			sbatch "job_delta_$delta-rEPO-$rEPO.sh"
		done
	done
done
