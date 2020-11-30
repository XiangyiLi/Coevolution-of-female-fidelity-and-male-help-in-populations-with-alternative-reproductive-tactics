import SimuBirdEPC
mutationRate=0.01
mutationSize=0.01
sexRatio=0.5
#delta=0
femaleAlpha=0
maleBetaPairing=0
maleBetaEPC=1
init_a=0.05
#rEPO=1.2
nb=50
popSize=5000
tMax=25000
#rep=1
SurvivalFunc=0
NoSneaker=0
TraitMode=0
SimuBirdEPC.SimuBirdEPC(sexRatio, delta, femaleAlpha, maleBetaPairing, maleBetaEPC, init_a, rEPO, nb, popSize, mutationRate, mutationSize, tMax, rep, SurvivalFunc, NoSneaker, TraitMode)