def SimuBirdEPC(sexRatio, delta, femaleAlpha, maleBetaPairing, maleBetaEPC, init_a, rEPO, nb, popSize, mutationRate, mutationSize, tMax, rep, SurvivalFunc, NoSneaker, TraitMode):
    #sexRatio is the proportion of males in the population
    #delta is the efficiency of male mate guarding
    #femaleAlpha is the factor to scale female fecundity
    #maleBetaPairing  is the factor to scale male competitiveness in forming a breeding pair
    #maleBetaEPC is the factor to scale male competitiveness in extra-pair copulations
    #rEPO is the survival advantage of extra-pair offspring compared to within-pair offspring
    #nb is the baseline fecundity of females
    #popSize is the total adult population size
    #tMax is the number of generations that will be simulated
    #rep is the number of independent simulation realizations that will be run for each parameter combination
    #if SurvivalFunc=0, the survival rate of offspring is proportional to the geometric mean of mother condition and father investment -- unpaired females cannot reproduce
    #if SurvivalFunc=1, the survival rate of offspring is proportional to the arithmetic mean of mother condition and father investment -- unpaired females can also reproduce, but take care of the offspring alone
    #if NoSneaker=1, all males take part in pair formation, mutation at this locus is also turned off
    #if TraitMode=0, there is no individual variation
    #if TraitMode=1, males and females express a shared trait locus with optimal allelic value at 0.5
    #if TraitMode=2, male and female conditions are determined by seperate loci, male has an allelic optimum at 1, female has an allelic optimum at 0, while there is no shared locus
    #if TraitMode=3, males and females have a shared locus and also sex-specific loci, but there is no intralocus sexual conflict, because male-trait determining locus is expressed only in males and vice versa for females.
    #if TraitMode=4, males and females have a shared locus that introduce intralocus sexual conflict
    
    import numpy as np
    import itertools
    
    tCollectStart=tMax-2000
    
    #First, construct the genetic structure and state property of an individual
    #The numbers are indices
    isMale=0 # sex of individuals: male->1, female->0
    lA=[1,2] # a male is a sneaker only if both alleles are 0 (a)
    lT=[3,4] # trait locus determining an individual's degree of adaptation, with an optimal allelic value of 0.5
    lTf=[5,6] # trait locus only expressed in females, with an optimal allelic value of 0.2
    lTm=[7,8] # trait locus only expressed in males, with an optimal allelic value of 0.8
    lTsc=[9,10] # trait locus expressed in both males and females, introducing intralocus sexual conflict, because male has an optimum at 0.8, while female has an optimum at 0.2
    h=[11,12] # locus controling the degree of male help, only expressed in males
    u=[13,14] # female fidelity locus, only expressed in females
    c=15 # condition of an individual
    sneaker=16 # value equals 1 if a male has the sneaker genotype
    cH=17 # share of male activity devoted to helping his female
    cEP=18 #share of male activity devoted to extra-pair copulation
    nO=19 # number of produced offspring for females
    
    nEntries=20 # because we start counting from 0
    
    evolvingLoci = list(itertools.chain.from_iterable([lT,lTf,lTm,lTsc,h,u])) # list of evolving loci with continuous allelic values
    nEvolvingLoci = int(len(evolvingLoci)/2)
    
    #initializing the population
    newPop = np.zeros(shape=(popSize, nEntries))
    # assign the sex of individuals
    newPop[:, isMale] = np.random.uniform(low=0,high=1,size=popSize)<sexRatio
    # initialize the lA locus, depending on if sneakers (the 'a' allele is present in the population)
    if NoSneaker ==1: # remember also to turn off mutation at th lA locus in this case
        newPop[:,lA] = np.ones(shape=(popSize,2))
    else:
        newPop[:,lA]=(np.random.uniform(low=0,high=1,size=(popSize, 2))<(1-init_a))*1 #initial frequency of a is 1-0.95=0.05
    # initialize the evolving loci
    newPop[:,evolvingLoci] = np.random.uniform(low=0.4, high=0.6, size=(popSize, 2*nEvolvingLoci))
    
    #collect dynamics data
    freqUnpairedM=[]
    freqSneaker=[]
    freq_a=[]
    medianH=[]
    medianU=[]
    freqEPO=[]
    freqEPOAdult=[]
    growthRate=[]
    
    t=0
    # life cycle of a generation
    while t < tMax:
        t += 1
        print("Generation: "+str(t))
        # generation turnover
        Pop=newPop
        
        #Males decide whether to take part in pair-formation by their genotype
        #males with aa genotype are "sneakers"
        idSneakers=np.where(np.logical_and(Pop[:,isMale]==1, np.sum(Pop[:,lA], axis=1)==0))[0]
        Pop[idSneakers,sneaker]=1
        
        #the "sneakers" skip pair formation, while the rest of males and females start to form monogamous pairs
        idMales = np.where(Pop[:,isMale]==1)[0]
        idMalesAvailable=np.where(np.logical_and(Pop[:,isMale]==1, Pop[:,sneaker]==0))[0]
        idFemales = np.where(Pop[:,isMale]==0)[0]
        
        #calculate the condition for males and females
        if TraitMode==0: #all individuals simply have the same condition of 1.
            Pop[:,c] = 1 
        elif TraitMode==1: #individuals only express one shared locus lT with an optimum value at 0.5
            Pop[:,c] = 1-np.absolute(0.5-np.mean(Pop[:,lT],axis=1))
        elif TraitMode==2: #females only express locus lTf with an optimum at 0.2 while males only express locus lTm with an optimum at 0.8
            Pop[idFemales,c] = 1-np.absolute(0.2-np.mean(Pop[idFemales,:][:,lTf],axis=1))
            Pop[idMales,c] = 1-np.absolute(0.8-np.mean(Pop[idMales,:][:,lTm],axis=1))
        elif TraitMode==3: #The shared Locus lT and sex-specific loci have equal weight in determining individual condition
            Pop[idFemales,c] = (2-np.absolute(0.5-np.mean(Pop[idFemales,:][:,lT],axis=1))-np.absolute(0.2-np.mean(Pop[idFemales,:][:,lTf],axis=1)))/2
            Pop[idMales,c] = (2-np.absolute(0.5-np.mean(Pop[idMales,:][:,lT],axis=1))-np.absolute(0.8-np.mean(Pop[idMales,:][:,lTm],axis=1)))/2
        elif TraitMode==4: #males and females have a shared locus (lTsc) that introduce intralocus sexual conflict
            Pop[idFemales,c] = (2-np.absolute(0.2-np.mean(Pop[idFemales,:][:,lTsc],axis=1))-np.absolute(0.2-np.mean(Pop[idFemales,:][:,lTf],axis=1)))/2
            Pop[idMales,c] = (2-np.absolute(0.8-np.mean(Pop[idMales,:][:,lTsc],axis=1))-np.absolute(0.8-np.mean(Pop[idMales,:][:,lTm],axis=1)))/2
        else:
            print("Please check the TraitMode, must be either 0, 1, 2, or 3, 4!")
            break
        
        #Females start to choose a partner from the available males to form monogamous pairs
        #The process continues until all individuals of the rarer sex are all paired.
        nMatingPairs=min(len(idFemales), len(idMalesAvailable))
        matingPairs=-1*np.ones([nMatingPairs,2])
        relativeCondF=Pop[idFemales,c]**femaleAlpha
        relativeCondF=relativeCondF/np.sum(relativeCondF)
        matingPairs[:,0]=np.random.choice(idFemales, size=nMatingPairs, replace=False, p=relativeCondF)
        relativeCondM=Pop[idMalesAvailable,c]**maleBetaPairing
        relativeCondM=relativeCondM/np.sum(relativeCondM)
        matingPairs[:,1]=np.random.choice(idMalesAvailable, size=nMatingPairs, replace=False, p=relativeCondM)
        matingPairs=matingPairs.astype(int)
        
        #Identify paired females, unpaired females, paired males and unpaird males (including sneakers and "leftover" normal males)
        idPairedF=matingPairs[:,0]
        idUnpairedF=np.setdiff1d(idFemales,idPairedF)
        idPairedM=matingPairs[:,1]
        idUnpairedNormalM=np.setdiff1d(idMalesAvailable,idPairedM)
        idUnpairedM=np.concatenate((idUnpairedNormalM, idSneakers), axis=None)
        
        #Males devide their activities into helping partner and extra-pair copulation, unpaired males only attempt extra-pair copulations
        #proportion of time spent on helping the social female
        propH=np.mean(Pop[idPairedM][:,h],axis=1)
        Pop[idPairedM,cH]=Pop[idPairedM,c]*propH
        Pop[idPairedM,cEP]=Pop[idPairedM,c]*(1-propH)
        Pop[idUnpairedM,cEP]=Pop[idUnpairedM,c]
        
        #Calculate the number of viable offspring each female can produce
        if SurvivalFunc == 0: # unpaired females cannot reproduce, the survival rate of offspring is the geometric mean of mother condition and father help
            Pop[idPairedF,nO]=np.rint(nb*np.sqrt(Pop[idPairedF,c]*Pop[idPairedM,cH]))
            idPairedMoms=np.where(Pop[:,nO]>0)[0]
        if SurvivalFunc == 1: # unpaired females can also produce, the survival rate of offspring is the arithmetic mean of mother condition and father help, for the unpaired females, father help equals 0
            Pop[idPairedF,nO]=np.rint(nb*(Pop[idPairedF,c]+Pop[idPairedM,cH])/2)
            Pop[idUnpairedF,nO]=np.rint(nb*Pop[idUnpairedF,c]/2)
            idPairedMoms=idPairedF[np.where(Pop[idPairedF,nO>0])[0]]
            idUnpairedMoms=idUnpairedF[np.where(Pop[idUnpairedF,nO>0])[0]]
            
        #assigning a father to each offspring
        parentsEPO=[] # extra-pair offspring, in each element: [idMother, idFather]
        parentsWPO=[] # within-pair offspring, in each element: [idMother, idFather]
        nTotalEPO=0 # accumulate the total numbers of extra-pair offspring
        nTotalWPO=0 # accumulate the total numbers of within-pair offspring
        
        # choose father for the offspring produced by paired mothers
        for i in idPairedMoms:
            #find the id of the social father
            idSocialFather=matingPairs[np.where(matingPairs[:,0]==i),1][0,0]
            #probability of producing EPO
            probEPO=(1-np.mean(Pop[i,u]))*(1-delta*np.mean(Pop[idSocialFather,h]))
            #number of offspring produced by the particular female
            nTotalO=int(Pop[i,nO])
            #number of extra-pair offspring produced by the female
            nEPO=np.sum(np.random.uniform(low=0,high=1,size=nTotalO)<probEPO)
            #number of within-pair offspring
            nWPO=nTotalO-nEPO
            
            
            if nEPO>0:
                nTotalEPO += nEPO
                #entering parents ID for extra-pair offspring
                idEPmates=np.delete(idMales,np.where(idMales==idSocialFather)) #The social father cannot be an extra-pair mate
                relativeCondEPMales=Pop[idEPmates,cEP]**maleBetaEPC
                relativeCondEPMales=relativeCondEPMales/np.sum(relativeCondEPMales)
                idEPFathers=np.random.choice(idEPmates, size=nEPO, replace=True, p=relativeCondEPMales)
                idParentsEPO=np.transpose(np.append([i*np.ones(nEPO)],[idEPFathers],axis=0))
                idParentsEPO=idParentsEPO.astype(int).tolist()
                parentsEPO.extend(idParentsEPO)
            if nWPO>0:
                nTotalWPO += nWPO
                #entering parentsID for within-pair offspring
                parentsWPO.extend([[i,idSocialFather]]*nWPO)
                
        if SurvivalFunc==1:
            for i in idUnpairedMoms:
                # we consider all offspring produced by unpaired females as extra-pair offspring
                nEPO=int(Pop[i,nO])
                nTotalEPO += nEPO
                relativeCondEPMales=Pop[idMales,cEP]**maleBetaEPC
                relativeCondEPMales=relativeCondEPMales/np.sum(relativeCondEPMales)
                idEPFathers=np.random.choice(idMales, size=nEPO, replace=True, p=relativeCondEPMales)
                idParentsEPO=np.transpose(np.append([i*np.ones(nEPO)],[idEPFathers],axis=0))
                idParentsEPO=idParentsEPO.astype(int).tolist()
                parentsEPO.extend(idParentsEPO)
            
        # Population size culling --survival to adults. Extra-pair offspring has a survival advantage because of maternal effect
        #convert parents id arrays into numpy arrays
        parentsEPO=np.asarray(parentsEPO) 
        parentsWPO=np.asarray(parentsWPO)
        nSurvivedEPO=int(popSize*(rEPO*nTotalEPO/(rEPO*nTotalEPO+nTotalWPO)))
        nSurvivedWPO=popSize-nSurvivedEPO
      
        #test whether population size can sustain, when not, break
        nTotalO=nTotalEPO+nTotalWPO
        if (nTotalEPO<nSurvivedEPO or nTotalWPO<nSurvivedWPO or nTotalEPO==0 or nTotalWPO==0):
            print("fecundity is too low! Total number of EPO is " + str(nTotalEPO) +", Total number of WPO is " + str(nTotalWPO))
            break
        else:
            parentsEPO = np.random.permutation(parentsEPO)
            parentsEPO = parentsEPO[0:nSurvivedEPO]
            parentsWPO = np.random.permutation(parentsWPO)
            parentsWPO = parentsWPO[0:nSurvivedWPO]
        
        Offspring = np.concatenate((parentsEPO,parentsWPO),axis=0)
        
        #Building the "genomes" of the next generation
        newPop=np.zeros(shape=(popSize,nEntries))
        newPop[:,isMale]=np.random.uniform(low=0,high=1,size=popSize)<sexRatio
        
        nInheritedLoci=nEvolvingLoci+1 #including lA and the evolving loci with continuous allelic values
        idMoms=Offspring[:,0]
        idDads=Offspring[:,1]
        momMatrix=Pop[idMoms,1:1+2*nInheritedLoci] 
        dadMatrix=Pop[idDads,1:1+2*nInheritedLoci]
        #only one allele at each locus comes from mother
        fromMomAllele1=np.random.randint(low=0,high=2, size=(popSize,nInheritedLoci)) 
        fromMomAllele2=1-fromMomAllele1
        momAlleles=np.empty((popSize,2*nInheritedLoci))
        momAlleles[:,::2]=fromMomAllele1
        momAlleles[:,1::2]=fromMomAllele2
        #the other allele must come from father
        dadAlleles=1-momAlleles
        OffsInheritedLoci = momAlleles * momMatrix + dadAlleles * dadMatrix
        
        #Mutations happen with probability mutationRate at the lA locus
        if NoSneaker==0: #sneaker is allowed
            mutOrNot = np.random.uniform(low=0, high=1, size=(popSize, len(lA)))<mutationRate
            OffsInheritedLoci[:,0:2]=OffsInheritedLoci[:,0:2]*(1-mutOrNot) + (1-OffsInheritedLoci[:,0:2]) * mutOrNot
        
        #Mutations happen at each continuous evolving loci (lT, h, and u) with mutationRate and the size follows a normal distribution with sd mutationSize
        mutOrNot = np.random.uniform(low=0,high=1,size=(popSize,nEvolvingLoci*2)) < mutationRate
        mutSize = np.random.normal(loc=0, scale=mutationSize, size=(popSize,nEvolvingLoci*2))
        mutations = mutOrNot*mutSize
        OffsInheritedLoci[:,2::] = OffsInheritedLoci[:,2::] + mutations
        # restrict the allelic value to between 0 and 1
        OffsInheritedLoci=np.minimum(np.maximum(OffsInheritedLoci,np.zeros_like(OffsInheritedLoci)),np.ones_like(OffsInheritedLoci))
        newPop[:,1:2*nInheritedLoci+1]=OffsInheritedLoci
        
        #collecting Statistics
        freqUnpairedM.append(len(idUnpairedM)/len(idMales))
        freqSneaker.append(len(idSneakers)/len(idMales))
        freq_a.append(1-np.sum(Pop[:,lA],axis=None)/(2*popSize))
        medianH.append(np.median(Pop[:,h],axis=None))
        medianU.append(np.median(Pop[:,u],axis=None))
        freqEPO.append(nTotalEPO/nTotalO)
        freqEPOAdult.append(nSurvivedEPO/popSize)
        growthRate.append(nTotalO/popSize)
    
        
    #At the end of simulation    
    if t==tMax: # the simulation did not break prematurally
        meanFreqUnpairedM=np.mean(freqUnpairedM[tCollectStart::])
        meanFreqSneaker=np.mean(freqSneaker[tCollectStart::])
        meanFreq_a=np.mean(freq_a[tCollectStart::])
        meanMedianH=np.mean(medianH[tCollectStart::])
        meanMedianU=np.mean(medianU[tCollectStart::])
        meanFreqEPO=np.mean(freqEPO[tCollectStart::])
        meanFreqEPOAdult=np.mean(freqEPOAdult[tCollectStart::])

        OutputStat=[delta,rEPO,sexRatio,meanFreqUnpairedM,meanFreqSneaker,meanFreq_a,meanMedianH,meanMedianU,meanFreqEPO,meanFreqEPOAdult]
        np.savetxt("Output-rEPO-"+str(rEPO)+"-delta-"+str(delta)+".csv",OutputStat,delimiter=",")
        
        # Trajectories
        np.savetxt("Traj-rEPO-"+str(rEPO)+"-delta-"+str(delta)+"-freqUnpairedM-rep-"+str(rep)+".csv",freqUnpairedM,delimiter=",")
        np.savetxt("Traj-rEPO-"+str(rEPO)+"-delta-"+str(delta)+"-freqSneaker-rep-"+str(rep)+".csv",freqSneaker,delimiter=",")
        np.savetxt("Traj-rEPO-"+str(rEPO)+"-delta-"+str(delta)+"-freq_a-rep-"+str(rep)+".csv",freq_a,delimiter=",")
        np.savetxt("Traj-rEPO-"+str(rEPO)+"-delta-"+str(delta)+"-medianH-rep-"+str(rep)+".csv",medianH,delimiter=",")
        np.savetxt("Traj-rEPO-"+str(rEPO)+"-delta-"+str(delta)+"-medianU-rep-"+str(rep)+".csv",medianU,delimiter=",")
        np.savetxt("Traj-rEPO-"+str(rEPO)+"-delta-"+str(delta)+"-freqEPO-rep-"+str(rep)+".csv",freqEPO,delimiter=",")
        np.savetxt("Traj-rEPO-"+str(rEPO)+"-delta-"+str(delta)+"-freqEPOAdult-rep-"+str(rep)+".csv",freqEPOAdult,delimiter=",")
        np.savetxt("Traj-rEPO-"+str(rEPO)+"-delta-"+str(delta)+"-growthRate-rep-"+str(rep)+".csv",growthRate,delimiter=",")
        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    