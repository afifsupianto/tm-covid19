import pandas as pd
import numpy as np
import re

class ddRule():
    def __init__(self, dis):
        self.disease = dis
        self.drugs = {}
    
    def addDrugs(self, dru):
        for i in dru:
            if i in self.drugs:
                self.drugs[i] = self.drugs[i]+1
            else:
                self.drugs[i] = 1

    def countDrugs(self):
        return(len(self.drugs))

    def showRule(self):
        #for drug in self.drugs:
        print(f"{self.disease} -> {self.drugs}")

class ddBatchRules():
    def __init__(self):
        self.brules = []
        self.bdiseases = []
        self.bdrugs = {}
        self.pairRule = {}
    
    def addBRule(self, rul):
        self.brules.append(rul)
    
    def countBRules(self):
        return(len(self.brules))
    
    def showBRules(self):
        #for drug in self.drugs:
        print(f"{self.bdiseases} -> {self.bdrugs}")
    
    def reCalculate(self):
        #listing diseases, drugs, and re-calculate the frequency
        for i in self.brules:
            self.bdiseases.append(i.disease)
            for key, value in i.drugs.items():
                #print(f'key: {key}, val: {value}')
                if key in self.bdrugs:
                    self.bdrugs[key] = self.bdrugs[key]+value
                else:
                    self.bdrugs[key] = value
            #print(self.bdrugs)
        self.pairingRules()

    def pairingRules(self):
        for i in self.brules:
            for key, value in self.bdrugs.items():
                self.pairRule[f'{self.bdiseases} -> {key}'] = value

class ddRules():
    def __init__(self):
        self.rules = []
        self.BR = []
        self.BRrules = []
        self.BRpairrules = {}
        self.selectedBRpairrules = {}
        self.sortedBRpairrules = []
        self.selectedDrugs = set()
        
    def addRule(self, dis, dru):
        self.rules.append(ddRule(dis))
        self.rules[len(self.rules)-1].addDrugs(dru)
        
    def countRules(self):
        return(len(self.rules))
    
    def countPairRules(self):
        val = 0
        for i in self.rules:
            val += i.countDrugs()
        return(val)

    def showRules(self):
        for i in self.rules:
            i.showRule()
    
    def countBatchRules(self):
        return(len(self.selectedBRpairrules))

    def showBatchRules(self):
        for pr in self.selectedBRpairrules.items():
            print(pr)
    
    def showPotentialDrugs(self):
        for dr in self.selectedDrugs:
            print(dr)

    def matrixSimilarity(self):
        return self.arSim
    
    def generateBatchRules(self, thresh):
        #shape array 2-dimension as number of rules for store the similarity
        self.arSim = np.full((len(self.rules), len(self.rules)), -1).astype(float)
        #calculate and fill the similarity to the array
        for i, rul1 in enumerate(self.rules): 
            for j, rul2 in enumerate(self.rules):
                if i!=j:
                    self.arSim[i,j] = self.getSimilarity(rul1, rul2)
        #grouping the rules based on similarity according to the given threshold
        self.BR = []
        for i, val in enumerate(np.argwhere(self.arSim >= thresh)):
            if i==0:
                cur = val[0]
                y = [val[0], val[1]]
            else:
                if val[0]==cur:
                    y = y + [val[1]]
                else:
                    ada = False
                    for j in self.BR:
                        if val[0] in j:
                            ada = True
                            continue
                    
                    if val[0] in y:
                        ada = True
                    
                    if ada==False:
                        self.BR = self.BR + [y]
                        cur = val[0]
                        y = [val[0], val[1]]
        self.BR = self.BR + [y]
        #add the individual rules to the batch rules indices list
        allidx = [*range(0, len(self.arSim), 1)]
        for i in self.BR:
            allidx = set(allidx)-set(i)
            allidx = list(allidx)
        self.BR = self.BR + [allidx]
        #collecting the rules object according to the indices
        for li in self.BR:
            self.BRrules.append(ddBatchRules())
            for j in li:
                self.BRrules[len(self.BRrules)-1].addBRule(self.rules[j])
            self.BRrules[len(self.BRrules)-1].reCalculate()
        #listing the batch rules
        for i in self.BRrules:
            self.BRpairrules = {**self.BRpairrules , **i.pairRule}
        
    def selectRules(self, minSup):
        #selecting and sorting the rule according to given minimum support
        self.selectedBRpairrules = dict((k, v) for k, v in self.BRpairrules.items() if v >= minSup)
        self.sortedBRpairrules = sorted(self.selectedBRpairrules, key=self.selectedBRpairrules.get, reverse=True)
        #listing the potential drugs
        for i in self.sortedBRpairrules:
            self.selectedDrugs.add(i.split(' -> ')[1])

    def getSimilarity(self, r1, r2):
        num1 =[int(s) for s in re.findall(r'\b\d+\b', r1.disease)]
        num2 =[int(s) for s in re.findall(r'\b\d+\b', r2.disease)]
        valSim = 0
        if num1 == num2:
            lstr1 = r1.disease.replace(' ', '-').lower().split('-')
            lstr2 = r2.disease.replace(' ', '-').lower().split('-')
            #print(lstr1, lstr2)
            maxLen = len(lstr1) if len(lstr1) > len(lstr2) else len(lstr2)
            compstr = set(lstr1) & set(lstr2)
            #print(compstr)
            valSim = len(compstr)/maxLen
            #print(valSim)
        return(valSim)

class ddDataset():
    def __init__(self, spath):
        self.lsdatadisease = []
        self.lsdatadrug = []
        self.ckey = []
        self.dfcovid = pd.read_csv(spath)
    
    def selectFeatures(self):
        #selecting fields
        self.seldfcovid = self.dfcovid[["disease", "drug"]]
        #lower all strings
        #self.seldfcovid['disease'] = self.seldfcovid['disease'].str.lower()
        #self.seldfcovid['drug'] = self.seldfcovid['drug'].str.lower()
        #cleaning
        self.seldfcovid = self.seldfcovid.dropna(subset=["disease", "drug"])
        self.lsdatadisease = self.seldfcovid.disease.tolist()
        self.lsdatadrug = self.seldfcovid.drug.tolist()
    
    def setFilterKeyword(self, spath):
        text_file = open(spath, "r")
        self.ckey = text_file.read().split('OR')
        self.ckey = [x.strip() for x in self.ckey]
        self.ckey = [x.lower() for x in self.ckey] 

class ddMain():
    def __init__(self):
        self.data = []
        
    def RunExperiments(self, ms, th):
        self.theRules = ddRules()
        for i in range(len(self.data.lsdatadisease)):
            ls_ds = self.data.lsdatadisease[i].lower().split("|")
            ls_dr = self.data.lsdatadrug[i].lower().split("|")
            for j in range(len(ls_ds)):
                if any(n in ls_ds[j] for n in self.data.ckey):
                    self.theRules.addRule(ls_ds[j], ls_dr)
        
        #self.theRules.generateBatchRules(th)
        #self.theRules.selectRules(ms)
        
    def initDataset(self, spathData, spathKey):
        self.data = ddDataset(spathData)
        self.data.selectFeatures()
        self.data.setFilterKeyword(spathKey)

    
foo = ddMain()
foo.initDataset('dataset/covid_results.csv', 'dataset/covid-keywords.txt')
foo.RunExperiments(15, 0.5)
print(f'Number of Rules: {foo.theRules.countRules()}')
print(f'Number of Pair Rules: {foo.theRules.countPairRules()}')

#print(f'Number of Batch Rules: {foo.theRules.countBatchRules()}')
#foo.theRules.showBatchRules()
