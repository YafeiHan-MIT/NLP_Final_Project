# helper class used for computing information retrieval metrics, including MAP / MRR / and Precision @ x
class Evaluation():
    def __init__(self,data):
        self.data = data
    
    def Precision_at_R(self,precision_at):
        precision_all = []
        for item in self.data:
            item_sel = item[:precision_at]
            count_pos = 0.0
            if len(item_sel)>0:
                for label in item_sel:
                    if label == 1:
                        count_pos += 1
                precision_all.append(count_pos/len(item_sel))
            else:
                precision_all.append(0.0)
        return sum(precision_all)/len(precision_all)
                
#	def Precision(self,precision_at):
#		scores = []
#		for item in self.data:
#			temp = item[:precision_at]
#			if any(val==1 for val in item):###WHY HAVE THIS? 
#				scores.append(sum([1 if val==1 else 0 for val in temp])*1.0 / len(temp) if len(temp) > 0 else 0.0)
#		return sum(scores)/len(scores) if len(scores) > 0 else 0.0
    
    def MAP(self):
        '''
        Mean Average Precision (MAP)
        
        Input: 
            self.data: a list of ranked retrievals' labels(1=relevant,0=not rel)
        Output: 
            float MAP
        '''
        AP = [] #list of Average Precision(AP) for all queries in self.data
        for labels in self.data: #examine each query
            count_pos = 0.0 ##accumulative count of relevant documents 
            Pk = [] #precision of the first (k+1) retrievals for a single query
            last=len(labels) - 1 - labels[::-1].index(1) #find the index for the last occurence of label=1
            for k,label in enumerate(labels): #k: rank of retrieved doc, label: 1 if relevant, 0 not relevant
                if k>last:break
                if label == 1:
                    count_pos += 1.0
                Pk.append(count_pos/(k+1)) #precision for the first (k+1) retrievals
            if len(Pk)>0: 
                AP.append(sum(Pk)/len(Pk))
            else:
                AP.append(0.0)
        return sum(AP)/len(AP) #average over all queries up to the last occurence of the pos example.
    
    def MRR(self):
        '''
        Mean reciprocal rank (MRR)
        MRR = 1/|Q| * sum_j(1/rank_j), where 
            Q: set of all queries,|Q| is the number of queries 
            rank_j: the rank of the first relevant document for query j in Q
            
        '''
        list_RR = [] #list of reciprocal rank for all queries
        for item in self.data:
            if len(item)==0: #no retrieval for current query
                list_RR.append(0.0)
            else:
                for i,label in enumerate(item):
                    if label == 1: #first encountering a relevant document 
                        list_RR.append(1.0/(i+1))  #record 1/rank 
                        break
                    if i==len(item)-1:#reach the end but not find relevant document
                        list_RR.append(0.0)  
        return sum(list_RR)/len(list_RR) if len(list_RR) > 0 else 0.0

              
##My testing code
#data = [[],[]]
#data1 = [[0],[0,1,1],[0,1,1,1,1],[0,1,1,1]]
#e=Evaluation(data)
#e1=Evaluation(data1)
#print e.MAP()
#print e1.MAP()
#print e.MRR()
#print e1.Precision_at_R(2)


                       






