import pandas as pan
import numpy as np
from scipy.spatial import distance 
from sklearn.metrics import mean_squared_error 
from sklearn.metrics import r2_score

class DecisionTree:
    def init(self):
        self.training=[]
        self.testing=[]
        self.labeling=[]
        self.root=None
        self.listx=[]
    
    class Node:  
        def __init__(self):  
            self.col = None
            self.left = None
            self.right = None
            self.col_value = None
            self.col_name=None
    
    def Sort(self,sub_li): 
        sub_li.sort(key = lambda x: x[0]) 
        return sub_li 
    
    def min_MSE(self,training,main_training):
        min_mm=[]
        for i in training:
    #        print(i)
    #        print(main_training[i])
            if(type(training.iloc[0][i])!=str):
    #            print("fa")
                global_mse=10000000000000000
                coll=0
                coll_value=0
                col_list=training[i].unique()
                col_list.sort()
    #            print(col_list)
                for j in range(0,len(col_list)-1):
                    msee=0
    #                print("ghusega")
    #                print(col_list[j],col_list[j+1])
                    middle=(col_list[j]+col_list[j+1])/2
    #                print("csd")
                    t1=main_training[main_training[i] < middle]
                    t2=main_training[main_training[i] >= middle]
    #                print("phir ghusega")
                    t1_mean=t1['SalePrice'].mean(skipna=True)
                    t2_mean=t2['SalePrice'].mean(skipna=True)
                    t1_mse=0
                    t2_mse=0
                    for k in range(0,len(t1)):
                        diff=t1['SalePrice'].iloc[k]-t1_mean
                        sq_diff=diff**2
                        t1_mse+=sq_diff
                    if(len(t1)>0):
                        t1_mse=t1_mse*len(t1)
                        t1_mse=t1_mse/(len(t1)+len(t2))
                        
                    for k in range(0,len(t2)):
                        diff=t2['SalePrice'].iloc[k]-t2_mean
                        sq_diff=diff**2
                        t2_mse+=sq_diff
                    if(len(t2)>0):
                        t2_mse=t2_mse*len(t2)
                        t2_mse=t2_mse/(len(t1)+len(t2))
                    msee=t1_mse+t2_mse
                    if(global_mse > msee):
                        global_mse = msee
                        coll=i
                        coll_value=middle
#                print(i,global_mse,coll_value)
                list2=[]
                list2.append(global_mse)
                list2.append(coll_value)
                list2.append(coll)
                min_mm.append(list2)
            else:
    #            print("dss")
                global_mse=10000000000000000
                coll=0
                coll_value=0
                col_list=training[i].unique()
                col_list.sort()
                for j in range(0,len(col_list)-1):
                    msee=0
    #                print("ghusega")
                    middle=col_list[j]
                    t1=main_training[main_training[i] == middle]
                    t2=main_training[main_training[i] != middle]
    #                print("phir ghusega")
                    t1_mean=t1['SalePrice'].mean(skipna=True)
                    t2_mean=t2['SalePrice'].mean(skipna=True)
                    t1_mse=0
                    t2_mse=0
                    for k in range(0,len(t1)):
                        diff=t1['SalePrice'].iloc[k]-t1_mean
                        sq_diff=diff**2
                        t1_mse+=sq_diff
                    if(len(t1)>0):
                        t1_mse=t1_mse*len(t1)
                        t1_mse=t1_mse/(len(t1)+len(t2))
                        
                    for k in range(0,len(t2)):
                        diff=t2['SalePrice'].iloc[k]-t2_mean
                        sq_diff=diff**2
                        t2_mse+=sq_diff
                    if(len(t2)>0):
                        t2_mse=t2_mse*len(t2)
                        t2_mse=t2_mse/(len(t1)+len(t2))
                    msee=t1_mse+t2_mse
                    if(global_mse > msee):
                        global_mse = msee
                        coll=i
                        coll_value=middle
#                print(i,global_mse,coll_value)
                list2=[]
                list2.append(global_mse)
                list2.append(coll_value)
                list2.append(coll)
                min_mm.append(list2)
        self.Sort(min_mm)
    #    print(min_mm)
        coll=min_mm[0][2]
        coll_value=min_mm[0][1]
        return coll,coll_value
                        
    def buildTree(self,training,height):
        if(height<=0):
            return None
        training_new=training.drop('SalePrice', axis=1)
        
        coll,coll_value=self.min_MSE(training_new,training)
#        print(coll,coll_value)
        
        if(type(coll_value)==str):
    #        print("e1")
            t1=training[training[coll] == coll_value]
            t2=training[training[coll] != coll_value]
            
        else:
    #        print("e2")
            t1=training[training[coll] < coll_value]
            t2=training[training[coll] >=coll_value]
            
    #   	
    #    t1 = t1.drop(coll , axis='columns')
    #    t2 = t2.drop(coll , axis='columns')
        root=self.Node()
        col_ind=training.columns.get_loc(coll)
        root.col=col_ind
        root.col_name=coll
        root.col_value=coll_value
        if(len(t1)>20):
            root.left=self.buildTree(t1,height-1)
        if(len(t2)>20):
            root.right=self.buildTree(t2,height-1)
        return root
    
    
    def traverse(self,root,training,testing,i):
        coll=root.col
        coll_value=root.col_value
        coll_name=root.col_name
        if(type(coll_value)!=str):
            mask = training[coll_name] < coll_value 
            t1 = training[mask]
            t2 = training[~mask]
            if(root.left is not None and root.right is not None):
                if(root.left is not None and testing.iloc[i,coll]<coll_value):
                    return self.traverse(root.left,t1,testing,i)
                elif(root.right is not None and testing.iloc[i,coll]>=coll_value):
                    return self.traverse(root.right,t2,testing,i)
            else:
    #            print(training['SalePrice'].mean(skipna=True))
                return training['SalePrice'].mean(skipna=True)
    
        else:
            mask = training[coll_name] == coll_value 
            t1 = training[mask]
            t2 = training[~mask] 
            if(root.left is not None and root.right is not None):
                if(root.left is not None and testing.iloc[i,coll]==coll_value):
                    return self.traverse(root.left,t1,testing,i)
                elif(root.right is not None and testing.iloc[i,coll]!=coll_value):
                    return self.traverse(root.right,t2,testing,i)
            else:
    #            print(training['SalePrice'].mean(skipna=True))
                return training['SalePrice'].mean(skipna=True)
        
    
        
            
            
        
        
    def predict1(self,root,testing):
        training=self.training
        testing=self.testing
#        test_id=testing.iloc[0:,0]
        testing=testing.drop(testing.columns[[0]], axis=1)
#        test_labels=testing1.iloc[0,0:]
        for i in self.listx:
            testing=testing.drop(testing.columns[[i]], axis=1)
        for i in testing:
            if(testing[i].dtypes==object):
                testing[i].fillna(testing[i].mode(dropna=True)[0],inplace=True)
            else:
                testing[i].fillna(testing[i].mean(skipna=True),inplace=True)        
        ans=[]
        test=pan.DataFrame(testing).to_numpy()
        for i in range(0,len(test)):
            ans.append(self.traverse(root,training,testing,i))
#        print(ans)
    #    print(len(ans))
        return ans
    
#    def inorder(self,root,height):
#        if(root is None):
#            return
#        
#        print(root.col,root.col_value,height)
#        height+=1
#        if(root.left is not None):
#            inorder(root.left,height)
#        if(root.right is not None):
#            inorder(root.right,height)
#    
    
    def train(self,path):
        training=pan.read_csv(path)
#        testing=pan.read_csv(path1)
#        labeling=pan.read_csv('/home/srg/Desktop/2sem/SMAI/ass1/Datasets/q3/test_labels.csv')
        self.training=training
#        self.testing=testing
        training1=pan.read_csv(path,header=None)
#        testing1=pan.read_csv(path1,header=None)
#        labeling1=pan.read_csv('/home/srg/Desktop/2sem/SMAI/ass1/Datasets/q3/test_labels.csv',header=None)
#        labeling1.drop([0], axis=1,inplace=True)
#        labeling11=pan.DataFrame(labeling1).to_numpy()
        
#        train_id=training.iloc[0:,0]
#        train_cost=training.iloc[0:,80]
        training=training.drop(training.columns[[0]], axis=1)
        
        
#        test_id=testing.iloc[0:,0]
#        testing=testing.drop(testing.columns[[0]], axis=1)
#        test_labels=testing1.iloc[0,0:]
        listx=[]
        for i in range(0,training.shape[1]):
            if(training.iloc[0:,i].isnull().sum()>500):
                listx.append(i)
        self.listx=listx[::-1]
        for i in listx:
            training=training.drop(training.columns[[i]], axis=1)
#            testing=testing.drop(testing.columns[[i]], axis=1)
#        train_labels=training1.iloc[0,0:]
        
        for i in training:
            if(training[i].dtypes==object):
                training[i].fillna(training[i].mode(dropna=True)[0],inplace=True)
            else:
                training[i].fillna(training[i].mean(skipna=True),inplace=True)
                
#        
#        for i in testing:
#            if(testing[i].dtypes==object):
#                testing[i].fillna(testing[i].mode(dropna=True)[0],inplace=True)
#            else:
#                testing[i].fillna(testing[i].mean(skipna=True),inplace=True)
#        print("ok")
        self.root=self.buildTree(training,8)
        
    def predict(self,path):
        
        testing=pan.read_csv(path)
        self.testing=testing
        return self.predict1(self.root,testing)
    