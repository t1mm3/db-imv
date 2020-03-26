#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import csv
from sys import argv

def parser(i_file, o_file,repeat_num):
	try:
#		f=open('test.txt','r')
		f=open(i_file,'r')
		lines=f.readlines()
		repeat=0
#		csv_writer=csv.writer(open('results.csv','a'))
		csv_writer=csv.writer(open(o_file,'a'))
		count=0
		results=[['dis','SIMDstatesize','scalarstatesize','thread_num','r_size','s_size','r_skew','s_skew',' ','Naive','SIMD','DVA','FVA','AMAC','IMV']]
		result_row=[]
		numbers=[]	
		starts=False
		for line in lines:
			if line.startswith('!!!'):
				starts=True
                		results=map(list,zip(*results))
		                for arr in results:
#               		        print(arr)
		                        csv_writer.writerow(arr)
				csv_writer.writerow([])
				csv_writer.writerow([])
				results=[['dis','SIMDstatesize','scalarstatesize','thread_num','r_size','s_size','r_skew','s_skew',' ','Naive','SIMD','DVA','FVA','AMAC','IMV']]
			if line.startswith('ARGS'):
				args=line.strip().split(' ')
				result_row.append(args[2])
				result_row.append(args[4])
				result_row.append(args[6])
				result_row.append(args[8])
				result_row.append(args[10])
				result_row.append(args[12])
				result_row.append(args[14])
				result_row.append(args[16])
				result_row.append(' ')
			if line.startswith('total'):
#				print(line.split(' ')[-1].strip())
				if repeat<repeat_num:
					numbers.append(float(line.split(' ')[-1].strip()))
					repeat=repeat+1
#					print('repeat',repeat)
					if repeat == repeat_num:
						repeat=0
						count=count+1
						nparr=np.asarray(numbers)
						min_=np.min(nparr)
						max_=np.max(nparr)
						print('nparr',nparr)
						numbers=[]
						print(min_,max_)
#						if max_-min_<500:
#							result_row.append(min_)
#						else:
#							print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
#							min_=-min_
#							result_row.append(min_)
						result_row.append(min_)
						print("===================")
						if count==6:
							results.append(result_row)
#							csv_writer.writerow(result_row)
							result_row=[]
							count=0
		if not starts:
			results=map(list,zip(*results))
			for arr in results:
				print(arr)
				csv_writer.writerow(arr)
	finally:
		if f:
			f.close()

if __name__=='__main__':
	parser(argv[1],argv[2],int(argv[3]))
