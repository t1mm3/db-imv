#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import csv
from sys import argv
import os
import operator
from functools import reduce

def get_all_file_path(dir_path):
	dir_list=[]
	dir_list.append(dir_path)
	file_list=[]
	while len(dir_list)!=0:
		path=dir_list.pop(0)
		files=os.listdir(path)
		for f in files:
			file_path=path+'/'+f
			if(os.path.isdir(file_path)):
				dir_list.append(file_path)
			if(os.path.isfile(file_path)):
				file_list.append(file_path)
	file_list=sorted(file_list,key=lambda x: os.path.getmtime(os.path.join(file_path,x)))
	return file_list

def merge(dir_path,output_file):
	if os.path.exists(output_file):
		return
	file_path_set=get_all_file_path(dir_path)
	data=[['dis','SIMDstatesize','scalarstatesize','thread_num','r_size','s_size','r_skew','s_skew',' ','Naive','SIMD','DVA','FVA','AMAC','IMV']]
	for file_path in file_path_set:
		file_name=file_path.split('/')[-1]
		file_name_suffex=file_name.split('.')[-1]
		if file_name_suffex!='csv':
			continue
		try:
			f=open(file_path,'r')
			reader=csv.reader(f)
			data_group=[]
			for line in reader:
				data_row=[]
				for j in np.arange(1,len(line)):
					string=line[j]
					data_row.append(string)
				data_group.append(data_row)
			data.append(data_group)
		finally:
			if f:
				f.close()
	csv_writer=csv.writer(open(output_file,'a'))
	results=map(list,zip(*data))
#	print(results)
	for arr in results:
		item=[]
		item.append(arr[0])
		index=0
		for it in arr:
			if index==0:
				index=index+1
				continue
			for i in it:
				item.append(i)
		csv_writer.writerow(item)

if __name__=='__main__':
	working_path=os.path.abspath(os.curdir)
	merge(working_path+'/'+argv[1],working_path+'/'+argv[1]+'/'+argv[2])
