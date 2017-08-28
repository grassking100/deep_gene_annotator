from Fasta_handler.Fasta_handler import *
from time import gmtime, strftime
import random
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio.Alphabet.IUPAC import IUPACAmbiguousDNA
from datetime import datetime
import configparser
import argparse
class Data_index_selecter:
	def __init__(self,min_len,max_len,input_file,shuffle,fractions,selected_limit,include_single_exon):
		self.__min_len=min_len
		self.__max_len=max_len
		self.__input_file=input_file
		self.__shuffle=shuffle
		if sum(fractions)!=1:
			raise "Fractions sum are not equal to one"
		self.__fractions=fractions
		self.__include_single_exon=include_single_exon
	def get_fraction_member_indice(self):
		(self.__names,self.__seqs)=fastas2arr(self.input_file)
		selected_seq_indices=seqs_index_selector(self.seqs,self.min_len,self.max_len,self.include_single_exon)
		if self.shuffle:
			random.shuffle(selected_seq_indices)
		fraction_member_indice=[]
		start_index=0
		if selected_limit>0:
			selected_seq_indices=selected_seq_indices[:selected_limit]
		self.__selected_number=len(selected_seq_indices)
		for fraction in fractions:
			end_index=start_index+int(fraction*self.__selected_number)
			fraction_member_indice+=[selected_seq_indices[start_index:end_index]]
			start_index=end_index
		return fraction_member_indice
	@property
	def selected_number(self):
		return self.__selected_number
	@property
	def include_single_exon(self):
		return self.__include_single_exon
	@property
	def names(self):
		return self.__names
	@property
	def seqs(self):
		return self.__seqs		
	@property
	def min_len(self):
		return self.__min_len
	@property
	def max_len(self):
		return self.__max_len
	@property
	def input_file(self):
		return self.__input_file
	@property
	def shuffle(self):
		return self.__shuffle
	@property
	def fractions(self):
		return self.__fractions

if __name__=='__main__':
	parser = argparse.ArgumentParser(description="Program will split data to desired partition")
	arguments=[{"prefix":'i',"full_name":'input_file',"description":'Fasta file\'s path',"required":True},
		{"prefix":'min',"full_name":'min',"description":'Minimum length to selected',"required":True},
		{"prefix":'max',"full_name":'max',"description":'Maximum length to selected',"required":True},
		{"prefix":'s',"full_name":'shuffle',"description":'Use shuffle',"required":True},
		{"prefix":'f',"full_name":'fractions',"description":'A list of number separated by comma to indicate fractions to split',"required":True},
		{"prefix":'l',"full_name":'selected_limit',"description":'Choose maximum number to selected',"required":False},
		{"prefix":'n',"full_name":'notes',"description":'Notes to wrote in exported file',"required":False},
		{"prefix":'e',"full_name":'include_single_exon',"description":'Choose to selected single exon or not',"required":True},
		{"prefix":'o',"full_name":'output_file',"description":'Export file to the path',"required":True}]
	for argument in arguments:
		parser.add_argument("-"+argument["prefix"],"--"+argument["full_name"],help=argument["description"], required=argument["required"])
	args = parser.parse_args()
	selected_limit=-1
	if args.selected_limit is not None:
		selected_limit=int(args.selected_limit)
	fractions=[float(fraction) for fraction in args.fractions.split(",")]
	shuffle=args.shuffle=="True"
	include_single_exon=args.include_single_exon=="True"
	selecter=Data_index_selecter(int(args.min),int(args.max),args.input_file,shuffle,fractions,selected_limit,bool(args.include_single_exon))
	fraction_member_indice=selecter.get_fraction_member_indice()
	fraction_id=1
	fraction_number=len(fractions)
	include_single_exon_prefix="" if include_single_exon else "not"
	shuffle_prefix="" if shuffle else "not"
	notes=args.notes if args.notes is not None else ""
	print("Total number:"+str(len(selecter.seqs)))
	print("Selected number:"+str(selecter.selected_number))
	for indice in fraction_member_indice: 
		output_seqs=[]
		description_to_export=notes+",length between "+str(selecter.min_len)+" to "+str(selecter.max_len)+"(inclusive),"+include_single_exon_prefix+" include single exon,and "+shuffle_prefix+" stored with random order"
		for index in indice:
			record = SeqRecord(Seq(selecter.seqs[index],IUPACAmbiguousDNA()),
							id=selecter.names[index],
                            description=description_to_export)
			output_seqs.append(record)
		include_single_exon_notes=include_single_exon_prefix
		if include_single_exon_prefix:
			include_single_exon_notes+="_"
		include_single_exon_notes+="include_single_exon_"
		file_name=args.output_file+datetime.now().strftime('%Y_%b_%d')+"_len_"+str(selecter.min_len)+"_"+str(selecter.max_len)+"_inclusive_"+include_single_exon_notes+str(fraction_id)+"_of_"+str(fraction_number)+".fasta"
		print("Export file:"+file_name)
		with open(file_name, "w") as output_handle:
			SeqIO.write(output_seqs, output_handle, "fasta")
		fraction_id+=1
	print("End of program")