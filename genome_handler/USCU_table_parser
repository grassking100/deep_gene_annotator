#Purpose:Parse file from USCU table
#Input:USCU table file path
#Output:Array of dictionary which stored transcription,tranlation position(zero-based),and region start and end index(zero-based)
class USCU_TableParser:
    def __init__(self,file):
        ######
        #all the index will be convert to zero-based
        #For more details about zero-based and one-based,please checkup the following link
        #http://bedtools.readthedocs.io/en/latest/content/overview.html
        ######
        self.__INFORMATION_KEYS=['chrom','strand','txStart','txEnd','cdsStart','cdsEnd','exonCount','exonStarts','exonEnds','name2','cdsStartStat','cdsEndStat']
        lines=[]
        self.__plus_strand=0
        self.__minus_strand=0
        self.__region_number=0
        self.__regions_information={}
        self.__names=[]
        with open(file,"r") as gtf_file:
            for line in gtf_file:
                lines+=[line]
        self.__header=lines[0].split("\t")
        self.__name_index=self.__header.index('name')
        self.__exon_count_index=self.__header.index('exonCount')
        for line in lines[1:]:
            self.__names.append(line.split("\t")[self.__name_index])
        for line in lines[1:]:
            self.__set_region_information(line)

    def get_index(self,key):
        return self.__INFORMATION_KEYS.index(key)
    @property
    def names(self):
        return self.__names
    @property
    def region_number(self):
        return self.__region_number
    @property
    def plus_strand_number(self):
        return self.__plus_strand
    @property
    def minus_strand_number(self):
        return self.__minus_strand
    def __set_region_information(self,information_text):
        information_list=information_text.split("\t")
        selected_information_list=[]
        self.__region_number+=1
        name=information_list[self.__name_index]
        exon_count=int(information_list[self.__exon_count_index])
        for key in self.__INFORMATION_KEYS:
            index=self.__header.index(key)
            selected_information_list.append(information_list[index])
        value_int_list=['txStart','txEnd','cdsStart','cdsEnd','exonCount']
        for key in value_int_list:
            index=self.get_index(key)
            selected_information_list[index]=int(selected_information_list[index])
        strand=selected_information_list[self.get_index('strand')]
        if strand=="+":
            self.__plus_strand+=1
        elif strand=="-":
            self.__minus_strand+=1
        else:
            raise Exception("Unexpected strand presentation:"+str(strand))
        for key in ['exonStarts','exonEnds']:
            key_index=self.get_index(key)
            temp=selected_information_list[key_index].split(",")[:exon_count]
            shift=(int)(key=='exonEnds')
            selected_information_list[key_index]=[int(t)-shift for t in temp]
        selected_information_list[self.get_index('txEnd')]-=1
        selected_information_list[self.get_index('cdsEnd')]-=1
        self.__regions_information[name]=selected_information_list
    def __get_raw_data(self,name):
        return self.__regions_information[name]
    def get_data(self,name):
        raw_data=self.__get_raw_data(name)
        dictionary=self.__parse_from_single_line(raw_data)
        dictionary['name']=raw_data[self.__name_index]
        return dictionary
    def get_data_value(self,name,attribute):
        return self.get_data(name)[attribute]
    def __parse_from_single_line(self,row_of_data):
        dictionary={}
        for i in range(len(self.__INFORMATION_KEYS)):
            dictionary[self.__INFORMATION_KEYS[i]]=row_of_data[i]
        return dictionary
